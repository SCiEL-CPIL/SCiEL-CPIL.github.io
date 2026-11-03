"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Combined approach:
- Chirp-structured initializations (quadratic phase) that inherently have low autocorrelation.
- Nesterov accelerated projected gradient descent with LogSumExp smooth surrogate.
- L-BFGS-B refinement with smooth objective.
- Coordinate descent polish on active set for final precision.
- Circular overlap (matching task spec/seed).
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 145.0  # 5s safety margin before 150s kill


def _circular_overlap_values(h):
    """
    Compute O(s) = sum_i h[i] * (1 - h[(i+s)%N]) for all s.
    O(s) = N/2 - R_circ(s)
    R_circ(s) = sum_i h[i] h[(i+s)%N]
    """
    n = len(h)
    H = np.fft.fft(h)
    R_circ = np.fft.ifft(H * np.conj(H)).real
    O = n / 2.0 - R_circ
    return O


def _score(h):
    """Compute the maximum overlap."""
    O = _circular_overlap_values(h)
    return float(np.max(O))


def _smooth_max_val_grad(h, beta):
    """
    Return (value, gradient) of LogSumExp(beta * O(h)).
    O(h) is the vector of overlaps.
    """
    n = len(h)
    O = _circular_overlap_values(h)
    max_O = np.max(O)
    exp_terms = np.exp(beta * (O - max_O))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp
    
    val = max_O + np.log(sum_exp) / beta
    
    # Gradient computation via circular correlation
    # grad_O[j] = - sum_s w_s * d/dh_j R_circ(s)
    # d/dh_j R_circ(s) = h[(j-s)%n] + h[(j+s)%n] for s != 0, and 2*h[j] for s=0
    # Since w is symmetric (O is symmetric), we can use FFT
    W = np.fft.fft(w)
    H = np.fft.fft(h)
    conv1 = np.fft.ifft(W * H).real  # sum_s w_s h[(j-s)%n]
    # For symmetric w, sum_s w_s h[(j+s)%n] = conv1[j]
    # But we need to handle s=0 separately: it contributes w[0]*2*h[j]
    # conv1 includes w[0]*h[j] from s=0 term
    # So grad_O = -(conv1 + conv1 - w[0]*h) = -(2*conv1 - w[0]*h)
    grad_O = -(2.0 * conv1 - w[0] * h)
    
    return val, grad_O


def _project_sum_box(y, target_sum):
    """
    Project y onto {x : 0 <= x_i <= 1, sum(x) = target_sum}
    Using binary search on the shift parameter tau.
    """
    lo = np.min(y) - 1.0
    hi = np.max(y) + 1.0
    
    def g(tau):
        x = np.clip(y + tau, 0.0, 1.0)
        return np.sum(x) - target_sum
        
    if g(lo) > 0:
        return np.clip(y + lo, 0.0, 1.0)
    if g(hi) < 0:
        return np.clip(y + hi, 0.0, 1.0)
        
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if g(mid) > 0:
            hi = mid
        else:
            lo = mid
            
    tau = (lo + hi) / 2.0
    return np.clip(y + tau, 0.0, 1.0)


def _nesterov_pgd(h0, beta, steps, lr, deadline):
    """
    Nesterov-accelerated Projected Gradient Descent.
    """
    h = _project_sum_box(h0, N / 2.0)
    y = h.copy()
    target_sum = N / 2.0
    prev_val, _ = _smooth_max_val_grad(h, beta)
    t = 1.0
    
    for i in range(steps):
        if time.time() > deadline:
            break
            
        # Nesterov step
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = h + (t / t_new) * (h - y)
        
        val, grad = _smooth_max_val_grad(y, beta)
        
        # Gradient step from y
        new_h = y - lr * grad
        new_h_proj = _project_sum_box(new_h, target_sum)
        new_val, _ = _smooth_max_val_grad(new_h_proj, beta)
        
        # Backtracking if needed
        step_lr = lr
        while new_val > prev_val + 1e-12 and step_lr > 1e-10:
            step_lr /= 2.0
            new_h = y - step_lr * grad
            new_h_proj = _project_sum_box(new_h, target_sum)
            new_val, _ = _smooth_max_val_grad(new_h_proj, beta)
            
        if new_val < prev_val:
            h = new_h_proj
            prev_val = new_val
            t = t_new
        else:
            # Reset momentum if no progress
            y = h.copy()
            t = 1.0
            if step_lr < 1e-10:
                break
                
    return h


def _refine_with_lbfgs(h0, deadline):
    """
    Refine a solution using L-BFGS-B with the smooth objective.
    """
    h = _project_sum_box(h0, N / 2.0)
    target_sum = N / 2.0
    
    def obj_and_grad(x):
        val, grad = _smooth_max_val_grad(x, beta=500.0)
        return val, grad
        
    try:
        res = minimize(
            obj_and_grad,
            h,
            jac=True,
            method='L-BFGS-B',
            bounds=[(0, 1)] * N,
            options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
        )
        if res.fun < _score(h):
            h_new = _project_sum_box(res.x, target_sum)
            return h_new
    except Exception:
        pass
        
    return h


def _coordinate_descent_polish(h0, deadline, max_iter=200):
    """
    Coordinate descent polish on the active set of constraints.
    """
    h = _project_sum_box(h0, N / 2.0)
    target_sum = N / 2.0
    current_score = _score(h)
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
            
        cs = _circular_overlap_values(h)
        max_val = np.max(cs)
        active = np.where(cs > max_val - 0.02)[0]
        
        improved = False
        for j in range(N):
            for delta in [0.001, -0.001, 0.0005, -0.0005, 0.002, -0.002]:
                h_trial = h.copy()
                h_trial[j] += delta
                h_trial = _project_sum_box(h_trial, target_sum)
                s = _score(h_trial)
                if s < current_score - 1e-12:
                    h = h_trial
                    current_score = s
                    improved = True
                    break
            if time.time() > deadline:
                break
        if not improved:
            break
            
    return h


def _chirp_initialization(sweep_rate, phase_offset=0.0):
    """
    Generate a chirp-based initialization with quadratic phase.
    h[i] = 0.5 + 0.5 * sin(2*pi*(sweep_rate * i^2 / N + phase_offset))
    This naturally has low autocorrelation properties.
    """
    i = np.arange(N)
    phase = 2.0 * np.pi * (sweep_rate * i**2 / N + phase_offset)
    h = 0.5 + 0.5 * np.sin(phase)
    return _project_sum_box(h, N / 2.0)


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _project_sum_box(h, N / 2.0)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
            
    i = np.arange(N)
    
    # Diverse initial guesses with chirp structures
    starts = []
    
    # 1. Constant
    starts.append(np.full(N, 0.5))
    
    # 2. Cosine
    starts.append(_project_sum_box(0.5 + 0.5 * np.cos(2.0 * np.pi * i / N), N / 2.0))
    
    # 3. Sine
    starts.append(_project_sum_box(0.5 + 0.5 * np.sin(2.0 * np.pi * i / N), N / 2.0))
    
    # 4. Chirp with various sweep rates
    for sweep in [0.5, 1.0, 1.5, 2.0, 3.0]:
        for phase in [0.0, 0.25, 0.5]:
            starts.append(_chirp_initialization(sweep, phase))
    
    # 5. Block pattern
    h_block = np.zeros(N)
    for k in range(N):
        h_block[k] = 1.0 if (k % 8) < 4 else 0.0
    starts.append(_project_sum_box(h_block, N / 2.0))
    
    # 6. Random
    for _ in range(5):
        starts.append(_project_sum_box(np.random.uniform(0, 1, N), N / 2.0))
    
    # Evaluate initial guesses
    for h0 in starts:
        _update_best(h0)
        
    # Iterative optimization with increasing beta
    betas = [5.0, 20.0, 50.0, 100.0, 200.0, 500.0]
    
    for beta in betas:
        if time.time() > _DEADLINE - 15.0:
            break
            
        # Optimize from current best
        h_opt = _nesterov_pgd(best_h, beta, steps=400, lr=0.02 / np.sqrt(beta), deadline=_DEADLINE - 15.0)
        _update_best(h_opt)
        
        # Also try from a perturbed version of best
        h_pert = best_h + 0.05 * np.random.randn(N)
        h_pert = _project_sum_box(h_pert, N / 2.0)
        h_opt2 = _nesterov_pgd(h_pert, beta, steps=200, lr=0.02 / np.sqrt(beta), deadline=_DEADLINE - 15.0)
        _update_best(h_opt2)
        
        # If time permits, try from chirp starts with high beta
        if time.time() < _DEADLINE - 30.0:
            for h0 in starts[3:8]:  # Chirp starts
                h_opt3 = _nesterov_pgd(h0, beta, steps=150, lr=0.02 / np.sqrt(beta), deadline=_DEADLINE - 30.0)
                _update_best(h_opt3)

    # L-BFGS-B Polish
    if best_h is not None and time.time() < _DEADLINE - 10.0:
        h_opt = _refine_with_lbfgs(best_h, _DEADLINE - 10.0)
        _update_best(h_opt)
        
    # Coordinate descent polish for final precision
    if best_h is not None and time.time() < _DEADLINE - 5.0:
        h_opt = _coordinate_descent_polish(best_h, _DEADLINE - 5.0, max_iter=100)
        _update_best(h_opt)

    if best_h is None:
        best_h = np.full(N, 0.5)
        
    return best_h
