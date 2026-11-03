import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 140  # 10s safety margin before 150s kill

def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    Vectorized using FFT for speed.
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs

def _score(h):
    """
    Compute the maximum linear overlap.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))

def _smooth_max_grad(h, beta):
    """
    Gradient of the smooth maximum objective (LogSumExp).
    Returns (value, gradient)
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    shifted = beta * (cs - m)
    shifted = np.clip(shifted, -500, 500)
    exp_terms = np.exp(shifted)
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp
    val = m + np.log(sum_exp) / beta
    
    grad = np.zeros(n)
    
    # C_k = 1 - (2/n) * r_k  (Note: 2/n * n/2 = 1, so C_k = 1 - 2/n * R_k)
    # Actually C_k = (2/n)(n/2 - R_k) = 1 - (2/n)R_k
    # dC_k/dh_j = -(2/n) * dR_k/dh_j
    
    for k in range(n):
        if w[k] < 1e-10:
            continue
        factor = -(2.0 / n) * w[k]
        
        if k > 0:
            grad[k:] += factor * h[:n-k]
        if k < n:
            grad[:n-k] += factor * h[k:]
            
    return val, grad

def _project_sum_box(y, target_sum):
    """
    Project y onto {x : 0 <= x_i <= 1, sum(x) = target_sum}
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
        
    for _ in range(50):
        mid = (lo + hi) / 2.0
        if g(mid) > 0:
            hi = mid
        else:
            lo = mid
            
    tau = (lo + hi) / 2.0
    return np.clip(y + tau, 0.0, 1.0)

def _nesterov_pgd(h0, beta, steps, lr, deadline):
    """
    Nesterov Accelerated Projected Gradient Descent.
    """
    h = h0.copy()
    target_sum = N / 2.0
    current_lr = lr
    y = h.copy()
    t = 1.0
    prev_val, _ = _smooth_max_grad(h, beta)
    
    for i in range(steps):
        if time.time() > deadline:
            break
            
        val, grad = _smooth_max_grad(y, beta)
        
        step_lr = current_lr
        new_h = y - step_lr * grad
        new_h_proj = _project_sum_box(new_h, target_sum)
        new_val, _ = _smooth_max_grad(new_h_proj, beta)
        
        while new_val > prev_val and step_lr > 1e-8:
            step_lr /= 2.0
            new_h = y - step_lr * grad
            new_h_proj = _project_sum_box(new_h, target_sum)
            new_val, _ = _smooth_max_grad(new_h_proj, beta)
            
        if new_val < prev_val:
            t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
            y = new_h_proj + ((t - 1.0) / t_new) * (new_h_proj - h)
            h = new_h_proj
            t = t_new
            prev_val = new_val
            if i % 10 == 0:
                current_lr = min(current_lr * 1.05, 0.1)
        else:
            current_lr = max(current_lr * 0.5, 1e-6)
            
    return h

def _coordinate_polish(h, deadline):
    """
    Coordinate descent polish to refine the solution.
    """
    h = h.copy()
    target_sum = N / 2.0
    n = len(h)
    improved = True
    max_iters = 200
    iters = 0
    
    while improved and iters < max_iters and time.time() < deadline:
        improved = False
        iters += 1
        for j in range(n):
            if time.time() > deadline:
                break
            
            # Try small perturbations in coordinate j
            base_val = _score(h)
            best_val = base_val
            best_delta = 0.0
            
            for delta in [0.01, 0.02, 0.05, 0.1, 0.2, -0.01, -0.02, -0.05, -0.1, -0.2]:
                h_trial = h.copy()
                h_trial[j] += delta
                h_trial = _project_sum_box(h_trial, target_sum)
                val = _score(h_trial)
                if val < best_val - 1e-8:
                    best_val = val
                    best_delta = delta
            
            if best_delta != 0.0:
                h[j] += best_delta
                h = _project_sum_box(h, target_sum)
                improved = True
                
    return h

def _refine_with_lbfgs(h0, beta, deadline):
    """
    Refine using L-BFGS-B on the smooth max objective.
    """
    def objective_and_grad(x):
        x = np.clip(x, 0.0, 1.0)
        val, grad = _smooth_max_grad(x, beta)
        return val, grad
    
    x0 = h0.copy()
    result = minimize(
        objective_and_grad,
        x0,
        method='L-BFGS-B',
        jac=True,
        bounds=[(0.0, 1.0)] * N,
        options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
    )
    
    h_opt = result.x
    h_opt = _project_sum_box(h_opt, N / 2.0)
    return h_opt

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
            
    # 1. Initial Candidates
    i = np.arange(N)
    
    # Uniform
    h_unif = np.full(N, 0.5)
    _update_best(h_unif)
    
    # Chirps (Quadratic Phase)
    for alpha in [0.1, 0.2, 0.5, 1.0, 2.0]:
        for phase in [0, np.pi/2]:
            h_chirp = 0.5 + 0.5 * np.cos(2.0 * np.pi * alpha * (i**2) / N + phase)
            h_chirp = _project_sum_box(h_chirp, N / 2.0)
            _update_best(h_chirp)

    # Linear Frequencies
    for freq in [1, 2, 3, 5, 10]:
        h_lin = 0.5 + 0.5 * np.cos(2.0 * np.pi * freq * i / N)
        h_lin = _project_sum_box(h_lin, N / 2.0)
        _update_best(h_lin)
        
    # Random
    for _ in range(10):
        h_rand = np.random.uniform(0, 1, N)
        h_rand = _project_sum_box(h_rand, N / 2.0)
        _update_best(h_rand)

    # 2. Optimization Phases
    # Phase 1: Coarse Nesterov
    if best_h is not None and time.time() - t0 < _DEADLINE - 70:
        h_opt = _nesterov_pgd(best_h, beta=10.0, steps=300, lr=0.05, deadline=_DEADLINE - 70)
        _update_best(h_opt)
        
    # Phase 2: Medium Nesterov
    if best_h is not None and time.time() - t0 < _DEADLINE - 40:
        h_opt = _nesterov_pgd(best_h, beta=50.0, steps=300, lr=0.01, deadline=_DEADLINE - 40)
        _update_best(h_opt)
        
    # Phase 3: L-BFGS refinement
    if best_h is not None and time.time() - t0 < _DEADLINE - 20:
        h_opt = _refine_with_lbfgs(best_h, beta=100.0, deadline=_DEADLINE - 20)
        _update_best(h_opt)
        
    # Phase 4: High-beta Nesterov polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 10:
        h_opt = _nesterov_pgd(best_h, beta=300.0, steps=200, lr=0.002, deadline=_DEADLINE - 10)
        _update_best(h_opt)
        
    # Phase 5: Coordinate polish
    if best_h is not None and time.time() - t0 < _DEADLINE - 3:
        h_opt = _coordinate_polish(best_h, deadline=_DEADLINE - 3)
        _update_best(h_opt)
        
    if best_h is None:
        best_h = np.full(N, 0.5)
        
    best_h = _project_sum_box(best_h, N / 2.0)
    return best_h
