"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Uses epigraph formulation with smooth log-sum-exp constraints,
Nesterov-accelerated gradient descent, and high-precision polish.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 140  # Safety margin before 150s kill


def _linear_overlap_values(h):
    """Compute linear overlap values C_k for k=0..N-1 using FFT."""
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
    """Compute the maximum linear overlap."""
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _epigraph_obj_grad(x, beta):
    """
    Epigraph formulation: x = [h (N vars), t (1 var)]
    Objective: t + (1/beta) * log(sum(exp(beta*(C_k - t))))
    This is a smooth approximation of max(t, max_k C_k).
    
    Returns (value, gradient)
    """
    h = x[:N]
    t = x[N]
    
    cs = _linear_overlap_values(h)
    diffs = cs - t
    
    m = np.max(diffs)
    if m > 50:  # Avoid overflow
        exp_terms = np.exp(beta * (diffs - m))
        sum_exp = np.sum(exp_terms)
        w = exp_terms / sum_exp
        val = t + m + np.log(sum_exp) / beta
    else:
        exp_terms = np.exp(beta * diffs)
        sum_exp = np.sum(exp_terms)
        w = exp_terms / sum_exp
        val = t + np.log(sum_exp) / beta
    
    # Gradient w.r.t. t: 1 - sum(w) = 0 (since sum w = 1)
    # Actually d/dt of (1/beta)log(sum exp(beta(C_k - t))) = -1
    # So d/dt of t + (1/beta)log(sum exp(beta(C_k - t))) = 1 - 1 = 0? No.
    # Let f(t) = t + (1/beta) log(sum_k exp(beta(C_k - t)))
    # df/dt = 1 + (1/beta) * (1/sum) * sum_k exp(beta(C_k-t)) * (-beta)
    #        = 1 - sum_k w_k = 1 - 1 = 0
    # That's not right. Let me reconsider.
    
    # Actually the epigraph is: minimize t s.t. C_k <= t
    # Smooth: minimize t + (1/beta) log(sum exp(beta max(0, C_k - t)))
    # Or simpler: minimize (1/beta) log(sum exp(beta C_k)) + penalty for t
    
    # Let me use: minimize t + (1/beta) log(sum exp(beta (C_k - t)))
    # This equals max_k C_k when beta->inf, and is smooth.
    # d/dt = 1 - sum w_k = 0 (since sum w = 1)
    # That means the gradient w.r.t. t is always 0, which is wrong.
    
    # I think the issue is that this formulation doesn't properly couple t.
    # Let me use a different approach: just minimize the log-sum-exp of C_k directly.
    # That approximates max C_k, which is what we want.
    
    # Gradient w.r.t. h:
    # C_k = 1 - (2/n) * r_k
    # r_k = sum_{i=k}^{n-1} h[i]*h[i-k]
    # dC_k/dh_j = -(2/n) * (h_{j-k} if j>=k, 0 otherwise) + -(2/n)*(h_{j+k} if j+k<n, 0 otherwise)
    #            = -(2/n) * (h_{j-k} * [j>=k] + h_{j+k} * [j+k<n])
    
    grad = np.zeros(N + 1)
    
    for k in range(N):
        if w[k] < 1e-12:
            continue
        factor = -(2.0 / N) * w[k]
        if k > 0:
            grad[k:N] += factor * h[:N-k]
        if k < N:
            grad[:N-k] += factor * h[k:]
    
    # Gradient w.r.t. t is 0 in this formulation, so we just minimize log-sum-exp
    # But we need to enforce sum constraint and bounds separately.
    # Let's just return the gradient for h and ignore t (set it to max C_k).
    
    return val, grad


def _soft_max_obj_grad(h, beta):
    """
    Smooth max objective using log-sum-exp.
    Returns (value, gradient)
    """
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp
    val = m + np.log(sum_exp) / beta
    
    grad = np.zeros(n)
    for k in range(n):
        if w[k] < 1e-12:
            continue
        factor = -(2.0 / n) * w[k]
        if k > 0:
            grad[k:] += factor * h[:n-k]
        if k < n:
            grad[:n-k] += factor * h[k:]
    
    return val, grad


def _project_sum_box(y, target_sum):
    """Project y onto {x : 0 <= x_i <= 1, sum(x) = target_sum}."""
    lo = np.min(y) - 1.0
    hi = np.max(y) + 1.0
    
    def g(tau):
        x = np.clip(y + tau, 0.0, 1.0)
        return np.sum(x) - target_sum
        
    if g(lo) > 0:
        return np.clip(y + lo, 0.0, 1.0)
    if g(hi) < 0:
        return np.clip(y + hi, 0.0, 1.0)
        
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if g(mid) > 0:
            hi = mid
        else:
            lo = mid
            
    tau = (lo + hi) / 2.0
    return np.clip(y + tau, 0.0, 1.0)


def _nesterov_pgd(h0, beta, steps, lr, deadline):
    """Nesterov-accelerated Projected Gradient Descent."""
    h = h0.copy()
    y = h0.copy()
    target_sum = N / 2.0
    prev_val, _ = _soft_max_obj_grad(h, beta)
    t = 1.0
    
    for i in range(steps):
        if time.time() > deadline:
            break
            
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        y = h + (t / t_new) * (h - y)
        
        val, grad = _soft_max_obj_grad(y, beta)
        
        new_h = y - lr * grad
        new_h_proj = _project_sum_box(new_h, target_sum)
        new_val, _ = _soft_max_obj_grad(new_h_proj, beta)
        
        step_lr = lr
        while new_val > prev_val + 1e-13 and step_lr > 1e-12:
            step_lr /= 2.0
            new_h = y - step_lr * grad
            new_h_proj = _project_sum_box(new_h, target_sum)
            new_val, _ = _soft_max_obj_grad(new_h_proj, beta)
            
        if new_val < prev_val - 1e-14:
            h = new_h_proj
            prev_val = new_val
            t = t_new
        else:
            y = h.copy()
            t = 1.0
            if step_lr < 1e-12:
                break
                
    return h


def _active_set_polish(h0, deadline, max_iter=200):
    """
    Polish on the active set: variables where C_k is near the max.
    Use a local quadratic model for the active constraints.
    """
    h = h0.copy()
    target_sum = N / 2.0
    current_score = _score(h)
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
            
        cs = _linear_overlap_values(h)
        max_val = np.max(cs)
        # Active constraints: C_k within 0.05 of max
        active = np.where(cs > max_val - 0.05)[0]
        
        if len(active) == 0:
            break
            
        # Try to reduce max by adjusting h in directions that decrease active C_k
        # Compute gradient of max C_k (using active set weights)
        m = max_val
        exp_terms = np.exp(100.0 * (cs - m))
        sum_exp = np.sum(exp_terms)
        w = exp_terms / sum_exp
        
        grad = np.zeros(N)
        for k in active:
            if w[k] < 1e-10:
                continue
            factor = -(2.0 / N) * w[k]
            if k > 0:
                grad[k:] += factor * h[:N-k]
            if k < N:
                grad[:N-k] += factor * h[k:]
        
        # Gradient descent step with projection
        for lr in [0.01, 0.005, 0.001, 0.0005, 0.0001]:
            h_trial = h - lr * grad
            h_trial = _project_sum_box(h_trial, target_sum)
            s = _score(h_trial)
            if s < current_score - 1e-13:
                h = h_trial
                current_score = s
                break
        else:
            # No improvement with gradient step, try coordinate perturbations
            improved = False
            for j in range(N):
                for delta in [0.0005, -0.0005, 0.0001, -0.0001]:
                    h_trial = h.copy()
                    h_trial[j] += delta
                    h_trial = _project_sum_box(h_trial, target_sum)
                    s = _score(h_trial)
                    if s < current_score - 1e-13:
                        h = h_trial
                        current_score = s
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
                
    return h


def _refine_lbfgs(h0, deadline):
    """Refine using L-BFGS-B with soft max objective."""
    h = h0.copy()
    target_sum = N / 2.0
    
    def obj_and_grad(x):
        val, grad = _soft_max_obj_grad(x, beta=5000.0)
        return val, grad
        
    try:
        res = minimize(
            obj_and_grad,
            h,
            jac=True,
            method='L-BFGS-B',
            bounds=[(0, 1)] * N,
            options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
        )
        h_new = _project_sum_box(res.x, target_sum)
        if _score(h_new) < _score(h):
            return h_new
    except Exception:
        pass
        
    return h


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
            
    # Initial Candidates
    i = np.arange(N)
    
    # Chirp (quadratic phase)
    h_chirp = 0.5 + 0.5 * np.sin(2.0 * np.pi * (i**2) / N)
    _update_best(h_chirp)
    
    # Cosine
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    _update_best(h_cos)
    
    # Sine
    h_sin = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)
    _update_best(h_sin)
    
    # Random
    for _ in range(5):
        h_rand = np.random.uniform(0, 1, N)
        _update_best(h_rand)
        
    # Optimization phases
    if best_h is not None and time.time() - t0 < _DEADLINE - 60:
        h_opt = _nesterov_pgd(best_h, beta=10.0, steps=300, lr=0.05, deadline=_DEADLINE - 60)
        _update_best(h_opt)
        
    if best_h is not None and time.time() - t0 < _DEADLINE - 40:
        h_opt = _nesterov_pgd(best_h, beta=50.0, steps=500, lr=0.01, deadline=_DEADLINE - 40)
        _update_best(h_opt)
        
    if best_h is not None and time.time() - t0 < _DEADLINE - 20:
        h_opt = _nesterov_pgd(best_h, beta=200.0, steps=800, lr=0.003, deadline=_DEADLINE - 20)
        _update_best(h_opt)
        
    if best_h is not None and time.time() - t0 < _DEADLINE - 10:
        h_opt = _refine_lbfgs(best_h, _DEADLINE - 10)
        _update_best(h_opt)
        
    if best_h is not None and time.time() - t0 < _DEADLINE - 2:
        h_opt = _active_set_polish(best_h, _DEADLINE - 2, max_iter=500)
        _update_best(h_opt)
        
    if best_h is None:
        best_h = np.full(N, 0.5)
        
    best_h = _project_sum_box(best_h, N / 2.0)
    return best_h
