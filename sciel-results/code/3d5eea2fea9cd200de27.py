import numpy as np
import time
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 145  # Safety margin before 150s kill


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


def _project_to_feasible(h):
    """
    Project h onto the feasible set: sum(h) = N/2, 0 <= h_i <= 1.
    Uses a thresholding approach to exactly satisfy the sum constraint
    while respecting box constraints.
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    
    # We want to find tau such that sum(clamp(h_i - tau, 0, 1)) = target
    # clamp(x, 0, 1) = max(0, min(1, x))
    
    # Binary search for tau
    # The function f(tau) = sum(clamp(h_i - tau, 0, 1)) is monotonic decreasing in tau.
    # We need f(tau) = target.
    
    # Range for tau:
    lo = np.min(h) - 1.0
    hi = np.max(h) + 1.0
    
    for _ in range(60):
        mid = (lo + hi) / 2.0
        h_proj = np.clip(h - mid, 0.0, 1.0)
        s = np.sum(h_proj)
        if s > target:
            # Need to decrease h_proj, so increase mid
            lo = mid
        else:
            # Need to increase h_proj, so decrease mid
            hi = mid
        
    h_proj = np.clip(h - (lo + hi) / 2.0, 0.0, 1.0)
    
    # Due to floating point, sum might not be exactly target.
    # Adjust the largest element that has room, or smallest if need to decrease.
    diff = target - np.sum(h_proj)
    if abs(diff) > 1e-12:
        if diff > 0:
            # Need to add value. Find index with max room (closest to 0)
            for i in range(N):
                if h_proj[i] < 1.0 - 1e-12:
                    room = 1.0 - h_proj[i]
                    add = min(diff, room)
                    h_proj[i] += add
                    diff -= add
                    if abs(diff) < 1e-12:
                        break
        else:
            # Need to subtract value. Find index with max value (closest to 1)
            for i in range(N):
                if h_proj[i] > 1e-12:
                    room = h_proj[i]
                    sub = min(-diff, room)
                    h_proj[i] -= sub
                    diff += sub
                    if abs(diff) < 1e-12:
                        break
    
    return h_proj


def _score(h):
    """
    Compute the maximum linear overlap.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _smooth_max_obj_grad(h, beta):
    """
    Objective and gradient for smooth max overlap.
    Returns (value, gradient)
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp  # Weights for the soft max
    
    val = m + np.log(sum_exp) / beta
    
    # Compute gradient using FFT-based convolution
    n_fft = 2 * n
    W = np.fft.rfft(w, n=n_fft)
    H = np.fft.rfft(h, n=n_fft)
    
    # conv1[j] = sum_k w[k] * h[j-k] (linear convolution)
    conv1 = np.fft.irfft(W * H, n=n_fft)[:n]
    
    # conv2[j] = sum_k w[k] * h[j+k]
    v = h[::-1]
    V = np.fft.rfft(v, n=n_fft)
    conv2_rev = np.fft.irfft(W * V, n=n_fft)[:n]
    conv2 = conv2_rev[n-1::-1]
    
    # Combine
    grad = -(2.0 / n) * (conv1 + conv2)
    
    return val, grad


def _slsqp_optimize(h0, beta, max_iter=1000, time_limit=10.0):
    """
    Optimize using SLSQP with smooth max objective.
    """
    h0 = _project_to_feasible(h0.copy())
    
    def objective(x):
        val, _ = _smooth_max_obj_grad(x, beta)
        return val
        
    def gradient(x):
        _, grad = _smooth_max_obj_grad(x, beta)
        return grad
        
    def constraint_sum(x):
        return np.sum(x) - N / 2.0
        
    def constraint_sum_grad(x):
        return np.ones(N)
        
    constraints = [{'type': 'eq', 'fun': constraint_sum, 'jac': constraint_sum_grad}]
    bounds = [(0.0, 1.0)] * N
    
    t_start = time.time()
    
    try:
        res = minimize(objective, h0, method='SLSQP', jac=gradient, 
                       bounds=bounds, constraints=constraints, 
                       options={'maxiter': max_iter, 'ftol': 1e-10})
        
        # Check if we ran out of time or failed
        if time.time() - t_start > time_limit:
             return h0, _score(h0)
             
        if res.success:
            h_opt = res.x
            # Ensure feasibility
            h_opt = _project_to_feasible(h_opt)
            return h_opt, _score(h_opt)
        else:
            return h0, _score(h0)
            
    except Exception as e:
        # Fallback to current best if SLSQP crashes
        return h0, _score(h0)


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _project_to_feasible(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
            
    # 1. Analytical Starts
    i = np.arange(N)
    
    # Cosine wave
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    _update_best(h_cos)
    
    # Sine wave
    h_sin = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)
    _update_best(h_sin)
    
    # Higher harmonic
    h_cos2 = 0.5 + 0.5 * np.cos(4.0 * np.pi * i / N)
    _update_best(h_cos2)
    
    # 2. SLSQP Optimization with Annealing
    # Phase 1: Coarse (Low Beta)
    if best_h is not None and time.time() - t0 < _DEADLINE - 30:
        h_res, val_res = _slsqp_optimize(best_h, beta=10.0, max_iter=500, time_limit=10.0)
        _update_best(h_res)
        
    # Phase 2: Medium
    if best_h is not None and time.time() - t0 < _DEADLINE - 20:
        h_res, val_res = _slsqp_optimize(best_h, beta=50.0, max_iter=1000, time_limit=10.0)
        _update_best(h_res)
        
    # Phase 3: Fine
    if best_h is not None and time.time() - t0 < _DEADLINE - 10:
        h_res, val_res = _slsqp_optimize(best_h, beta=100.0, max_iter=1000, time_limit=10.0)
        _update_best(h_res)
        
    # Phase 4: Very Fine
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        h_res, val_res = _slsqp_optimize(best_h, beta=200.0, max_iter=2000, time_limit=10.0)
        _update_best(h_res)
        
    # 3. Random Restarts with SLSQP
    if time.time() - t0 < _DEADLINE - 5:
        for _ in range(5):
            if time.time() - t0 > _DEADLINE - 2:
                break
            noise = np.random.normal(0, 0.1, N)
            h_pert = best_h + noise
            _update_best(h_pert)
            
            # Quick SLSQP polish
            if time.time() - t0 < _DEADLINE - 2:
                h_res, val_res = _slsqp_optimize(h_pert, beta=50.0, max_iter=500, time_limit=2.0)
                _update_best(h_res)
                
    if best_h is None:
        best_h = np.full(N, 0.5)
        
    best_h = _project_to_feasible(best_h)
    return best_h
