"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Combines linear overlap definition (FFT-accelerated) with analytical gradient optimization.
Structural change: uses vectorized gradient and epigraph formulation for SLSQP."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 140  # Safety margin before 150s kill


def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    Vectorized using FFT for speed.
    """
    n = len(h)
    # r_k = sum_{i=k}^{n-1} h[i] * h[i-k] for k=0..n-1
    # This is the autocorrelation. Use FFT for O(n log n) instead of O(n^2).
    # Zero-pad to avoid circular convolution issues
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    # Autocorrelation: ifft(|H|^2) gives circular autocorrelation of padded signal
    # For linear autocorrelation of length-n signal, we need:
    # r[k] = sum_{i=0}^{n-1-k} h[i+k] * h[i] for k=0..n-1
    # Using FFT: correlate h with h
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    # corr[k] for k=0..n-1 gives r_k = sum_{i=k}^{n-1} h[i]*h[i-k]
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


def _smooth_max(h, beta):
    """
    Smooth maximum of the linear overlap values using log-sum-exp.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    # Avoid overflow in exp
    return m + np.log(np.sum(np.exp(beta * (cs - m)))) / beta


def _smooth_max_grad(h, beta):
    """
    Gradient of the smooth maximum objective.
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
    
    grad = np.zeros(n)
    
    # C_k = 1 - (2/n) * r_k
    # dC_k/dh_j = -(2/n) * ( h_{j-k} (if j>=k) + h_{j+k} (if j+k<n) )
    
    # Vectorized gradient computation
    for k in range(n):
        if w[k] == 0.0:
            continue
        factor = -(2.0 / n) * w[k]
        
        # Term 1: h_{j-k} for j >= k
        if k > 0:
            grad[k:] += factor * h[:n-k]
        # Term 2: h_{j+k} for j+k < n
        if k < n:
            grad[:n-k] += factor * h[k:]
            
    return val, grad


def _fix_sum(h):
    """
    Adjust h to have sum exactly N/2, keeping values in [0, 1].
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
    # Distribute the difference evenly, but clip
    h += diff / N
    h = np.clip(h, 0.0, 1.0)
    
    # If clipping changed the sum, do a second pass
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        mask = (h > 0) & (h < 1)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
            
    # Final tiny correction on one element to ensure exact sum in floating point
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        if diff > 0:
            idx = np.argmin(h)
        else:
            idx = np.argmax(h)
        h[idx] += diff
        h = np.clip(h, 0.0, 1.0)
        
    return h


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    best_h = None
    best_score = float('inf')
    
    def _update_best(h):
        nonlocal best_h, best_score
        h = _fix_sum(h)
        s = _score(h)
        if s < best_score:
            best_score = s
            best_h = h.copy()
    
    # 1. Analytical Starts
    i = np.arange(N)
    
    # Cosine wave
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    h_cos = _fix_sum(h_cos)
    _update_best(h_cos)
    
    # Sine wave
    h_sin = 0.5 + 0.5 * np.sin(2.0 * np.pi * i / N)
    h_sin = _fix_sum(h_sin)
    _update_best(h_sin)
    
    # Triangle wave
    h_tri = 0.5 + 0.25 * np.sign(np.sin(2.0 * np.pi * i / N)) * np.abs(np.sin(2.0 * np.pi * i / N))
    h_tri = _fix_sum(h_tri)
    _update_best(h_tri)
    
    # 2. Optimization with Analytical Gradient
    # Use SLSQP with the smooth max objective and its gradient
    
    def _run_optimization(h0, beta, maxiter, ftol):
        nonlocal best_h, best_score
        
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0,
                 'jac': lambda x: np.ones(N)}]
        bounds = [(0.0, 1.0)] * N
        
        def obj_func(x):
            val, _ = _smooth_max_grad(x, beta)
            return val
            
        def jac_func(x):
            _, grad = _smooth_max_grad(x, beta)
            return grad
            
        try:
            res = minimize(obj_func, h0.copy(), jac=jac_func, method='SLSQP',
                           bounds=bounds, constraints=cons,
                           options={'maxiter': maxiter, 'ftol': ftol})
            if res.success or res.fun < best_score:
                _update_best(res.x)
        except Exception:
            pass

    # Phase 1: Refine Cosine Start
    if time.time() - t0 < _DEADLINE - 50:
        _run_optimization(h_cos, beta=50.0, maxiter=300, ftol=1e-9)
        
    # Phase 2: Refine Sine Start
    if time.time() - t0 < _DEADLINE - 40:
        _run_optimization(h_sin, beta=50.0, maxiter=200, ftol=1e-9)
        
    # Phase 3: Try random starts if time permits
    if time.time() - t0 < _DEADLINE - 30:
        for _ in range(3):
            if time.time() - t0 > _DEADLINE - 20:
                break
            h_rand = np.random.uniform(0, 1, N)
            h_rand = _fix_sum(h_rand)
            _update_best(h_rand)
            _run_optimization(h_rand, beta=50.0, maxiter=100, ftol=1e-8)
            
    # Phase 4: Epigraph polish - use auxiliary variable t to directly minimize max(C_k)
    # This is a linear program in the epigraph formulation which SLSQP can handle well
    if best_h is not None and time.time() - t0 < _DEADLINE - 10:
        def _epigraph_obj(x):
            """Objective: t (the last variable), with constraints C_k <= t for all k."""
            h = x[:N]
            t = x[N]
            return t
        
        def _epigraph_jac(x):
            h = x[:N]
            t = x[N]
            cs = _linear_overlap_values(h)
            # Gradient w.r.t. h: sum of active constraint gradients
            # For simplicity, use smooth max gradient as approximation
            _, grad_h = _smooth_max_grad(h, 200.0)
            # Gradient w.r.t. t: 1
            grad = np.concatenate([grad_h, [1.0]])
            return grad
        
        def _epigraph_cons(x):
            h = x[:N]
            t = x[N]
            cs = _linear_overlap_values(h)
            # Constraints: C_k - t <= 0 for all k
            cons_list = []
            for k in range(N):
                cons_list.append({
                    'type': 'ineq',
                    'fun': lambda x, k=k: t - cs[k],  # t - C_k >= 0
                    'jac': lambda x, k=k: None  # Let SLSQP compute numerically
                })
            # Sum constraint
            cons_list.append({
                'type': 'eq',
                'fun': lambda x: np.sum(x[:N]) - N/2.0,
                'jac': lambda x: np.concatenate([np.ones(N), [0.0]])
            })
            return cons_list
        
        try:
            x0 = np.concatenate([best_h, [best_score + 0.01]])
            bounds = [(0.0, 1.0)] * N + [(0.0, 2.0)]
            
            def obj(x):
                return x[N]  # minimize t
            
            def jac(x):
                grad_h = np.zeros(N)
                # Use gradient of smooth max as proxy
                _, gh = _smooth_max_grad(x[:N], 200.0)
                return np.concatenate([gh, [1.0]])
            
            # Use inequality constraints for epigraph
            cons = []
            for k in range(N):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda x, k=k: x[N] - _linear_overlap_values(x[:N])[k],
                    'jac': lambda x, k=k: None
                })
            cons.append({
                'type': 'eq',
                'fun': lambda x: np.sum(x[:N]) - N/2.0,
                'jac': lambda x: np.concatenate([np.ones(N), [0.0]])
            })
            
            res = minimize(obj, x0, jac=jac, method='SLSQP',
                           bounds=bounds, constraints=cons,
                           options={'maxiter': 500, 'ftol': 1e-12})
            if res.success or res.fun < best_score:
                _update_best(res.x[:N])
        except Exception:
            pass
        
    # Phase 5: Final smooth-max polish with very high beta
    if best_h is not None and time.time() - t0 < _DEADLINE - 3:
        _run_optimization(best_h, beta=200.0, maxiter=500, ftol=1e-14)
        
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _fix_sum(best_h)
    return best_h
