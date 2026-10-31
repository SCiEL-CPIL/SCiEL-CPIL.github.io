"""
Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Combines linear overlap definition (from top inspirations) with analytical gradient optimization.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 110  # Safety margin before 120s kill

# EVOLVE-BLOCK-START

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
    Vectorized gradient computation.
    """
    h = np.asarray(h, dtype=np.float64)
    n = len(h)
    cs = _linear_overlap_values(h)
    m = np.max(cs)
    exp_terms = np.exp(beta * (cs - m))
    sum_exp = np.sum(exp_terms)
    w = exp_terms / sum_exp  # Weights for the soft max
    
    val = m + np.log(sum_exp) / beta
    
    # dC_k/dh_j = -(2/n) * (h_{j-k} [if j>=k] + h_{j+k} [if j+k<n])
    # grad_j = sum_k w_k * dC_k/dh_j = -(2/n) * sum_k w_k * (h_{j-k} + h_{j+k})
    # Let's compute S_j = sum_k w_k * (h_{j-k} [j>=k] + h_{j+k} [j+k<n])
    # = sum_k w_k * h_{j-k} [j>=k] + sum_k w_k * h_{j+k} [j+k<n]
    # First term: for fixed j, sum over k=0..j of w_k * h_{j-k} = (w * h)[j] (convolution)
    # Second term: for fixed j, sum over k=0..n-1-j of w_k * h_{j+k}
    # This is like a cross-correlation
    
    # Use FFT-based convolution for efficiency
    # First term: conv_w_h[j] = sum_{k=0}^{j} w[k] * h[j-k]
    # This is standard convolution (w * h)[j]
    n_fft = 2 * n
    W = np.fft.rfft(w, n=n_fft)
    H = np.fft.rfft(h, n=n_fft)
    conv_wh = np.fft.irfft(W * H, n=n_fft)[:n]
    
    # Second term: for fixed j, sum_{k=0}^{n-1-j} w[k] * h[j+k]
    # Let m = j+k, so k = m-j, m ranges from j to n-1
    # = sum_{m=j}^{n-1} w[m-j] * h[m]
    # This is a cross-correlation-like quantity.
    # Define v[k] = w[n-1-k] (reversed w)
    # Then sum_{m=j}^{n-1} w[m-j] * h[m] = sum_{m=j}^{n-1} v[n-1-(m-j)] * h[m]
    # = sum_{m=j}^{n-1} v[n-1-m+j] * h[m]
    # Let p = n-1-m, so m = n-1-p, p ranges from n-1-j to 0
    # = sum_{p=0}^{n-1-j} v[p+j] * h[n-1-p]
    # This is getting complex. Let's use a different approach.
    # Second term T_j = sum_{k=0}^{n-1-j} w[k] * h[j+k]
    # This is the (j)-th element of the "upper triangular" part of w*h correlation.
    # We can compute it as: T = corr(w, h) where corr[i] = sum_k w[k]*h[i+k]
    # Using FFT: corr = ifft(fft(w) * conj(fft(h))) -- but need to be careful with indexing.
    # Actually, let's just compute it directly but vectorized with numpy.
    # For n=100, a simple loop over k with vectorized j should be fast enough.
    
    # Alternative: T_j = sum_k w[k] * h[j+k] for j+k < n
    # = sum_k w[k] * h[j+k] * 1[j+k < n]
    # We can compute this as a matrix-vector product or use the fact that
    # for each k, it contributes w[k]*h[k:] to positions 0..n-1-k
    
    # Let's use a cumsum-like approach or just the loop (n=100 is small)
    T = np.zeros(n)
    for k in range(n):
        if w[k] != 0.0:
            T[:n-k] += w[k] * h[k:]
    
    S = conv_wh + T
    grad = -(2.0 / n) * S
    
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

def _epigraph_objective(x, beta):
    """
    Epigraph formulation: minimize t subject to C_k(h) <= t for all k.
    x = [h_0, ..., h_{N-1}, t]
    Returns (objective, gradient) for scipy minimize.
    """
    h = x[:N]
    t = x[N]
    cs = _linear_overlap_values(h)
    
    # Smooth max of (cs - t) to make it differentiable
    # We want to penalize when any C_k > t
    # Use soft constraint: sum exp(beta*(C_k - t)) 
    # But for SLSQP, let's use a simpler approach:
    # Objective: t + penalty for violations
    # Or: minimize t where the constraints are C_k <= t
    
    # Actually, for SLSQP with constraints, we just need:
    # Objective: t
    # Constraints: C_k - t <= 0 for all k, sum(h) = N/2, 0 <= h_i <= 1
    
    return t, None  # Gradient handled separately

def construct():
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
    
    # 1. Analytical Start: Cosine Wave
    i = np.arange(N)
    h_cos = 0.5 + 0.5 * np.cos(2.0 * np.pi * i / N)
    h_cos = _fix_sum(h_cos)
    _update_best(h_cos)
    
    # 2. Epigraph formulation optimization
    # x = [h (N vars), t (1 var)], minimize t
    # Constraints: C_k(h) <= t for all k, sum(h) = N/2, 0 <= h_i <= 1
    
    def _run_epigraph(h0, t0, maxiter, ftol):
        nonlocal best_h, best_score
        
        x0 = np.concatenate([h0, [t0]])
        
        # Objective: just t
        def obj(x):
            return x[N]
        
        def obj_grad(x):
            g = np.zeros(N + 1)
            g[N] = 1.0
            return g
        
        # Inequality constraints: C_k(h) - t <= 0
        def cons_ineq(x):
            h = x[:N]
            t = x[N]
            cs = _linear_overlap_values(h)
            return t - cs  # >= 0 means C_k <= t
        
        def cons_ineq_jac(x):
            h = x[:N]
            t = x[N]
            cs = _linear_overlap_values(h)
            # We need d/dx (t - C_k) for each k
            # = [dC_k/dh_0, ..., dC_k/dh_{N-1}, -1]
            # Get gradient of C_k w.r.t. h
            # We can reuse _smooth_max_grad logic but for individual C_k
            # For efficiency, compute all at once
            n = N
            # dC_k/dh_j = -(2/n) * (h_{j-k} [j>=k] + h_{j+k} [j+k<n])
            # Build Jacobian matrix: shape (N, N+1)
            J = np.zeros((n, n + 1))
            for k in range(n):
                for j in range(n):
                    g = 0.0
                    if j >= k:
                        g += h[j-k]
                    if j + k < n:
                        g += h[j+k]
                    J[k, j] = -(2.0 / n) * g
                J[k, n] = -1.0  # d(t - C_k)/dt = 1, but constraint is t - C_k >= 0
            return J
        
        # Equality constraint: sum(h) = N/2
        def cons_eq(x):
            return np.sum(x[:N]) - N/2.0
        
        def cons_eq_jac(x):
            g = np.zeros(N + 1)
            g[:N] = 1.0
            return g
        
        constraints = [
            {'type': 'ineq', 'fun': cons_ineq, 'jac': cons_ineq_jac},
            {'type': 'eq', 'fun': cons_eq, 'jac': cons_eq_jac}
        ]
        bounds = [(0.0, 1.0)] * N + [(0.0, 1.0)]  # h in [0,1], t in [0,1]
        
        try:
            res = minimize(obj, x0, jac=obj_grad, method='SLSQP',
                           bounds=bounds, constraints=constraints,
                           options={'maxiter': maxiter, 'ftol': ftol})
            if res.x is not None:
                h_new = res.x[:N]
                _update_best(h_new)
        except Exception:
            pass

    # Phase 1: Epigraph optimization from cosine start
    if time.time() - t0 < _DEADLINE - 40:
        t_init = _score(h_cos)
        _run_epigraph(h_cos, t_init, maxiter=300, ftol=1e-10)
        
    # Phase 2: Try smooth-max SLSQP from best found so far
    if best_h is not None and time.time() - t0 < _DEADLINE - 20:
        def _run_smooth_max(h0, beta, maxiter, ftol):
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
                if res.success or (res.fun < best_score and np.all(res.x >= -1e-6)):
                    _update_best(res.x)
            except Exception:
                pass
        
        _run_smooth_max(best_h, beta=50.0, maxiter=200, ftol=1e-10)
        
    # Phase 3: Final polish with epigraph
    if best_h is not None and time.time() - t0 < _DEADLINE - 5:
        t_init = _score(best_h)
        _run_epigraph(best_h, t_init, maxiter=200, ftol=1e-12)
        
    if best_h is None:
        best_h = np.full(N, 0.5)
    
    best_h = _fix_sum(best_h)
    return best_h

# EVOLVE-BLOCK-END
