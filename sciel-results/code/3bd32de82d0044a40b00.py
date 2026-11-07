"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
def _value(h):
    """Compute the evaluator's value."""
    return np.max(np.correlate(h, 1.0 - h, mode="full")) / N * 2.0

def _objective_smooth(h, beta=50.0):
    """Smooth surrogate: soft-max of correlation terms via log-sum-exp."""
    corr = np.correlate(h, 1.0 - h, mode="full")
    # log-sum-exp for numerical stability
    m = np.max(corr)
    if m < -1e15:
        return 0.0
    lse = m + np.log(np.sum(np.exp(corr - m)))
    return lse / N * 2.0

def _objective_grad(h, beta=50.0):
    """Objective and gradient for the smooth surrogate."""
    corr = np.correlate(h, 1.0 - h, mode="full")
    m = np.max(corr)
    exp_corr = np.exp(corr - m)
    w = exp_corr / np.sum(exp_corr)  # softmax weights
    lse = m + np.log(np.sum(exp_corr))
    obj = lse / N * 2.0
    # Gradient of log-sum-exp wrt h: sum_k w_k * d(corr_k)/dh
    # corr[k] = sum_i h[i] * (1 - h[i+k]) for valid i
    # d(corr[k])/dh[i] = (1 - h[i+k]) - h[i-k] (where indices valid)
    # Build gradient via correlation with weight vector
    # corr[k] corresponds to lag k-N+1 in 0-indexed full output
    # For lag d (from -(N-1) to N-1), corr index = d + N - 1
    # d(corr[d])/dh[i] = (1-h[i+d]) - h[i-d] if both valid
    # We can compute: grad[i] = sum_d w[d+N-1] * ((1-h[i+d]) - h[i-d])
    # = sum_d w[d+N-1] * (1 - h[i+d] - h[i-d] + h[i+d]*h[i-d] ... no, linear)
    # Actually: (1-h[i+d]) - h[i-d]
    # Let's compute via two correlations:
    # sum_d w[d] * (1-h[i+d]) = convolve(h, w_reversed) style
    # More carefully: let w_full be the weight vector of length 2N-1
    # term1[i] = sum_d w[d+N-1] * (1-h[i+d]) for d where i+d valid
    # term2[i] = sum_d w[d+N-1] * h[i-d] for d where i-d valid
    # grad[i] = term1[i] - term2[i]
    # term1[i] = sum_d w[d+N-1] - sum_d w[d+N-1]*h[i+d]
    #           = sum(w) - (w * h)_i  (cross-correlation)
    # term2[i] = sum_d w[d+N-1] * h[i-d] = (w * h)_i with flipped w
    # Let's just use np.correlate properly.
    # corr_full[k] = sum_i h[i] * g[i+k] where g=1-h, k from -(N-1) to N-1
    # d corr_full[k] / dh[i] = g[i+k] - h[i-k] (when valid)
    # grad[i] = sum_k w[k] * (g[i+k] - h[i-k])
    #         = sum_k w[k]*g[i+k] - sum_k w[k]*h[i-k]
    # First term: for each i, sum over k of w[k]*g[i+k] = (w convolved with g)[i]
    # In numpy correlate: np.correlate(w, g, 'full') gives sum w[k]*g[k+i]... 
    # Let me just compute directly for N=100 (199*100 = 19900 ops, fast)
    g = 1.0 - h
    grad = np.zeros(N)
    # w is length 2N-1, index k corresponds to lag d = k - (N-1)
    for k in range(2*N - 1):
        d = k - (N - 1)
        wk = w[k]
        if wk < 1e-15:
            continue
        # d(corr[k])/dh[i] = g[i+d] - h[i-d] for valid i
        if d >= 0:
            # i+d < N => i < N-d; i-d >= 0 => i >= d
            i_min = d
            i_max = N - d
            if i_min < i_max:
                grad[i_min:i_max] += wk * (g[i_min + d:i_max + d] - h[i_min - d:i_max - d])
        else:
            dd = -d
            # i+d >= 0 => i >= -d = dd; i-d < N => i < N+dd
            i_min = dd
            i_max = N
            if i_min < i_max:
                grad[i_min:i_max] += wk * (g[i_min + d:i_max + d] - h[i_min - d:i_max - d])
    grad = grad / N * 2.0
    return obj, grad

def _project(h):
    """Project onto [0,1]^N with sum constraint using bisection on water level."""
    h = np.clip(h, 0.0, 1.0)
    # Adjust to satisfy sum = N/2
    target = N / 2.0
    diff = target - np.sum(h)
    if abs(diff) < 1e-12:
        return h
    # Add/subtract uniformly, then re-clip and iterate
    for _ in range(50):
        h = np.clip(h + diff / N, 0.0, 1.0)
        diff = target - np.sum(h)
        if abs(diff) < 1e-12:
            break
    return h

def _objective_constrained(h):
    """Objective with penalty for constraint violations."""
    h = np.clip(h, 0.0, 1.0)
    sum_h = np.sum(h)
    penalty = 1000.0 * (sum_h - N/2.0)**2
    corr = np.correlate(h, 1.0 - h, mode="full")
    return np.max(corr) / N * 2.0 + penalty

def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill
    
    # Start from seed
    h0 = np.full(N, 0.5)
    best_h = h0.copy()
    best_val = _value(best_h)
    
    # Try multiple random initializations and optimize
    rng = np.random.RandomState(42)
    
    # First, try a structured initialization: alternating pattern
    # To reduce max correlation, we want h and 1-h to be "anti-correlated" at all lags
    # A good strategy: make h look like a low-pass filtered signal vs high-pass
    
    # Try gradient descent on smooth surrogate from multiple starts
    n_starts = 10
    for s in range(n_starts):
        if time.time() - start_time > deadline:
            break
        
        if s == 0:
            h = h0.copy()
        else:
            # Random perturbation around 0.5
            h = 0.5 + 0.3 * rng.randn(N)
            h = np.clip(h, 0.0, 1.0)
            # Adjust sum
            h = _project(h)
        
        # Gradient descent with adaptive step size
        lr = 0.01
        for it in range(2000):
            if time.time() - start_time > deadline:
                break
            
            obj, grad = _objective_grad(h, beta=30.0)
            
            # Gradient step
            h_new = h - lr * grad
            h_new = _project(h_new)
            
            # Check if improved
            val_new = _value(h_new)
            if val_new < best_val - 1e-10:
                best_val = val_new
                best_h = h_new.copy()
            
            h = h_new
            
            # Adaptive learning rate
            if it % 100 == 99:
                if val_new < obj:  # decreasing
                    lr *= 1.1
                else:
                    lr *= 0.8
                lr = np.clip(lr, 1e-4, 0.1)
        
        # Polish with scipy from current best
        if time.time() - start_time > deadline:
            break
        
        # Use scipy minimize with bounds and equality constraint
        def obj_scipy(x):
            x = np.clip(x, 0.0, 1.0)
            return _value(x)
        
        def obj_scipy_grad(x):
            x = np.clip(x, 0.0, 1.0)
            # Compute gradient of max correlation
            corr = np.correlate(x, 1.0 - x, mode="full")
            max_idx = np.argmax(corr)
            d = max_idx - (N - 1)
            g = 1.0 - x
            grad = np.zeros(N)
            if d >= 0:
                i_min = d
                i_max = N - d
                if i_min < i_max:
                    grad[i_min:i_max] = g[i_min + d:i_max + d] - x[i_min - d:i_max - d]
            else:
                dd = -d
                i_min = dd
                i_max = N
                if i_min < i_max:
                    grad[i_min:i_max] = g[i_min + d:i_max + d] - x[i_min - d:i_max - d]
            return grad / N * 2.0
        
        try:
            res = minimize(
                obj_scipy, 
                best_h,
                jac=obj_scipy_grad,
                bounds=[(0.0, 1.0)] * N,
                constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0,
                             'jac': lambda x: np.ones(N)}],
                method='SLSQP',
                options={'maxiter': 500, 'ftol': 1e-12}
            )
            if res.success or res.fun < best_val:
                h_res = np.clip(res.x, 0.0, 1.0)
                # Ensure sum constraint
                h_res = _project(h_res)
                val_res = _value(h_res)
                if val_res < best_val:
                    best_val = val_res
                    best_h = h_res.copy()
        except Exception:
            pass
    
    # Final polish: local search on best_h
    if time.time() - start_time < deadline:
        # Try small perturbations
        for trial in range(50):
            if time.time() - start_time > deadline:
                break
            h_try = best_h + 0.01 * rng.randn(N)
            h_try = _project(h_try)
            val_try = _value(h_try)
            if val_try < best_val:
                best_val = val_try
                best_h = h_try.copy()
    
    # Ensure final result is valid
    best_h = np.clip(best_h, 0.0, 1.0)
    best_h = _project(best_h)
    
    # Verify
    assert np.all(np.isfinite(best_h))
    assert abs(np.sum(best_h) - N/2.0) < 1e-9 * N
    assert np.all(best_h >= 0.0) and np.all(best_h <= 1.0)
    
    return best_h
# EVOLVE-BLOCK-END
