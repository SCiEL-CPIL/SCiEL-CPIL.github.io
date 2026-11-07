"""Erdos minimum overlap - L-BFGS-B with smooth max surrogate and polish."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100

def _objective_and_grad(x):
    """Objective: smooth max of correlation terms (log-sum-exp), plus penalty for violations."""
    h = x
    # Correlation: np.correlate(h, 1-h, mode="full")
    # For efficiency, compute via FFT or direct
    one_minus_h = 1.0 - h
    # Full correlation: (1-h) * h reversed... let's use np.correlate semantics
    # np.correlate(a, v) = sum a[m+k] * v[k]
    # We want correlate(h, 1-h, "full")
    # This is a 1D cross-correlation
    corr = np.correlate(h, one_minus_h, mode="full")
    
    # Smooth max via log-sum-exp with temperature
    # scale corr to avoid overflow
    c = corr * 2.0 / N  # the actual value components
    # We want max(c), use soft-max
    beta = 50.0  # temperature - higher = sharper max
    c_shifted = c - np.max(c)
    soft_max = np.max(c) + np.log(np.sum(np.exp(beta * c_shifted))) / beta
    
    # Constraints:
    # 1. sum(h) = N/2 (i.e., 50)
    sum_h = np.sum(h)
    sum_penalty = 100.0 * (sum_h - N / 2.0) ** 2
    
    # 2. h in [0, 1]
    lower_viol = np.maximum(0, -h)
    upper_viol = np.maximum(0, h - 1.0)
    bound_penalty = 1000.0 * (np.sum(lower_viol**2) + np.sum(upper_viol**2))
    
    total = soft_max + sum_penalty + bound_penalty
    
    # Gradient
    # Gradient of soft_max w.r.t. h
    # d(soft_max)/d(c_i) = exp(beta * c_shifted_i) / sum(exp(...))
    weights = np.exp(beta * c_shifted)
    weights = weights / np.sum(weights)
    # d(c_i)/d(h) where c = corr * 2/N
    # corr = correlate(h, 1-h, "full")
    # This is complex; let's compute gradient numerically or analytically
    # For now, use numerical gradient for the correlation part
    # Actually, let's compute it properly
    
    # corr[k] = sum_j h[j] * (1-h)[j+k-N+1] ... this is getting complex
    # Let's use the fact that correlate(h, 1-h, "full")[k] for k=0..2N-2
    # For k < N: sum_{j=0}^{N-1-k} h[j+k] * (1-h)[j]
    # For k >= N: sum_{j=k-N+1}^{N-1} h[j] * (1-h)[j-(k-N+1)]
    
    # Gradient of corr w.r.t. h:
    # Let f = 1-h, so corr = correlate(h, f, "full")
    # d(corr)/d(h_i) involves both h and f=1-h dependencies
    
    # Simpler: compute gradient of soft_max w.r.t. corr, then corr w.r.t. h
    grad_corr = weights * 2.0 / N  # d(soft_max)/d(corr_k)
    
    # Now d(corr_k)/d(h_i):
    # corr = correlate(h, 1-h, "full")
    # Let's compute this via the structure
    # correlate(h, v, "full")[k] = sum_j h[j+k] * v[j] for appropriate ranges
    # Here v = 1-h
    
    # For the full correlation of length 2N-1:
    # For lag k (0-indexed from 0 to 2N-2):
    # If k < N: corr[k] = sum_{j=0}^{N-1-k} h[j+k] * (1-h[j])
    # If k >= N: corr[k] = sum_{j=k-N+1}^{N-1} h[j] * (1-h[j-(k-N+1)])
    
    # d(corr[k])/d(h[i]):
    # For k < N: 
    #   term from h[j+k] where j+k=i => j=i-k: h[i]*(1-h[i-k]) contributes (1-h[i-k]) if 0<=i-k<N-1-k
    #   term from (1-h[j]) where j=i: -h[i+k]*(1) if 0<=i<N-1-k and 0<=i+k<N
    # For k >= N:
    #   let m = k-N+1
    #   corr[k] = sum_{j=m}^{N-1} h[j] * (1-h[j-m])
    #   d/d(h[i]): if i in [m, N-1]: (1-h[i-m]) - h[i+m] (if i+m < N)
    
    grad_h = np.zeros(N)
    for k in range(2 * N - 1):
        gk = grad_corr[k]
        if gk == 0:
            continue
        if k < N:
            # corr[k] = sum_{j=0}^{N-1-k} h[j+k] * (1-h[j])
            for j in range(N - 1 - k):
                # h[j+k] * (1-h[j])
                # d/d(h[j+k]): (1-h[j])
                grad_h[j+k] += gk * (1.0 - h[j])
                # d/d(h[j]): -h[j+k]
                grad_h[j] += gk * (-h[j+k])
        else:
            m = k - N + 1
            # corr[k] = sum_{j=m}^{N-1} h[j] * (1-h[j-m])
            for j in range(m, N):
                # h[j] * (1-h[j-m])
                # d/d(h[j]): (1-h[j-m])
                grad_h[j] += gk * (1.0 - h[j-m])
                # d/d(h[j-m]): -h[j]
                grad_h[j-m] += gk * (-h[j])
    
    # Add penalty gradients
    grad_h += 200.0 * (sum_h - N / 2.0)  # d(sum_penalty)/d(h_i)
    # Bound penalties
    grad_h += 2000.0 * (-lower_viol + upper_viol)
    
    return total, grad_h


def construct():
    deadline = time.time() + 140.0  # safety margin
    
    # Start from seed
    best_h = np.full(N, 0.5)
    best_val = _compute_value(best_h)
    
    # Multiple random restarts with L-BFGS-B
    rng = np.random.default_rng(42)
    
    n_restarts = 20
    for restart in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Generate initial point
        if restart == 0:
            h0 = np.full(N, 0.5)
        else:
            # Perturb from best or random
            if restart % 2 == 0:
                h0 = best_h + rng.normal(0, 0.1, N)
            else:
                # Structured perturbation: alternating pattern
                h0 = 0.5 + 0.3 * np.sin(2 * np.pi * rng.uniform(1, 5) * np.arange(N) / N)
            h0 = np.clip(h0, 0, 1)
            # Adjust to satisfy sum constraint
            h0 = h0 + (N / 2.0 - np.sum(h0)) / N
            h0 = np.clip(h0, 0, 1)
        
        try:
            res = minimize(
                _objective_and_grad,
                h0,
                method='L-BFGS-B',
                jac=True,
                bounds=[(0, 1)] * N,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if res.success or res.fun < best_val:
                h_new = np.clip(res.x, 0, 1)
                # Enforce sum constraint precisely
                diff = N / 2.0 - np.sum(h_new)
                h_new = h_new + diff / N
                h_new = np.clip(h_new, 0, 1)
                
                val = _compute_value(h_new)
                if val < best_val and np.all(np.isfinite(h_new)):
                    best_val = val
                    best_h = h_new.copy()
        except Exception:
            continue
    
    # Final polish: high-precision local optimization
    if time.time() < deadline - 10.0:
        try:
            # Use tighter tolerances
            res = minimize(
                _objective_and_grad,
                best_h,
                method='L-BFGS-B',
                jac=True,
                bounds=[(0, 1)] * N,
                options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            h_polished = np.clip(res.x, 0, 1)
            diff = N / 2.0 - np.sum(h_polished)
            h_polished = h_polished + diff / N
            h_polished = np.clip(h_polished, 0, 1)
            
            val = _compute_value(h_polished)
            if val < best_val and np.all(np.isfinite(h_polished)):
                best_val = val
                best_h = h_polished.copy()
        except Exception:
            pass
    
    # Final verification and correction
    # Ensure sum is exactly N/2 within tolerance
    diff = N / 2.0 - np.sum(best_h)
    if abs(diff) > 1e-12:
        # Distribute correction evenly
        best_h = best_h + diff / N
        best_h = np.clip(best_h, 0, 1)
    
    # Ensure finite
    if not np.all(np.isfinite(best_h)):
        return np.full(N, 0.5)
    
    return best_h


def _compute_value(h):
    """Compute the evaluator's value."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / len(h) * 2


# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
