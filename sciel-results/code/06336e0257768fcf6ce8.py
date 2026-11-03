"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Strategy: Spectral Convex Bisection.
We reframe the problem in the frequency domain. The overlap values C_k are linear
functions of the autocorrelation r_k, which is the inverse FFT of the power spectrum
|H(f)|^2. The constraint that h_i in [0,1] and sum(h)=N/2 can be enforced via
projection. We use bisection on the maximum overlap value C_max: for a given C_max,
we check if there exists h such that all C_k <= C_max, sum(h)=N/2, and 0<=h_i<=1.
This is a convex feasibility problem in the spectral domain (since C_k are linear in
|H|^2). We use scipy's minimize with a penalty method to solve the convex feasibility
problem at each bisection step.
"""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = time.time() + 140  # Safety margin before 150s kill

def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = 1 - (2/N) * r_k, where r_k = sum_j h_j * h_{(j+k) mod N} (circular autocorr)
    But for linear overlap, we use the linear autocorrelation.
    Actually, from the seed: C_k = (2/n) * (n/2 - r_k) where r is linear autocorr.
    Let's stick to the seed's definition.
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

def _fix_sum(h):
    """
    Adjust h to have sum exactly N/2, keeping values in [0, 1].
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    diff = target - np.sum(h)
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

def _spectral_feasibility(c_max, h0):
    """
    Check if there exists h such that max(C_k) <= c_max, sum(h)=N/2, 0<=h_i<=1.
    We minimize the violation of the constraint max(C_k) <= c_max.
    Objective: sum of max(0, C_k - c_max)^2 + penalty for sum and bounds.
    """
    n = N
    target_sum = n / 2.0
    
    def objective_and_grad(h):
        cs = _linear_overlap_values(h)
        violations = np.maximum(0.0, cs - c_max)
        obj = np.sum(violations ** 2)
        
        # Gradient of sum of max(0, C_k - c_max)^2
        # d/dh_j sum_k max(0, C_k - c_max)^2 = 2 * sum_{k: C_k > c_max} (C_k - c_max) * dC_k/dh_j
        grad = np.zeros(n)
        mask = cs > c_max
        if np.any(mask):
            weights = 2.0 * (cs[mask] - c_max)
            for k in np.where(mask)[0]:
                w = weights[np.where(mask)[0].tolist().index(k)]
                factor = -(2.0 / n) * w
                if k > 0:
                    grad[k:] += factor * h[:n-k]
                if k < n:
                    grad[:n-k] += factor * h[k:]
        
        # Add penalty for sum constraint: lambda * (sum(h) - target_sum)
        # We'll handle sum constraint via projection or separate penalty
        return obj, grad
    
    # Use a penalty method for the sum and bound constraints
    # Or better: use SLSQP with constraints
    
    def obj_func(h):
        cs = _linear_overlap_values(h)
        violations = np.maximum(0.0, cs - c_max)
        return np.sum(violations ** 2)
    
    def jac_func(h):
        cs = _linear_overlap_values(h)
        grad = np.zeros(n)
        mask = cs > c_max
        if np.any(mask):
            for k in np.where(mask)[0]:
                w = 2.0 * (cs[k] - c_max)
                factor = -(2.0 / n) * w
                if k > 0:
                    grad[k:] += factor * h[:n-k]
                if k < n:
                    grad[:n-k] += factor * h[k:]
        return grad
    
    cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - target_sum,
             'jac': lambda x: np.ones(n)}]
    bounds = [(0.0, 1.0)] * n
    
    try:
        res = minimize(obj_func, h0, jac=jac_func, method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': 100, 'ftol': 1e-8})
        # Check if feasible
        h_sol = res.x
        cs_sol = _linear_overlap_values(h_sol)
        return np.max(cs_sol) <= c_max + 1e-6, h_sol
    except Exception:
        return False, h0

def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    # Initial guess: balanced random
    h0 = np.full(N, 0.5)
    h0 = _fix_sum(h0)
    
    # Bisection on C_max
    # Lower bound: 0 (theoretical minimum might be > 0)
    # Upper bound: current score
    lo = 0.0
    hi = _score(h0)
    
    best_h = h0.copy()
    best_score = hi
    
    # Bisection iterations
    max_iter = 30
    for it in range(max_iter):
        if time.time() - t0 > _DEADLINE - 20:
            break
        
        mid = (lo + hi) / 2.0
        
        feasible, h_sol = _spectral_feasibility(mid, h0)
        
        if feasible:
            hi = mid
            best_h = h_sol.copy()
            best_score = mid
        else:
            lo = mid
        
        # Update h0 to the solution for better warm-start
        if feasible:
            h0 = h_sol.copy()
        else:
            # Use the best feasible so far or the current best
            pass
    
    # Final refinement: local optimization to reduce max overlap
    if time.time() - t0 < _DEADLINE - 10:
        def _smooth_max_obj_grad(h, beta=50.0):
            cs = _linear_overlap_values(h)
            m = np.max(cs)
            exp_terms = np.exp(beta * (cs - m))
            sum_exp = np.sum(exp_terms)
            w = exp_terms / sum_exp
            val = m + np.log(sum_exp) / beta
            grad = np.zeros(N)
            for k in range(N):
                if w[k] == 0.0:
                    continue
                factor = -(2.0 / N) * w[k]
                if k > 0:
                    grad[k:] += factor * h[:N-k]
                if k < N:
                    grad[:N-k] += factor * h[k:]
            return val, grad
        
        def obj_func(h):
            val, _ = _smooth_max_obj_grad(h)
            return val
        
        def jac_func(h):
            _, grad = _smooth_max_obj_grad(h)
            return grad
        
        cons = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0,
                 'jac': lambda x: np.ones(N)}]
        bounds = [(0.0, 1.0)] * N
        
        try:
            res = minimize(obj_func, best_h.copy(), jac=jac_func, method='SLSQP',
                           bounds=bounds, constraints=cons,
                           options={'maxiter': 200, 'ftol': 1e-9})
            h_final = res.x
            s_final = _score(h_final)
            if s_final < best_score:
                best_h = h_final
                best_score = s_final
        except Exception:
            pass
    
    best_h = _fix_sum(best_h)
    return best_h
