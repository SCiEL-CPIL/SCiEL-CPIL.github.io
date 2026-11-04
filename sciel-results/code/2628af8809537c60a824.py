"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
TARGET_SUM = N / 2.0

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _get_linear_overlaps(h):
    """
    Compute linear cross-correlation of h with (1-h) for all non-zero shifts.
    Returns an array of overlap values C_k for k != 0.
    C_k = sum_i h_i * (1 - h_{i+k})
    """
    g = 1.0 - h
    # np.correlate(h, g, 'full') gives shifts from -(N-1) to N-1
    # Index N is shift 0.
    c = np.correlate(h, g, mode='full')
    c_neg = c[:N-1]
    c_pos = c[N+1:] # Exclude c[N] which is shift 0
    return np.concatenate([c_neg, c_pos])

def _max_overlap(h):
    """Return the maximum linear overlap value."""
    overlaps = _get_linear_overlaps(h)
    if len(overlaps) == 0:
        return 0.0
    return float(np.max(overlaps))

def _project_to_feasible(h):
    """
    Project h onto the feasible set: 0 <= h_i <= 1 and sum(h) = N/2.
    Uses bisection on a uniform shift for exact sum.
    """
    h = np.clip(h, 0.0, 1.0).copy()
    target = TARGET_SUM
    current = np.sum(h)
    
    if abs(current - target) < 1e-14:
        return h
    
    # Bisection to find delta such that sum(clip(h + delta, 0, 1)) = target
    lo, hi = -1.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        h_test = np.clip(h + mid, 0.0, 1.0)
        s = np.sum(h_test)
        if s < target:
            lo = mid
        else:
            hi = mid
        if abs(s - target) < 1e-14:
            break
    
    h = np.clip(h + (lo + hi) / 2.0, 0.0, 1.0)
    # Final exact correction
    diff = target - np.sum(h)
    if abs(diff) > 1e-15:
        # Distribute remaining diff to non-bound elements
        mask = (h > 1e-12) & (h < 1.0 - 1e-12)
        if np.any(mask):
            h[mask] += diff / np.sum(mask)
            h = np.clip(h, 0.0, 1.0)
    
    return h

def _lse_objective_and_grad(h, beta, lam_sum):
    """
    Compute the Log-Sum-Exp objective and its gradient.
    Objective: t + penalty, where t approximates max(overlaps).
    We optimize h directly.
    LSE = (1/beta) * log(sum(exp(beta * overlaps)))
    Grad = sum( p_k * grad_k )
    """
    overlaps = _get_linear_overlaps(h)
    
    if len(overlaps) == 0:
        return 0.0, np.zeros(N)
    
    # Numerical stability
    max_ov = np.max(overlaps)
    exp_term = np.exp(beta * (overlaps - max_ov))
    sum_exp = np.sum(exp_term)
    
    if sum_exp == 0:
        lse_val = max_ov / beta # Fallback
        probs = np.ones_like(overlaps) / len(overlaps)
    else:
        lse_val = max_ov / beta + np.log(sum_exp) / beta
        probs = exp_term / sum_exp
        
    # Compute Gradient
    # grad_k is vector of length N
    # (grad_k)_i = 1 - h_{i+k} - h_{i-k}
    # We need sum_k probs[k] * (grad_k)_i
    # This is a convolution-like operation.
    # Let's compute it efficiently.
    
    # We have shifts k. 
    # c_neg corresponds to shifts k = -(N-1) ... -1
    # c_pos corresponds to shifts k = 1 ... N-1
    # probs is aligned with this order.
    
    # Let's create an array of weights for shifts
    # We need to sum probs[k] * (1 - h_{i+k} - h_{i-k})
    # Note: h is defined on [0, N-1]. h_{i+k} is 0 if out of bounds.
    
    # Efficient calculation using np.roll or explicit loops?
    # N=100, so a loop over k (198 iterations) is fine.
    
    grad = np.zeros(N)
    
    # Unpack probs into negative and positive shifts
    n_neg = N - 1
    n_pos = N - 1
    p_neg = probs[:n_neg]
    p_pos = probs[n_neg:]
    
    # Negative shifts: k = -1, -2, ..., -(N-1)
    # Index in p_neg: 0 is k=-1, 1 is k=-2, etc.
    # For a given k (negative), (grad_k)_i = 1 - h_{i+k} - h_{i-k}
    # h_{i+k} is h shifted by -k (right shift by |k|)
    # h_{i-k} is h shifted by k (left shift by |k|)
    
    # Let's iterate over shifts
    for idx, p in enumerate(p_neg):
        k = -(idx + 1) # k is -1, -2, ...
        # h_{i+k} : shift h by -k (which is positive)
        # h_{i-k} : shift h by k (which is negative)
        
        # We need to compute v_i = 1 - h_{i+k} - h_{i-k} for all i
        # h_{i+k}: if i+k in [0, N-1], h[i+k], else 0
        # h_{i-k}: if i-k in [0, N-1], h[i-k], else 0
        
        # Using np.roll is tricky with boundaries (wrap around).
        # Explicit loop is O(N^2) total, which is 10k ops. Fast enough.
        
        # Actually, we can precompute the contributions.
        # Let's just do it directly for clarity and correctness.
        
        # Create shifted versions
        # h_shifted_plus[i] = h[i+k]
        # h_shifted_minus[i] = h[i-k]
        
        # Instead of creating arrays, we can sum into grad.
        # grad += p * (1 - h[i+k] - h[i-k])
        
        # This is effectively:
        # grad += p * 1
        # grad -= p * h[i+k]
        # grad -= p * h[i-k]
        
        # Sum of p * 1 is just sum(p_neg)
        # The other terms are weighted sums of h.
        
        # Let's optimize this by accumulating weights for h.
        # This approach is O(N^2) if done naively.
        # For N=100, 10k operations is nothing.
        
        for i in range(N):
            val = 1.0
            if 0 <= i + k < N:
                val -= h[i + k]
            if 0 <= i - k < N:
                val -= h[i - k]
            grad[i] += p * val

    for idx, p in enumerate(p_pos):
        k = idx + 1 # k is 1, 2, ...
        for i in range(N):
            val = 1.0
            if 0 <= i + k < N:
                val -= h[i + k]
            if 0 <= i - k < N:
                val -= h[i - k]
            grad[i] += p * val

    # Add sum constraint gradient
    sum_diff = np.sum(h) - TARGET_SUM
    grad += lam_sum * 2.0 * sum_diff * np.ones(N)
    
    # Objective value
    obj_val = lse_val + lam_sum * sum_diff**2
    
    return obj_val, grad

def construct():
    """
    Construct the step heights h to minimize the maximum overlap of h with shifted 1-h.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    rng = np.random.default_rng(42)
    
    best_h = None
    best_score = float('inf')
    
    # Initial guesses
    inits = []
    # 1. Sinusoidal
    t = np.arange(N)
    h1 = np.full(N, 0.5) + 0.3 * np.sin(2 * np.pi * 3 * t / N)
    inits.append(_project_to_feasible(h1))
    
    # 2. Different frequency
    h2 = np.full(N, 0.5) + 0.3 * np.sin(2 * np.pi * 7 * t / N)
    inits.append(_project_to_feasible(h2))
    
    # 3. Random
    h3 = rng.uniform(0, 1, N)
    inits.append(_project_to_feasible(h3))
    
    # 4. Alternating
    h4 = np.zeros(N)
    h4[::2] = 0.8
    h4[1::2] = 0.2
    inits.append(_project_to_feasible(h4))
    
    # 5. Random Gaussian projected
    h5 = rng.normal(0.5, 0.2, N)
    inits.append(_project_to_feasible(h5))

    # Optimize each initial guess
    for h_init in inits:
        if time.time() > deadline - 10.0:
            break
            
        score_init = _max_overlap(h_init)
        if score_init < best_score:
            best_score = score_init
            best_h = h_init.copy()
            
        # Use L-BFGS-B with analytic gradient
        # We optimize a smoothed version of the max (LSE)
        # We can use a schedule of betas
        
        current_h = h_init.copy()
        
        # Phase 1: Coarse optimization
        betas = [10, 50, 200]
        
        for beta in betas:
            if time.time() > deadline - 5.0:
                break
                
            def obj(x):
                val, grad = _lse_objective_and_grad(x, beta, 100.0)
                return val, grad
            
            try:
                result = minimize(
                    obj, current_h, method='L-BFGS-B', jac=True,
                    bounds=[(0.0, 1.0)] * N,
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                current_h = _project_to_feasible(result.x)
                score_new = _max_overlap(current_h)
                if score_new < best_score:
                    best_score = score_new
                    best_h = current_h.copy()
            except Exception:
                pass
        
        # Phase 2: Fine optimization with high beta
        if time.time() < deadline - 5.0:
            beta = 1000
            def obj_fine(x):
                val, grad = _lse_objective_and_grad(x, beta, 1000.0)
                return val, grad
                
            try:
                result = minimize(
                    obj_fine, current_h, method='L-BFGS-B', jac=True,
                    bounds=[(0.0, 1.0)] * N,
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                current_h = _project_to_feasible(result.x)
                score_new = _max_overlap(current_h)
                if score_new < best_score:
                    best_score = score_new
                    best_h = current_h.copy()
            except Exception:
                pass

    # Final Polish: Local search with incremental updates
    if best_h is not None and time.time() < deadline - 2.0:
        h_polished = best_h.copy()
        # We can do a few rounds of coordinate descent with exact line search
        # But given the gradient info, maybe just one more L-BFGS step with very high beta?
        # Or a simple hill climb.
        
        # Let's try a simple random perturbation hill climb
        best_local_score = _max_overlap(h_polished)
        for _ in range(50):
            if time.time() > deadline - 1.0:
                break
            # Pick a random coordinate
            i = rng.integers(N)
            # Try small perturbations
            for delta in [0.01, -0.01, 0.005, -0.005, 0.001, -0.001]:
                h_test = h_polished.copy()
                h_test[i] += delta
                h_test = _project_to_feasible(h_test)
                score = _max_overlap(h_test)
                if score < best_local_score:
                    best_local_score = score
                    h_polished = h_test
                    break
            
        if best_local_score < best_score:
            best_score = best_local_score
            best_h = h_polished

    # Ensure final validity
    if best_h is not None:
        best_h = _project_to_feasible(best_h)
        # Final exact sum correction
        sum_h = np.sum(best_h)
        diff = TARGET_SUM - sum_h
        if abs(diff) > 1e-12:
            if best_h[0] + diff <= 1.0 and best_h[0] + diff >= 0.0:
                best_h[0] += diff
            else:
                best_h = best_h * (TARGET_SUM / sum_h)
                best_h = np.clip(best_h, 0.0, 1.0)
                best_h = _project_to_feasible(best_h)
        
        return best_h
    else:
        # Fallback
        h_fallback = np.full(N, 0.5)
        return h_fallback

# EVOLVE-BLOCK-END
