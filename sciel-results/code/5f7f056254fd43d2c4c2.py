import time
import numpy as np
from scipy.optimize import minimize

N = 400
_T0 = time.time()
_DEADLINE = _T0 + 145.0  # Safety margin for 150s limit

def _compute_ac(h):
    """Compute full linear autoconvolution."""
    return np.convolve(h, h, mode='full')

def _soft_max_abs_ac_grad(h, beta=10.0):
    """
    Compute soft max of absolute autoconvolution and its gradient.
    Returns (value, gradient)
    """
    ac = _compute_ac(h)
    abs_ac = np.abs(ac)
    
    # Numerical stability for log-sum-exp
    max_ac = np.max(abs_ac)
    if max_ac < 1e-12:
        return 1e-12, np.zeros_like(h)
        
    # Shift to prevent overflow
    shifted = beta * (abs_ac - max_ac)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    
    # Soft max value
    val = max_ac + np.log(sum_exp) / beta
    
    # Weights for the soft max
    weights = exp_shifted / sum_exp
    
    # Gradient calculation
    # d/dh_j (sum_k w_k |ac_k|) = sum_k w_k * sign(ac_k) * (h_{j+k} + h_{j-k})
    # Note: ac_k is defined for k in [-(N-1), N-1]. 
    # np.convolve(h, h, 'full') gives lags from -(N-1) to N-1? 
    # No, np.convolve(h, h, 'full') index i corresponds to lag i-(N-1).
    # Let's map indices carefully.
    # ac_full[i] = sum_{j} h[j] h[i-j] where valid indices are used.
    # Standard correlation/convolution index: 
    # ac_full[k] for k=0..2N-2. Lag = k - (N-1).
    
    grad = np.zeros_like(h)
    
    # We iterate over the lags. 
    # It is more efficient to iterate over the weights and add contributions.
    # However, N=400, so 2N-1=799 lags. 
    # For each lag, we add a shifted version of h to grad.
    
    # Pre-compute signs
    signs = np.sign(ac)
    
    # To speed up, we can use FFT for the weighted sum of shifts, 
    # but for N=400, direct loops are likely faster than FFT overhead in Python.
    # Let's try direct loops with vectorized inner steps.
    
    for k_idx in range(len(ac)):
        w = weights[k_idx]
        if w < 1e-10:
            continue
        sgn = signs[k_idx]
        lag = k_idx - (N - 1)
        
        # Contribution to grad from this lag:
        # grad[j] += w * sgn * (h[j+lag] + h[j-lag])
        
        # Term 1: h[j+lag]
        # Valid j: 0 <= j < N, 0 <= j+lag < N
        # j >= -lag, j < N-lag
        start1 = max(0, -lag)
        end1 = min(N, N - lag)
        if start1 < end1:
            grad[start1:end1] += w * sgn * h[start1+lag:end1+lag]
            
        # Term 2: h[j-lag]
        # Valid j: 0 <= j < N, 0 <= j-lag < N
        # j >= lag, j < N+lag
        start2 = max(0, lag)
        end2 = min(N, N + lag)
        if start2 < end2:
            grad[start2:end2] += w * sgn * h[start2-lag:end2-lag]
            
    return val, grad

def _objective_and_grad(h, beta=10.0):
    """
    Objective: log(soft_max_abs_ac) - 2*log(|sum h|)
    """
    smax, grad_smax = _soft_max_abs_ac_grad(h, beta)
    s = np.sum(h)
    
    if abs(s) < 1e-12:
        # Avoid division by zero in log
        return 1e10, np.zeros_like(h)
        
    # d/dh log(smax) = grad_smax / smax
    # d/dh (-2 log|s|) = -2 * ones / s
    g1 = grad_smax / smax
    g2 = -2.0 * np.ones_like(h) / s
    
    obj = np.log(smax) - 2.0 * np.log(np.abs(s))
    
    return obj, g1 + g2

def _score(h):
    """Compute the actual C3' score."""
    total = h.sum()
    if abs(total) < 1e-12:
        return np.inf
    ac = _compute_ac(h)
    return np.max(np.abs(ac)) / (total * total)

def _local_search_discrete(h, max_iters=20000, deadline=None):
    """
    Local search: single sign flips to minimize C3'.
    """
    h = h.copy()
    current_val = _score(h)
    
    # Precompute ac for speed? 
    # Flipping h[i] changes ac by:
    # ac_new[k] = ac_old[k] - 2*h[i]*h[i+k] - 2*h[i]*h[i-k] + 2*h[i]*h[i+k] + 2*h[i]*h[i-k] ?
    # Actually, ac(k) = sum_j h(j) h(j+k).
    # If we flip h(i) -> -h(i):
    # New ac(k) = Old ac(k) - 2*h(i)*(h(i+k) + h(i-k))
    # This allows O(N) update of ac for each flip candidate, and O(1) max check if we maintain a max-heap or similar.
    # But for N=400, recomputing ac via FFT or direct conv is fast enough.
    # np.convolve is C-optimized.
    
    start_time = time.time()
    improved = True
    iters = 0
    
    while improved and iters < max_iters:
        if deadline and time.time() > deadline:
            break
        improved = False
        iters += 1
        
        # Try flipping each element
        best_flip_idx = -1
        best_flip_val = current_val
        
        # Randomize order to avoid bias
        indices = np.arange(N)
        np.random.shuffle(indices)
        
        for i in indices:
            if deadline and time.time() > deadline:
                break
                
            trial = h.copy()
            trial[i] *= -1
            
            # Quick check: if sum becomes 0, skip
            new_total = h.sum() - 2*h[i]
            if abs(new_total) < 1e-12:
                continue
                
            val = _score(trial)
            
            if val < best_flip_val - 1e-12:
                best_flip_val = val
                best_flip_idx = i
                
        if best_flip_idx >= 0:
            h[best_flip_idx] *= -1
            current_val = best_flip_val
            improved = True
            
    return h, current_val

def construct():
    global _DEADLINE
    
    best_h = None
    best_val = np.inf
    
    # Initial diverse candidates
    candidates = []
    
    # 1. All ones
    candidates.append(np.ones(N))
    # 2. Random +/- 1
    for i in range(10):
        rng = np.random.RandomState(42 + i)
        h = rng.choice([-1.0, 1.0], size=N)
        # Ensure non-zero sum
        if abs(h.sum()) < 1.0:
            h[h > 0] += 0.1 # Slight bias to avoid zero sum, will be projected later
        candidates.append(h)
        
    # 3. Structured: Sines
    for freq in [1, 2, 3, 5]:
        t = np.arange(N)
        h = np.sin(2 * np.pi * freq * t / N)
        candidates.append(h)
        
    # 4. Chirp
    t = np.arange(N)
    h = np.sin(2 * np.pi * (0.5 * t / N)**2 * N)
    candidates.append(h)
    
    # Optimization Loop
    deadline_opt = _DEADLINE - 20.0 # Leave time for final polishing
    
    for idx, h0 in enumerate(candidates):
        if time.time() > deadline_opt:
            break
            
        # Ensure h0 is float
        h0 = np.asarray(h0, dtype=float)
        
        # L-BFGS-B Optimization
        try:
            res = minimize(
                lambda x: _objective_and_grad(x, beta=20.0)[0],
                h0,
                jac=lambda x: _objective_and_grad(x, beta=20.0)[1],
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if res.fun < 1e9: # Valid result
                h_opt = res.x
                # Project to +/- 1 for discrete evaluation? 
                # The task asks for "signed step heights". Usually implies discrete, 
                # but the score is defined on the raw value. 
                # The prompt says "sequence of signed step heights". 
                # Inspiration 1 returned continuous values. 
                # However, "signed step functions" usually implies piecewise constant. 
                # If continuous is allowed, the minimum might be lower. 
                # Let's check the score of the continuous solution.
                val_cont = _score(h_opt)
                
                # Also try the discrete projection
                h_disc = np.sign(h_opt)
                # Handle zeros in sign
                h_disc[h_disc == 0] = 1.0
                val_disc = _score(h_disc)
                
                # Compare
                if val_cont < val_disc:
                    candidate_h = h_opt
                    candidate_val = val_cont
                else:
                    candidate_h = h_disc
                    candidate_val = val_disc
                
                if candidate_val < best_val:
                    best_val = candidate_val
                    best_h = candidate_h.copy()
                    
        except Exception as e:
            continue
            
        # If we have a good candidate, do a bit of discrete local search on it
        if best_h is not None and time.time() < _DEADLINE - 5.0:
            # Only do local search if the discrete version was chosen or if it's close
            # Actually, let's just run local search on the best found so far
            pass

    # Final Polish: Local search on the best candidate found
    if best_h is not None and time.time() < _DEADLINE - 2.0:
        # If best_h is continuous, we should probably snap to discrete for the "step function" interpretation
        # But if the continuous score is better, we return that.
        # Let's assume the evaluator accepts any real-valued array.
        
        # If we have a continuous solution, local search on discrete flips might not apply directly 
        # unless we project first. 
        # Let's project the best continuous solution to discrete, run local search, and compare.
        h_disc = np.sign(best_h)
        h_disc[h_disc == 0] = 1.0
        val_disc_init = _score(h_disc)
        
        # Run local search on the discrete version
        h_polished, val_polished = _local_search_discrete(h_disc, max_iters=50000, deadline=_DEADLINE - 2.0)
        
        if val_polished < best_val:
            best_val = val_polished
            best_h = h_polished
        else:
            # Maybe the continuous one was better, keep it
            pass

    # Fallback
    if best_h is None:
        best_h = np.ones(N)
        
    return best_h

# Force execution check if run directly
if __name__ == "__main__":
    res = construct()
    print(f"Score: {_score(res)}")
