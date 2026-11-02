"""
Strategy: Combinatorial Support Search + Amplitude Refinement.

The objective J(h) = ||h||_2^2 / (||h||_1 * ||h*h||_inf) is scale-invariant.
The structure of h (which indices are non-zero and their relative magnitudes)
determines the score. 

1. Represent h by a binary support mask S (length 50) and amplitudes A (length 50).
   h_i = A_i if S_i=1 else 0.
2. Use Simulated Annealing to find a good support structure S.
   - Moves: Flip a bit in S, or swap two bits in S.
   - For each candidate S, quickly compute the optimal amplitudes A on that support
     using a few steps of projected gradient ascent (or a simplified local opt).
     Actually, for speed in SA, we can just use fixed amplitudes (e.g., 1.0) or 
     a very fast 1-step optimization to evaluate the "potential" of a support.
     Better: Use a fixed set of "template" amplitudes (e.g., uniform, gaussian, decaying)
     to evaluate supports, then refine the best supports found.
     
     Refined Plan:
     Phase 1: SA on support masks, evaluating with uniform amplitudes on support.
              This finds supports that are "spread out" to minimize autoconv peak.
     Phase 2: For the top K supports, run a fast continuous optimization (BFGS or Nelder-Mead)
              on the amplitudes only (dimension = |S|, which is much smaller than 50 if S is sparse,
              or just 50 but constrained).
"""

import numpy as np
import time
import random

N = 50

def _autoconv_max(h):
    """Compute max of discrete autoconvolution of h."""
    if np.all(h == 0):
        return 0.0
    conv = np.convolve(h, h, mode='full')
    return np.max(conv)

def _ratio(h):
    """Compute the objective: ||h||_2^2 / (||h||_1 * ||h*h||_inf)."""
    h = np.asarray(h, dtype=float)
    l1 = np.sum(h)
    if l1 <= 1e-12:
        return 0.0
    l2_sq = np.sum(h**2)
    conv_max = _autoconv_max(h)
    if conv_max <= 1e-12:
        return 0.0
    return l2_sq / (l1 * conv_max)

def _score_support_amplitudes(mask, amps):
    """
    Given a binary mask and amplitudes (defined on full length, but only mask=1 matters),
    compute the ratio.
    """
    h = np.zeros(N)
    h[mask] = amps[mask]
    return _ratio(h)

def _refine_amplitudes(mask, initial_amps, max_iters=500, timeout=2.0):
    """
    Refine amplitudes on the fixed support defined by mask.
    Uses Nelder-Mead on the sub-vector of amplitudes where mask=1.
    """
    indices = np.where(mask)[0]
    if len(indices) == 0:
        return initial_amps, 0.0
    
    # Extract initial values
    x0 = initial_amps[indices].copy()
    
    # Define objective on the sub-vector
    def obj_sub(x_sub):
        h = np.zeros(N)
        h[indices] = np.maximum(0, x_sub) # Enforce non-negativity during opt
        return -_ratio(h)
    
    start_time = time.time()
    best_x = x0
    best_val = obj_sub(x0)
    
    # Use Nelder-Mead for the sub-problem
    # Since dimension is <= 50, NM is okay if we limit iters.
    # But we can also try a simple coordinate descent for speed.
    # Let's use a simple iterative projected gradient ascent with adaptive step.
    # f(x) = sum(x^2) / (sum(x) * max_conv(x))
    # This is non-convex. NM is safer.
    
    try:
        from scipy.optimize import minimize
        # Run NM on the sub-vector
        res = minimize(obj_sub, x0, method='Nelder-Mead', 
                       options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
        
        if res.fun < best_val:
            best_x = np.maximum(0, res.x)
            best_val = res.fun
    except Exception:
        pass
        
    # Map back to full vector
    full_amps = initial_amps.copy()
    full_amps[indices] = best_x
    return full_amps, -best_val

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initialize best
    best_h = np.ones(N)
    best_score = _ratio(best_h)
    
    # Phase 1: Simulated Annealing on Support Masks
    # We evaluate a support by placing uniform amplitudes (1.0) on it.
    # This is a proxy for "how spread out" the support is.
    # Supports that are too clustered have high conv peak. Supports that are sparse/random have lower peak relative to L1/L2 tradeoff?
    # Actually, for uniform amps on support S of size k:
    # L1 = k, L2^2 = k.
    # Conv max depends on the geometry of S.
    # Ratio = k / (k * max_conv) = 1 / max_conv.
    # So maximizing ratio with uniform amps is equivalent to MINIMIZING the max autoconvolution of the indicator of S.
    # This is a known problem: minimizing the peak of the autocorrelation of a binary sequence.
    # "Sidon sets" or "Golomb rulers" have low autocorrelation peaks.
    
    # Let's use SA to find binary sequences with low autocorrelation peaks.
    
    current_mask = np.ones(N, dtype=int)
    current_conv_peak = _autoconv_max(current_mask.astype(float))
    current_score_proxy = 1.0 / current_conv_peak if current_conv_peak > 0 else 0.0
    
    best_mask = current_mask.copy()
    best_conv_peak = current_conv_peak
    best_score_proxy = current_score_proxy
    
    # Add some random initial masks to start SA from diverse points
    initial_masks = [
        np.ones(N, dtype=int),
        np.zeros(N, dtype=int),
        np.random.choice([0, 1], N, p=[0.5, 0.5]),
        np.random.choice([0, 1], N, p=[0.8, 0.2]), # Sparse
        np.random.choice([0, 1], N, p=[0.2, 0.8]), # Dense
    ]
    
    # Evaluate initial masks
    for m in initial_masks:
        if np.sum(m) < 2: continue
        peak = _autoconv_max(m.astype(float))
        score_p = 1.0 / peak if peak > 0 else 0.0
        if score_p > best_score_proxy:
            best_score_proxy = score_p
            best_conv_peak = peak
            best_mask = m.copy()
            
    current_mask = best_mask.copy()
    current_conv_peak = best_conv_peak
    current_score_proxy = best_score_proxy
    
    T = 1.0
    T_min = 0.001
    cooling = 0.999
    
    # Track top K masks for refinement
    top_masks = [] # List of (score_proxy, mask)
    
    while time.time() < deadline - 20.0: # Reserve 20s for refinement
        # Generate neighbor
        neighbor = current_mask.copy()
        move_type = random.random()
        if move_type < 0.7:
            # Flip a bit
            idx = random.randint(0, N-1)
            neighbor[idx] = 1 - neighbor[idx]
        else:
            # Swap two bits
            i, j = random.sample(range(N), 2)
            if neighbor[i] != neighbor[j]:
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        
        # Evaluate neighbor
        if np.sum(neighbor) < 2:
            continue
        peak = _autoconv_max(neighbor.astype(float))
        score_p = 1.0 / peak if peak > 0 else 0.0
        
        delta = score_p - current_score_proxy
        
        if delta > 0 or random.random() < np.exp(delta / T):
            current_mask = neighbor
            current_conv_peak = peak
            current_score_proxy = score_p
            
            if current_score_proxy > best_score_proxy:
                best_score_proxy = current_score_proxy
                best_conv_peak = current_conv_peak
                best_mask = current_mask.copy()
                
                # Add to top masks list (keep top 10 unique)
                # Simple check: is it in top_masks?
                found = False
                for i, (sp, m) in enumerate(top_masks):
                    if np.array_equal(m, current_mask):
                        found = True
                        break
                if not found:
                    top_masks.append((current_score_proxy, current_mask.copy()))
                    top_masks.sort(key=lambda x: -x[0])
                    if len(top_masks) > 10:
                        top_masks.pop()
        
        T *= cooling
        if T < T_min:
            T = T_min
            # Reheat occasionally?
            if random.random() < 0.01:
                T = 0.5
    
    # Phase 2: Refine amplitudes for the best masks
    # The proxy score (1/max_conv) assumes uniform amplitudes.
    # Optimizing amplitudes can significantly improve the ratio.
    # We will take the top unique masks and refine them.
    
    unique_masks = []
    seen = set()
    for sp, m in top_masks:
        key = tuple(m)
        if key not in seen:
            seen.add(key)
            unique_masks.append((sp, m))
    
    # Also include the absolute best mask found
    best_key = tuple(best_mask)
    if best_key not in seen:
        unique_masks.append((best_score_proxy, best_mask))
        unique_masks.sort(key=lambda x: -x[0])
        
    # Limit refinement to top 5 masks to save time
    masks_to_refine = [m for _, m in unique_masks[:5]]
    
    # Ensure we also try the global best mask even if it wasn't in top 5 (it should be)
    if tuple(best_mask) not in [tuple(m) for m in masks_to_refine]:
        masks_to_refine.append(best_mask)
        
    for mask in masks_to_refine:
        if time.time() > deadline - 5.0:
            break
            
        # Initialize amplitudes uniformly on support
        init_amps = np.zeros(N)
        init_amps[mask] = 1.0
        
        # Refine
        refined_amps, score = _refine_amplitudes(mask, init_amps)
        
        if score > best_score:
            best_score = score
            best_h = refined_amps
    
    # Final check: ensure non-negativity and return
    best_h = np.maximum(0, best_h)
    
    # One last sanity check optimization on the final best_h
    # Just in case the refinement left some tiny negatives or suboptimal values
    try:
        from scipy.optimize import minimize
        def final_obj(x):
            h = np.maximum(0, x)
            return -_ratio(h)
        
        res = minimize(final_obj, best_h, method='Nelder-Mead', 
                       options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
        if res.fun < -best_score:
            best_h = np.maximum(0, res.x)
            best_score = -res.fun
    except:
        pass
        
    return list(best_h)

# EVOLVE-BLOCK-END
