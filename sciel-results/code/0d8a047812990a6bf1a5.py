"""
Strategy: Binary/Combinatorial Search with Local Pair-Swap Polish.

The current program uses smooth optimization (SLSQP/L-BFGS) on continuous variables.
This approach hypothesizes that the global optimum for the "minimum overlap" problem
is well-approximated by a binary (0/1) or near-binary sequence, or a sequence with
few distinct levels. 

We use Differential Evolution to search over a binary mask (or soft-binary) 
representation. Then, we apply a highly targeted local search that only considers
swapping values between pairs of indices (i, j) to maintain the sum constraint 
exactly while minimizing the max correlation. This avoids the high-dimensional 
continuous local minima traps.
"""

import time
import numpy as np
from scipy.optimize import differential_evolution

N = 100
TARGET_SUM = N / 2.0

def _correlation_value(h):
    """Compute the evaluator's value."""
    corr = np.correlate(h, 1 - h, mode="full")
    return np.max(corr) / N * 2.0

def _project_to_feasible(h):
    """
    Project h onto the feasible set: sum = N/2, 0 <= h <= 1.
    Uses a robust iterative clipping and adjustment method.
    """
    h = np.clip(h, 0.0, 1.0)
    diff = np.sum(h) - TARGET_SUM
    for _ in range(100):
        if abs(diff) < 1e-12:
            break
        if diff > 0:
            # Need to decrease sum. Decrease elements that are > 0.
            mask = h > 1e-9
            if not np.any(mask):
                break
            # Distribute decrease proportionally to current value or equally?
            # Equally is safer to avoid hitting 0 too fast on some.
            # But if some are 0, they can't decrease.
            # Let's just decrease the largest ones first? Or all non-zero equally?
            # Let's try: decrease all non-zero by diff / count_nonzero.
            count = np.sum(mask)
            step = diff / count
            h[mask] -= step
            h = np.clip(h, 0.0, 1.0)
        else:
            # Need to increase sum. Increase elements that are < 1.
            mask = h < 1.0 - 1e-9
            if not np.any(mask):
                break
            count = np.sum(mask)
            step = -diff / count
            h[mask] += step
            h = np.clip(h, 0.0, 1.0)
        diff = np.sum(h) - TARGET_SUM
    
    # Final exact adjustment
    diff = np.sum(h) - TARGET_SUM
    if abs(diff) > 1e-15:
        if diff > 0:
            # Find index with smallest h to decrease
            idx = np.argmin(h)
            h[idx] -= diff
        else:
            # Find index with largest h to increase
            idx = np.argmax(h)
            h[idx] -= diff
        # Clip again and if it breaks sum, do a second pass on neighbors
        h = np.clip(h, 0.0, 1.0)
        diff = np.sum(h) - TARGET_SUM
        if abs(diff) > 1e-15:
             # This is a corner case, adjust the next best candidate
             if diff > 0:
                 mask = h > 0
                 mask[idx] = False
                 if np.any(mask):
                     idx2 = np.argmin(np.where(mask, h, 2)) # Smallest among remaining
                     h[idx2] -= diff
             else:
                 mask = h < 1
                 mask[idx] = False
                 if np.any(mask):
                     idx2 = np.argmax(np.where(mask, h, -1)) # Largest among remaining
                     h[idx2] -= diff
    return h

def _pair_swap_polish(h, max_iters=500, time_limit=10.0):
    """
    Local search using pair swaps.
    We maintain sum(h) = N/2 by swapping: h[i] += d, h[j] -= d.
    This keeps the sum constant.
    We want to minimize max(corr).
    
    Strategy:
    1. Compute current corr and max index.
    2. Identify the 'critical' lag L* where max occurs.
    3. Try to reduce corr[L*] by adjusting h at positions that contribute to it.
    4. Alternatively, just try random pair swaps with small d and accept if max(corr) decreases.
    
    To be efficient, we will try swaps that target the specific lag causing the max.
    """
    best_h = h.copy()
    best_val = _correlation_value(best_h)
    start_time = time.time()
    
    # Precompute correlation matrix structure? 
    # corr[p] = sum_k h[k] * (1-h[k+L]) where L = p - (N-1)
    # We want to reduce max_p corr[p].
    
    # Randomized hill climbing with pair moves
    rng = np.random.default_rng(42)
    
    for it in range(max_iters):
        if time.time() > start_time + time_limit:
            break
            
        # Get current max correlation and its lag
        corr = np.correlate(best_h, 1 - best_h, mode="full")
        max_idx = np.argmax(corr)
        L = max_idx - (N - 1)
        
        # The lag L involves pairs (i, i+L) for i in [0, N-1-L] (if L>=0)
        # or (i, i-L) for i in [L, N-1] (if L<0)
        
        # We want to change h[i] and h[j] such that sum is preserved.
        # Let's pick a random pair (i, j) that are "active" in the critical lag L.
        
        if L >= 0:
            # Pairs are (k, k+L) for k in 0..N-1-L
            valid_k = np.arange(0, N - L)
            if len(valid_k) == 0:
                continue
            k1 = rng.choice(valid_k)
            i1 = k1
            j1 = k1 + L
        else:
            # Pairs are (k, k-L) for k in -L..N-1 -> i = k, j = k-L
            # Actually corr[p] = sum h[k] (1-h[k+L]). If L < 0, let M = -L.
            # corr[p] = sum h[k] (1-h[k-M]) = sum h[k-M+M] (1-h[k]) ... index shift
            # Let's stick to the definition: term is h[a] * (1-h[b]) where b = a + L.
            # If L < 0, b < a.
            # Valid a: 0 <= a < N and 0 <= a+L < N => -L <= a < N.
            valid_a = np.arange(-L, N)
            if len(valid_a) == 0:
                continue
            a = rng.choice(valid_a)
            i1 = a
            j1 = a + L
            
        # We have a pair (i1, j1) contributing to the max lag.
        # We want to change h[i1] and h[j1].
        # To keep sum constant, we do h[i1] += d, h[j1] -= d.
        # The change in corr for lag L is roughly:
        # delta_corr_L = (h[i1]+d)(1-h[j1]-(-d)) - h[i1](1-h[j1]) ... wait.
        # Original term: h[i1] * (1-h[j1])
        # New term: (h[i1]+d) * (1 - (h[j1]-d)) = (h[i1]+d)(1-h[j1]+d)
        #           = h[i1](1-h[j1]) + h[i1]d + d(1-h[j1]) + d^2
        #           = orig + d(h[i1] + 1 - h[j1]) + d^2
        # We want to DECREASE corr[L], so we want d(h[i1] + 1 - h[j1]) + d^2 < 0.
        
        coeff = h[i1] + 1 - h[j1]
        if coeff > 0:
            # We need d < 0 (mostly) to decrease.
            # Max negative d is limited by h[j1] (can't go below 0) and h[i1] (can't go below 0? No, h[i1] decreases).
            # h[i1] + d >= 0 => d >= -h[i1]
            # h[j1] - d <= 1 => -d <= 1 - h[j1] => d >= h[j1] - 1
            # h[j1] - d >= 0 => d <= h[j1]
            # h[i1] + d <= 1 => d <= 1 - h[i1]
            # So d in [max(-h[i1], h[j1]-1), min(h[j1], 1-h[i1])]
            # We want d negative.
            d_min = max(-h[i1], h[j1] - 1)
            d_max = min(h[j1], 1 - h[i1])
            
            # We want d such that d*coeff + d^2 < 0.
            # Roots of d^2 + coeff*d = 0 are 0 and -coeff.
            # Since coeff > 0, for d in (-coeff, 0), the term is negative.
            # So we want d in (max(d_min, -coeff), 0).
            d_lower = max(d_min, -coeff)
            d_upper = 0.0
            
            if d_lower < d_upper:
                # Pick a random d in this range
                d = rng.uniform(d_lower, d_upper)
                # Also try small perturbations
                for d_try in [d, d*0.5, d*0.1, -0.01, 0.01]:
                    if d_try == 0: continue
                    new_i = h[i1] + d_try
                    new_j = h[j1] - d_try
                    if new_i < -1e-9 or new_i > 1 + 1e-9 or new_j < -1e-9 or new_j > 1 + 1e-9:
                        continue
                    new_h = best_h.copy()
                    new_h[i1] = np.clip(new_i, 0, 1)
                    new_h[j1] = np.clip(new_j, 0, 1)
                    # Re-project to ensure sum is exact (clip might break it slightly)
                    new_h = _project_to_feasible(new_h)
                    new_val = _correlation_value(new_h)
                    if new_val < best_val - 1e-12:
                        best_val = new_val
                        best_h = new_h
                        break # Accept and restart inner loop
        
        # Also try random global swaps to escape local minima
        if it % 10 == 0:
            i_rand = rng.integers(0, N)
            j_rand = rng.integers(0, N)
            if i_rand == j_rand: continue
            # Determine best direction
            # Try +d and -d
            for d_sign in [1, -1]:
                for d_mag in [0.01, 0.05, 0.1]:
                    d = d_sign * d_mag
                    new_i = h[i_rand] + d
                    new_j = h[j_rand] - d
                    if new_i < 0 or new_i > 1 or new_j < 0 or new_j > 1:
                        continue
                    new_h = best_h.copy()
                    new_h[i_rand] = new_i
                    new_h[j_rand] = new_j
                    new_h = _project_to_feasible(new_h)
                    new_val = _correlation_value(new_h)
                    if new_val < best_val - 1e-12:
                        best_val = new_val
                        best_h = new_h
                        break
                else:
                    continue
                break
    
    return best_h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    best_h = np.full(N, 0.5)
    best_val = _correlation_value(best_h)
    
    # Phase 1: Differential Evolution on Binary-like variables
    # We optimize 100 variables, but we bias the objective to prefer 0s and 1s.
    # Objective: max_corr + lambda * sum((h-0.5)^2) ? No, that pushes to 0.5.
    # We want to push to 0 or 1. So penalty for being in (0,1)?
    # Or just optimize max_corr, but initialize with binary-like seeds.
    
    # Let's use DE with bounds [0,1].
    # To encourage discreteness, we can add a small penalty for non-binary values?
    # Actually, just finding a good continuous solution is fine, the pair-swap polish
    # will move it towards the structure.
    
    def de_objective(h):
        h = np.clip(h, 0, 1)
        val = _correlation_value(h)
        # Add a tiny penalty for not being close to 0 or 1 to help DE jump between basins?
        # Or just keep it pure.
        return val
    
    # DE is slow for 100 dims. Let's use a lower dim representation first?
    # The prompt suggests a DIFFERENT strategy. 
    # Let's try DE on 100 dims but with a small population and maxiter.
    # Or, better: use DE on a "block" representation?
    # Let's stick to 100 dims but use a fast, coarse DE.
    
    try:
        # Use a smaller population to fit in time
        # popsize=10, maxiter=50
        result = differential_evolution(
            de_objective,
            bounds=[(0.0, 1.0)] * N,
            maxiter=20, # Very few iterations
            popsize=10,
            tol=1e-4,
            mutation=(0.5, 1.0),
            recombination=0.8,
            seed=100,
            polish=False, # We will polish ourselves
            init='latinhypercube'
        )
        
        if result.x is not None:
            h_de = _project_to_feasible(result.x)
            val_de = _correlation_value(h_de)
            if val_de < best_val:
                best_val = val_de
                best_h = h_de.copy()
    except Exception:
        pass
    
    # Phase 2: Multi-start Local Pair-Swap Polish
    # We will run the polish from several different starting points.
    
    starts = [best_h.copy()]
    
    # Add some structured binary starts
    # 1. Alternating 0, 1
    h_alt = np.array([1.0 if i % 2 == 0 else 0.0 for i in range(N)])
    starts.append(_project_to_feasible(h_alt))
    
    # 2. Blocks of 4
    h_block4 = np.zeros(N)
    for i in range(N):
        h_block4[i] = 1.0 if (i // 4) % 2 == 0 else 0.0
    starts.append(_project_to_feasible(h_block4))
    
    # 3. Blocks of 10
    h_block10 = np.zeros(N)
    for i in range(N):
        h_block10[i] = 1.0 if (i // 10) % 2 == 0 else 0.0
    starts.append(_project_to_feasible(h_block10))
    
    # 4. Random Binary
    rng = np.random.default_rng(123)
    for _ in range(5):
        h_bin = (rng.random(N) > 0.5).astype(float)
        # Adjust sum
        # Count ones
        s = np.sum(h_bin)
        # Target 50
        # Flip random bits until sum is 50
        # This is a fast way to get a valid binary vector
        indices = np.arange(N)
        rng.shuffle(indices)
        # If s > 50, flip first s-50 ones to zeros
        # If s < 50, flip first 50-s zeros to ones
        # This is O(N^2) worst case but N=100 is small.
        # Better:
        ones = np.where(h_bin == 1.0)[0]
        zeros = np.where(h_bin == 0.0)[0]
        
        if s > 50:
            to_flip = ones[:int(s - 50)]
            h_bin[to_flip] = 0.0
        elif s < 50:
            to_flip = zeros[:int(50 - s)]
            h_bin[to_flip] = 1.0
        
        # Now sum is exactly 50.
        starts.append(h_bin.copy())
        
    # Run polish on each start
    for i, h_start in enumerate(starts):
        if time.time() > deadline - 20.0:
            break
            
        # Allocate time based on remaining budget
        time_left = deadline - time.time()
        # Give more time to the best current solution
        if i == 0:
            alloc_time = min(30.0, time_left * 0.5)
        else:
            alloc_time = min(5.0, time_left / max(1, len(starts) - i))
            
        if alloc_time < 1.0:
            continue
            
        h_polished = _pair_swap_polish(h_start, max_iters=1000, time_limit=alloc_time)
        val_polished = _correlation_value(h_polished)
        
        if val_polished < best_val:
            best_val = val_polished
            best_h = h_polished.copy()
            
    # Phase 3: Final Polish on the best solution found
    if time.time() < deadline - 5.0:
        time_left = deadline - time.time()
        best_h = _pair_swap_polish(best_h, max_iters=2000, time_limit=time_left - 2.0)
        best_val = _correlation_value(best_h)
        
    # Final projection
    best_h = _project_to_feasible(best_h)
    
    return best_h

# Keep the function name and signature
construct = construct
