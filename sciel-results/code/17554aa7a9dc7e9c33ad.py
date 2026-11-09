import time
import itertools
import numpy as np

# EVOLVE-BLOCK-START

def _generate_5_smooth_numbers(limit=10000):
    """Generate all 5-smooth numbers (Hamming numbers) <= limit."""
    smooths = set([1])
    # 2^a * 3^b * 5^c <= limit
    max_a = int(np.log2(limit))
    max_b = int(np.log(limit) / np.log(3))
    max_c = int(np.log(limit) / np.log(5))
    
    for a in range(max_a + 1):
        p2 = 2 ** a
        if p2 > limit: break
        for b in range(max_b + 1):
            p3 = 3 ** b
            val = p2 * p3
            if val > limit: break
            for c in range(max_c + 1):
                p5 = 5 ** c
                num = val * p5
                if num > limit: break
                smooths.add(num)
                p5 *= 5
            # Optimization: if val > limit, break outer? No, p5=1 is min.
            if val > limit: break
        if p2 > limit: break
        
    return sorted(list(smooths))

def _compute_objective(A):
    """Compute max(|A+A|, |A*A|) for a set of integers A."""
    A = list(A)
    sums = set()
    prods = set()
    n = len(A)
    for i in range(n):
        for j in range(i, n):
            sums.add(A[i] + A[j])
            prods.add(A[i] * A[j])
    return max(len(sums), len(prods))

def construct():
    start_time = time.time()
    deadline = 55.0 # Safety margin for 60s limit
    
    # 1. Generate candidate pool
    # Using 5-smooth numbers as they provide dense multiplicative structure
    smooths = _generate_5_smooth_numbers(10000)
    
    if len(smooths) < 20:
        # Fallback if generation failed or limit too small (unlikely for 10000)
        return list(range(1, 21))

    # 2. Initial Search: Sliding Window over sorted smooths
    best_set = None
    best_score = float('inf')
    
    n = 20
    for i in range(len(smooths) - n + 1):
        window = smooths[i:i+n]
        # Quick check: if window range is too large, sum set might explode
        # But let's just compute
        score = _compute_objective(window)
        if score < best_score:
            best_score = score
            best_set = window[:]
            
        if time.time() - start_time > deadline:
            break

    if best_set is None:
        return list(range(1, 21))

    # 3. Local Search / Hill Climbing
    # Try to replace elements in best_set with other smooth numbers
    # to further reduce the objective.
    
    # Candidate pool for replacement: all smooths not in current best_set
    current_set = set(best_set)
    candidates = [x for x in smooths if x not in current_set]
    
    improved = True
    while improved:
        improved = False
        # To save time, we might not check ALL candidates for ALL positions
        # But n=20 and candidates ~100-200 is 2000-4000 evaluations.
        # Each eval is O(400). Total ~1.6M ops. Fast.
        
        for i in range(n):
            if time.time() - start_time > deadline:
                return best_set
            
            original_val = best_set[i]
            # Try replacing best_set[i] with each candidate
            # Optimization: Only try candidates close to original_val?
            # Or all? Let's try all, but break if too slow.
            
            for x in candidates:
                if x in current_set:
                    continue
                
                # Construct new set
                # We can update the sum/prod sets incrementally, but for n=20,
                # recomputing from scratch is simpler and less error-prone.
                new_set = best_set[:]
                new_set[i] = x
                
                # Check distinctness (x is not in current_set, and current_set is distinct, so new_set is distinct)
                
                score = _compute_objective(new_set)
                
                if score < best_score:
                    best_score = score
                    best_set = new_set
                    current_set = set(best_set)
                    improved = True
                    # After an improvement, we might want to restart the loop
                    # or continue. Continuing is fine, but 'candidates' list
                    # should be updated? x is now in current_set.
                    # We can just let the 'if x in current_set' check handle it.
                    # But we removed original_val. original_val is now a candidate.
                    # Let's update candidates list slightly or just rely on set check.
                    # For efficiency, we can append original_val to candidates if not there?
                    # Actually, just iterating over the static `candidates` list 
                    # might miss the opportunity to swap back, but hill climbing 
                    # usually works with a fixed pool or dynamic. 
                    # Let's keep it simple: static pool of all smooths.
                    # The check `if x in current_set` prevents duplicates.
                    
        if time.time() - start_time > deadline:
            break

    return best_set

# EVOLVE-BLOCK-END
