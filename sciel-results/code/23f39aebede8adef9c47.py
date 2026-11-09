"""Program for integer sum-product extremal sets using local search."""
import time
import random
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 55.0  # Safety margin
    n = 20
    max_val = 10000
    
    def evaluate(A):
        """Compute max(|A+A|, |A*A|) for set A using NumPy for speed."""
        A = np.array(sorted(A), dtype=np.int64)
        # Generate all pairs (i, j) with i <= j
        i_idx, j_idx = np.triu_indices(n)
        sums = A[i_idx] + A[j_idx]
        prods = A[i_idx] * A[j_idx]
        
        # Count unique values
        # np.unique is efficient for small arrays
        unique_sums = np.unique(sums)
        unique_prods = np.unique(prods)
        
        return max(len(unique_sums), len(unique_prods))
    
    best_set = None
    best_score = float('inf')
    
    # Generate high-quality initial candidates
    candidates = []
    
    # 1. Geometric Progressions (small product set, moderate sum set)
    # A = {a * r^i}. If r is integer, products are a^2 * r^(i+j).
    # Sum set size is roughly n(n+1)/2 if no collisions, but GP has structure.
    for base in [1, 2, 3, 5, 10]:
        for ratio in [2, 3, 4, 5]:
            cand = []
            val = base
            for i in range(n):
                if val > max_val:
                    break
                cand.append(val)
                val *= ratio
            if len(cand) == n:
                candidates.append(cand)
        
        # Try non-integer ratios by scaling? No, must be integers.
        # Try base * k^i for small k
    
    # 2. Arithmetic Progressions (small sum set, large product set)
    # A = {a + i*d}. Sum set is 2a + (i+j)d. Size ~ 2n.
    # Product set is (a+id)(a+jd) = a^2 + a(i+j)d + ijd. Size ~ n^2.
    # APs usually have large product sets.
    for start in [1, 2, 3]:
        for step in [1, 2, 3, 5, 10]:
            cand = [start + i * step for i in range(n)]
            if max(cand) <= max_val:
                candidates.append(cand)
                
    # 3. Mixed / Hybrid
    # Try sets with small elements to keep products small, but spread out to keep sums small?
    # Actually, to minimize max(|A+A|, |A*A|), we want both small.
    # Small numbers -> small products.
    # Distinct numbers -> sums are distinct if gap is large enough?
    # If A is {1, 2, 4, 8, ...}, sums are distinct (binary representation). |A+A| = n(n+1)/2.
    # Products are also distinct. |A*A| = n(n+1)/2.
    # For n=20, 20*21/2 = 210.
    # Can we do better?
    # If we use a GP with ratio 2, |A+A| = 210, |A*A| = 210. Max = 210.
    # If we use AP 1..20, |A+A| = 39 (1+1 to 20+20 = 2 to 40, all integers), |A*A| is much larger (~200+).
    # So GP is likely better for minimizing the MAX.
    
    # Let's add some random sets of small integers
    for _ in range(50):
        # Sample from 1 to 100 to keep products small
        cand = sorted(random.sample(range(1, 101), n))
        candidates.append(cand)
        
    # Evaluate candidates
    for cand in candidates:
        if len(set(cand)) != n or max(cand) > max_val:
            continue
        score = evaluate(cand)
        if score < best_score:
            best_score = score
            best_set = list(cand)
            
    if best_set is None:
        # Fallback: Geometric progression 1, 2, 4, ...
        best_set = [1 << i for i in range(n)]
        best_score = evaluate(best_set)
        
    # Local Search: Perturb the best set
    current_set = list(best_set)
    current_score = best_score
    
    # Strategy: Try replacing elements with nearby values or swapping.
    # Since GP is likely optimal, try to tweak it.
    
    while time.time() < deadline:
        improved = False
        # Pick a random element to modify
        idx = random.randint(0, n - 1)
        old_val = current_set[idx]
        
        # Try a few random replacements in a local range
        # Or try swapping with another element?
        # Let's try replacing with a random small integer
        for _ in range(10):
            # Bias towards small numbers to keep products small
            new_val = random.randint(1, 200)
            if new_val == old_val:
                continue
            if new_val in current_set:
                continue
                
            current_set[idx] = new_val
            new_score = evaluate(current_set)
            
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_set = list(current_set)
                improved = True
            else:
                current_set[idx] = old_val
                
            if time.time() > deadline:
                break
                
        if not improved and random.random() < 0.05:
            # Random restart from best
            current_set = list(best_set)
            current_score = best_score
            
    return best_set
# EVOLVE-BLOCK-END
