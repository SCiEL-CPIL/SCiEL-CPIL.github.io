import itertools
import math
import random

def calculate_score(U):
    if not U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 0
    sum_set = set()
    diff_set = set()
    for a in U:
        for b in U:
            sum_set.add(a + b)
            diff_set.add(a - b)
    
    size_sum = len(sum_set)
    size_diff = len(diff_set)
    
    if size_sum == 0:
        return 0
        
    ratio = size_diff / size_sum
    denom = math.log(2 * max_u + 1)
    if denom == 0:
        return 0
        
    val = 1 + (math.log(ratio) / denom)
    return val

def construct():
    # Strategy:
    # 1. Generate a diverse pool of candidate sets using several parametric families:
    #    - Geometric progressions (with various bases)
    #    - "Compressed" geometric (replace largest elements with smaller ones if no collision)
    #    - Random sets with bias towards small values
    #    - Sets constructed from known good small sets, scaled/shifted
    # 2. Perform local search (swap-based) on the best candidates to further optimize.
    # 3. Return the global best found.
    
    best_score = 0
    best_set = [0, 1, 3]
    
    def update_best(U):
        nonlocal best_score, best_set
        if not U or 0 not in U:
            return
        U = sorted(set(U))
        if len(U) < 2:
            return
        s = calculate_score(U)
        if s > best_score:
            best_score = s
            best_set = U
    
    # --- Generate candidates ---
    
    # Family 1: Geometric progressions {0, 1, b, b^2, ..., b^(k-1)}
    for b in range(2, 15):
        for k in range(2, 8):
            U = [0] + [b**i for i in range(1, k)]
            # Ensure distinct
            if len(set(U)) == len(U):
                update_best(U)
    
    # Family 2: Compressed geometric
    # Start with geometric, try to replace the largest element with a smaller unused one
    for b in range(2, 10):
        for k in range(3, 7):
            U = [0] + [b**i for i in range(1, k)]
            U = sorted(U)
            # Try to reduce the largest element
            for _ in range(50):  # iterations
                changed = False
                max_elem = U[-1]
                # Try replacing max_elem with something smaller not in U
                # We want to keep the set size the same
                for new_val in range(max_elem - 1, 0, -1):
                    if new_val not in U:
                        new_U = U[:-1] + [new_val]
                        new_U = sorted(new_U)
                        # Check if score improves or is same (prefer smaller max)
                        old_s = calculate_score(U)
                        new_s = calculate_score(new_U)
                        if new_s >= old_s:
                            U = new_U
                            changed = True
                            break
                if not changed:
                    break
            update_best(U)
    
    # Family 3: Random sets with power-law distribution
    random.seed(42)
    for trial in range(200):
        n = random.randint(3, 8)
        # Pick n-1 distinct random numbers from 1..50, biased towards small
        # Use exponential distribution
        vals = set()
        while len(vals) < n - 1:
            # x = -ln(1-u) / lambda, scaled
            u = random.random()
            x = int(-math.log(1 - u) * 5) + 1  # mean ~5
            if x > 0:
                vals.add(x)
        U = [0] + sorted(vals)
        update_best(U)
    
    # Family 4: Local search from best candidates
    # Take the current best_set and try swapping elements
    
    # We'll do a few rounds of local search on the top 5 candidates
    # First, let's collect a few good candidates
    candidates = []
    # Re-generate a few specific ones
    for b in [2, 3, 4]:
        for k in [3, 4, 5]:
            U = [0] + [b**i for i in range(1, k)]
            if len(set(U)) == len(U):
                candidates.append(U)
    
    # Add current best
    candidates.append(list(best_set))
    
    # Local search: swap one element with another
    def local_search(U, max_iters=100):
        U = sorted(set(U))
        current_score = calculate_score(U)
        for _ in range(max_iters):
            improved = False
            # Try swapping each element (except 0) with a random new value
            for i in range(1, len(U)):
                # Try a few random replacements
                for _ in range(5):
                    new_val = random.randint(1, max(U) + 10)
                    if new_val in U:
                        continue
                    new_U = U[:]
                    new_U[i] = new_val
                    new_U = sorted(new_U)
                    new_score = calculate_score(new_U)
                    if new_score > current_score:
                        U = new_U
                        current_score = new_score
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return U, current_score
    
    for cand in candidates:
        improved_U, s = local_search(cand)
        update_best(improved_U)
    
    return best_set
