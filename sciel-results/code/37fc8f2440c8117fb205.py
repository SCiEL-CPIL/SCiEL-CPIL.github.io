# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large subset of Z/205Z with no two elements differing by a nonzero square mod 205.
    Uses randomized greedy construction with local search improvements.
    """
    m = 205
    start_time = time.time()
    deadline = start_time + 600  # 600 second budget, keep safety margin
    
    # Compute forbidden differences: nonzero squares mod 205
    forbidden = set()
    for x in range(1, m):
        sq = pow(x, 2, m)
        if sq != 0:
            forbidden.add(sq)
    
    # Precompute conflict matrix for fast lookups
    # conflict[i][j] = True if (i-j) mod m is in forbidden
    conflict = [[False] * m for _ in range(m)]
    for i in range(m):
        for j in range(m):
            d = (i - j) % m
            if d in forbidden:
                conflict[i][j] = True
    
    def is_valid_subset(subset):
        """Check if subset is valid (no forbidden differences)."""
        for i in range(len(subset)):
            for j in range(i + 1, len(subset)):
                a, b = subset[i], subset[j]
                if conflict[a][b] or conflict[b][a]:
                    return False
        return True
    
    def greedy_construct(order):
        """Greedy construction given an order of candidates."""
        selected = []
        for candidate in order:
            ok = True
            for s in selected:
                if conflict[candidate][s]:
                    ok = False
                    break
            if ok:
                selected.append(candidate)
        return selected
    
    def local_search(subset):
        """Try to improve subset by removing one element and adding another."""
        subset = list(subset)
        improved = True
        while improved:
            improved = False
            n = len(subset)
            for i in range(n):
                # Try removing subset[i]
                trial = subset[:i] + subset[i+1:]
                # Try adding each element not in trial
                for candidate in range(m):
                    if candidate in trial:
                        continue
                    # Check if candidate can be added
                    ok = True
                    for s in trial:
                        if conflict[candidate][s]:
                            ok = False
                            break
                    if ok:
                        new_subset = trial + [candidate]
                        if len(new_subset) > len(subset):
                            subset = new_subset
                            improved = True
                            break
                if improved:
                    break
        return subset
    
    # Start with a good deterministic greedy
    best_subset = greedy_construct(list(range(m)))
    best_size = len(best_subset)
    
    # Try multiple randomized greedy constructions
    random.seed(42)
    num_trials = 500
    
    for trial in range(num_trials):
        if time.time() > deadline - 5:  # Safety margin
            break
        
        # Random permutation
        order = list(range(m))
        random.shuffle(order)
        
        # Greedy construction
        subset = greedy_construct(order)
        
        # Local search improvement
        subset = local_search(subset)
        
        if len(subset) > best_size:
            best_size = len(subset)
            best_subset = subset
            # If we've found a large set, we can reduce remaining trials
            if best_size >= 15:
                break
    
    # Additional local search passes on the best subset
    if time.time() < deadline - 10:
        best_subset = local_search(best_subset)
        best_size = len(best_subset)
    
    # Final verification
    if not is_valid_subset(best_subset):
        # Fallback to simple greedy
        best_subset = greedy_construct(list(range(m)))
    
    return np.array(best_subset, dtype=np.int64)

# EVOLVE-BLOCK-END
