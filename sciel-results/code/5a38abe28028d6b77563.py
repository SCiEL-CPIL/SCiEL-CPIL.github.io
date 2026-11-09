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
    deadline = start_time + 595
    
    # Compute forbidden differences: nonzero squares mod 205
    forbidden = set()
    for x in range(1, m):
        sq = pow(x, 2, m)
        if sq != 0:
            forbidden.add(sq)
    
    # Precompute conflict as numpy boolean array
    conflict = np.zeros((m, m), dtype=bool)
    for i in range(m):
        for j in range(m):
            d = (i - j) % m
            if d in forbidden:
                conflict[i, j] = True
    
    # compatible[i][j] = True if i and j can coexist
    compatible = ~(conflict | conflict.T)
    
    def get_addable(subset):
        """Get all elements that can be added to subset (as a set)."""
        if not subset:
            return set(range(m))
        subset_arr = np.array(subset, dtype=int)
        # For each candidate c, check if compatible with all in subset
        mask = np.all(compatible[np.ix_(range(m), subset_arr)], axis=1)
        mask[subset_arr] = False
        return set(np.where(mask)[0].tolist())
    
    def greedy_construct(order):
        """Greedy construction using numpy for speed."""
        selected = []
        available = np.ones(m, dtype=bool)
        for candidate in order:
            if not available[candidate]:
                continue
            selected.append(candidate)
            incompatible_with_candidate = ~compatible[candidate]
            available[incompatible_with_candidate] = False
            available[candidate] = False
        return selected
    
    def local_search(subset):
        """Improve subset by trying swaps."""
        subset = set(subset)
        improved = True
        max_iter = 30
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            n = len(subset)
            if n == 0:
                break
            subset_list = list(subset)
            
            for i in range(n):
                elem = subset_list[i]
                trial = subset - {elem}
                
                addable = get_addable(trial)
                
                if len(addable) >= 1:
                    # Pick a random addable element for diversity
                    new_elem = random.choice(list(addable))
                    new_subset = trial | {new_elem}
                    if len(new_subset) > n:
                        subset = new_subset
                        improved = True
                        break
                    # Same size swap - accept with some probability to escape local optima
                    elif random.random() < 0.3:
                        subset = new_subset
        return list(subset)
    
    # Start with deterministic greedy
    best_subset = greedy_construct(list(range(m)))
    best_size = len(best_subset)
    
    # Try many randomized greedy constructions
    random.seed(42)
    num_trials = 5000
    
    for trial in range(num_trials):
        if time.time() > deadline - 5:
            break
        
        order = list(range(m))
        random.shuffle(order)
        
        subset = greedy_construct(order)
        subset = local_search(subset)
        
        if len(subset) > best_size:
            best_size = len(subset)
            best_subset = subset
    
    # Final local search on best
    if time.time() < deadline - 10:
        best_subset = local_search(best_subset)
        best_size = len(best_subset)
    
    # Verification
    def is_valid_subset(subset):
        for i in range(len(subset)):
            for j in range(i + 1, len(subset)):
                a, b = subset[i], subset[j]
                if conflict[a, b] or conflict[b, a]:
                    return False
        return True
    
    if not is_valid_subset(best_subset):
        best_subset = greedy_construct(list(range(m)))
    
    return np.array(best_subset, dtype=np.int64)

# EVOLVE-BLOCK-END
