# EVOLVE-BLOCK-START
import numpy as np
import time
import random
from itertools import combinations


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Furstenberg-Sarkozy problem (modular formulation). Find a subset of Z/mZ of maximum
    size that does NOT contain any two elements differing by a nonzero perfect k-th power
    modulo m. Here m = 205 and k = 2 (squares), i.e. forbidden differences are the nonzero
    quadratic residues modulo 205.

    Returns:
        subset: np.ndarray of shape (s,), dtype int, a list of distinct residues in
            {0, ..., 204} with no two differing by a nonzero square modulo 205.
    """
    m = 205
    k = 2
    
    # Compute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, k, m))
    forbidden.discard(0)
    
    # Build adjacency: adj[i] is set of j such that (i-j) mod m is forbidden
    adj = [set() for _ in range(m)]
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            d = (i - j) % m
            if d in forbidden:
                adj[i].add(j)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 600.0  # 600s limit, we'll check with margin
    
    def time_remaining():
        return deadline - time.time()
    
    def is_valid_subset(subset):
        """Check if subset is valid (no forbidden differences)."""
        s = set(subset)
        for a in s:
            for b in s:
                if a == b:
                    continue
                d = (a - b) % m
                if d in forbidden:
                    return False
        return True
    
    def greedy_construct(order):
        """Greedy construction given an ordering of candidates."""
        selected = []
        selected_set = set()
        for candidate in order:
            if candidate in selected_set:
                continue
            ok = True
            for s in selected:
                d1 = (candidate - s) % m
                d2 = (s - candidate) % m
                if d1 in forbidden or d2 in forbidden:
                    ok = False
                    break
            if ok:
                selected.append(candidate)
                selected_set.add(candidate)
        return selected
    
    def local_search_improve(subset, max_iter=100000):
        """Try to improve subset via 1-swap and 2-swap moves."""
        subset_set = set(subset)
        current_size = len(subset)
        best_subset = list(subset)
        best_size = current_size
        no_improve_count = 0
        
        for iteration in range(max_iter):
            if time_remaining() < 30.0:
                break
            
            improved = False
            
            # Try 1-removal + 1-addition (swap or add)
            # First try adding any element
            for candidate in range(m):
                if candidate in subset_set:
                    continue
                # Check if candidate can be added
                can_add = True
                for s in subset:
                    d = (candidate - s) % m
                    if d in forbidden:
                        can_add = False
                        break
                if can_add:
                    new_subset = subset + [candidate]
                    best_subset = new_subset
                    best_size = len(new_subset)
                    subset = new_subset
                    subset_set.add(candidate)
                    improved = True
                    no_improve_count = 0
                    break
            
            if improved:
                continue
            
            # Try swaps: remove one, add one
            for remove_idx in range(len(subset)):
                if time_remaining() < 30.0:
                    break
                remove_elem = subset[remove_idx]
                temp_subset = [x for i, x in enumerate(subset) if i != remove_idx]
                temp_set = subset_set - {remove_elem}
                
                for candidate in range(m):
                    if candidate in temp_set:
                        continue
                    # Check if candidate can be added to temp_subset
                    can_add = True
                    for s in temp_subset:
                        d = (candidate - s) % m
                        if d in forbidden:
                            can_add = False
                            break
                    if can_add:
                        new_size = len(temp_subset) + 1
                        if new_size > best_size:
                            best_subset = temp_subset + [candidate]
                            best_size = new_size
                            subset = best_subset
                            subset_set = set(subset)
                            improved = True
                            no_improve_count = 0
                            break
                if improved:
                    break
            
            if not improved:
                no_improve_count += 1
                if no_improve_count > 5000:
                    break
        
        return best_subset
    
    def random_restart_local_search(num_restarts=50):
        """Multiple random restarts with local search."""
        best_overall = []
        best_size = 0
        random.seed(42)
        
        for restart in range(num_restarts):
            if time_remaining() < 60.0:
                break
            
            # Random ordering
            order = list(range(m))
            random.shuffle(order)
            
            # Greedy construction
            subset = greedy_construct(order)
            
            # Local search
            subset = local_search_improve(subset)
            
            if len(subset) > best_size:
                best_size = len(subset)
                best_overall = subset
        
        return best_overall
    
    # Strategy 1: Deterministic greedy (original)
    subset1 = greedy_construct(list(range(m)))
    
    # Strategy 2: Greedy with different orderings
    best_subset = subset1
    best_size = len(subset1)
    
    # Try several fixed orderings
    orderings = [
        list(range(m)),
        list(range(m-1, -1, -1)),
        list(range(m//2)) + list(range(m//2, m)),
        [i for i in range(m) if i % 2 == 0] + [i for i in range(m) if i % 2 == 1],
        [i for i in range(m) if i % 5 == 0] + [i for i in range(m) if i % 5 != 0],
    ]
    
    for order in orderings:
        if time_remaining() < 60.0:
            break
        subset = greedy_construct(order)
        if len(subset) > best_size:
            best_size = len(subset)
            best_subset = subset
    
    # Strategy 3: Local search on best greedy result
    if time_remaining() > 60.0:
        improved = local_search_improve(best_subset)
        if len(improved) > best_size:
            best_size = len(improved)
            best_subset = improved
    
    # Strategy 4: Random restarts with local search
    if time_remaining() > 120.0:
        random_best = random_restart_local_search(num_restarts=200)
        if len(random_best) > best_size:
            best_size = len(random_best)
            best_subset = random_best
    
    # Final validation
    if not is_valid_subset(best_subset):
        # Fallback to simple greedy
        best_subset = greedy_construct(list(range(m)))
    
    return np.array(best_subset, dtype=np.int64)


# EVOLVE-BLOCK-END
