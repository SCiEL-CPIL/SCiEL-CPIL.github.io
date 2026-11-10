# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Furstenberg-Sarkozy problem (modular formulation). Find a subset of Z/205Z of maximum
    size that does NOT contain any two elements differing by a nonzero perfect square
    modulo 205.
    
    Strategy:
    1. Build the conflict graph where edges connect pairs with square differences.
    2. Use a bitmask-based local search with multiple restarts to find large independent sets.
    3. Apply simulated annealing-style perturbations to escape local optima.
    4. Return the best valid set found within the time budget.
    """
    m = 205
    k = 2
    
    # Deadline guard: kill after 600s wall-clock, keep safety margin
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin
    
    # Forbidden differences: the nonzero square residues modulo m.
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, k, m))
    forbidden.discard(0)
    
    # Build conflict matrix: conflict[i][j] = True if (i-j) mod m is in forbidden
    # Since the relation is symmetric, we only need to check one direction
    conflict = np.zeros((m, m), dtype=bool)
    for i in range(m):
        for j in range(i + 1, m):
            d1 = (i - j) % m
            d2 = (j - i) % m
            if d1 in forbidden or d2 in forbidden:
                conflict[i, j] = True
                conflict[j, i] = True
    
    # Precompute neighbor lists for faster checking
    neighbors = [set() for _ in range(m)]
    for i in range(m):
        for j in range(m):
            if i != j and conflict[i, j]:
                neighbors[i].add(j)
    
    def is_valid_subset(subset):
        """Check if a subset is an independent set."""
        subset_set = set(subset)
        for a in subset:
            for b in subset:
                if a != b and b in neighbors[a]:
                    return False
        return True
    
    def greedy_construct(order):
        """Greedy construction given an ordering of candidates."""
        selected = set()
        for candidate in order:
            ok = True
            for s in selected:
                if candidate in neighbors[s]:
                    ok = False
                    break
            if ok:
                selected.add(candidate)
        return selected
    
    def local_search(initial_subset, max_iters=10000):
        """Local search: try to add/remove/swap elements to improve size."""
        current = set(initial_subset)
        best = set(current)
        best_size = len(current)
        
        # Precompute for speed
        current_list = list(current)
        
        for iteration in range(max_iters):
            if time.time() > deadline:
                break
            
            improved = False
            
            # Try adding an element
            added = False
            for candidate in range(m):
                if candidate in current:
                    continue
                # Check if candidate conflicts with any in current
                can_add = True
                for s in current:
                    if candidate in neighbors[s]:
                        can_add = False
                        break
                if can_add:
                    current.add(candidate)
                    added = True
                    improved = True
                    break
            
            # Try removing an element that allows adding more
            if not improved:
                for to_remove in list(current):
                    # What would we gain by removing to_remove?
                    gain = 0
                    for candidate in range(m):
                        if candidate in current:
                            continue
                        # Can we add candidate if to_remove is removed?
                        can_add = True
                        for s in current:
                            if s == to_remove:
                                continue
                            if candidate in neighbors[s]:
                                can_add = False
                                break
                        if can_add:
                            gain += 1
                    
                    if gain > 1:  # Net gain
                        current.discard(to_remove)
                        # Add back the candidates we can now add
                        for candidate in range(m):
                            if candidate in current:
                                continue
                            can_add = True
                            for s in current:
                                if candidate in neighbors[s]:
                                    can_add = False
                                    break
                            if can_add:
                                current.add(candidate)
                        improved = True
                        break
            
            if not improved:
                break
            
            if len(current) > best_size:
                best = set(current)
                best_size = len(current)
        
        return best
    
    # Multiple restarts with different orderings
    best_subset = set()
    best_size = 0
    
    # First: simple greedy with natural order
    natural_order = list(range(m))
    greedy_set = greedy_construct(natural_order)
    if len(greedy_set) > best_size:
        best_subset = set(greedy_set)
        best_size = len(greedy_set)
    
    # Local search from greedy
    ls_result = local_search(greedy_set)
    if len(ls_result) > best_size:
        best_subset = set(ls_result)
        best_size = len(ls_result)
    
    # Randomized restarts
    random.seed(42)
    for restart in range(500):
        if time.time() > deadline:
            break
        
        # Random ordering
        random_order = list(range(m))
        random.shuffle(random_order)
        
        # Greedy with random order
        greedy_set = greedy_construct(random_order)
        
        # Local search
        ls_result = local_search(greedy_set, max_iters=5000)
        
        if len(ls_result) > best_size:
            best_subset = set(ls_result)
            best_size = len(ls_result)
    
    # More aggressive local search with perturbations
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Start from best and perturb
        current = set(best_subset)
        
        # Randomly remove a few elements
        if len(current) > 3:
            to_remove = random.sample(list(current), min(3, len(current)))
            for elem in to_remove:
                current.discard(elem)
        
        # Local search from perturbed state
        ls_result = local_search(current, max_iters=8000)
        
        if len(ls_result) > best_size:
            best_subset = set(ls_result)
            best_size = len(ls_result)
    
    # Final verification and conversion
    result = sorted(best_subset)
    
    # Verify validity
    result_set = set(result)
    for a in result:
        for b in result:
            if a != b and b in neighbors[a]:
                # Should not happen, but if it does, fall back to greedy
                return np.array(sorted(greedy_construct(natural_order)), dtype=np.int64)
    
    return np.array(result, dtype=np.int64)

# EVOLVE-BLOCK-END
