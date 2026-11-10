# EVOLVE-BLOCK-START
import numpy as np
import time


def furstenberg_sarkozy_205() -> np.ndarray:
    """
    Find a large square-difference-free set modulo 205.
    Uses randomized greedy construction with multiple restarts, followed by
    local search (add/remove-refill cycles) to escape local minima.
    """
    m = 205
    deadline = time.time() + 620  # safety margin before 630s kill

    # Precompute forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    forbidden_list = list(forbidden)
    
    # Precompute adjacency: adj[i] is the set of vertices j such that (i-j) mod m is in forbidden
    adj = [set() for _ in range(m)]
    for i in range(m):
        for f in forbidden_list:
            j = (i + f) % m
            if j != i:
                adj[i].add(j)
            j2 = (i - f) % m
            if j2 != i:
                adj[i].add(j2)
    
    # Convert to boolean arrays for fast checking
    # adj_mask[i] is a bitset (Python int) of neighbors of i
    adj_mask = [0] * m
    for i in range(m):
        mask = 0
        for j in adj[i]:
            mask |= (1 << j)
        adj_mask[i] = mask

    def is_valid_addition(selected_set, candidate):
        """Check if candidate can be added to selected_set."""
        for s in selected_set:
            d1 = (candidate - s) % m
            d2 = (s - candidate) % m
            if d1 in forbidden or d2 in forbidden:
                return False
        return True

    def greedy_random(order):
        """Greedy construction following a given order."""
        selected = set()
        for candidate in order:
            ok = True
            for s in selected:
                d1 = (candidate - s) % m
                d2 = (s - candidate) % m
                if d1 in forbidden or d2 in forbidden:
                    ok = False
                    break
            if ok:
                selected.add(candidate)
        return selected

    def local_search(selected, time_limit=10.0):
        """Local search with add and remove-refill moves."""
        selected = set(selected)
        start_time = time.time()
        improved = True
        
        while improved and time.time() - start_time < time_limit:
            improved = False
            selected_list = list(selected)
            np.random.shuffle(selected_list)
            
            # Try to add each vertex not in selected
            for v in range(m):
                if v in selected:
                    continue
                if is_valid_addition(selected, v):
                    selected.add(v)
                    improved = True
            
            # Try remove-refill: remove a vertex, then greedily add as many as possible
            if not improved:
                for v in selected_list:
                    if v not in selected:
                        continue
                    # Remove v
                    old_selected = set(selected)
                    selected.remove(v)
                    # Try to add vertices that were blocked by v
                    added_count = 0
                    for u in range(m):
                        if u in selected:
                            continue
                        if is_valid_addition(selected, u):
                            selected.add(u)
                            added_count += 1
                    if added_count > 1:
                        improved = True
                    else:
                        selected = old_selected
        
        return selected

    # Multiple randomized greedy restarts
    best_solution = set()
    np.random.seed(42)
    
    num_restarts = 50
    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break
        
        # Random permutation of vertices
        order = list(range(m))
        np.random.shuffle(order)
        
        # Greedy construction
        solution = greedy_random(order)
        
        # Local search
        solution = local_search(solution, time_limit=5.0)
        
        if len(solution) > len(best_solution):
            best_solution = solution
    
    # Final local search on best solution with more time
    if time.time() < deadline - 10:
        best_solution = local_search(best_solution, time_limit=30.0)
    
    result = np.array(sorted(best_solution), dtype=np.int64)
    return result


# EVOLVE-BLOCK-END
