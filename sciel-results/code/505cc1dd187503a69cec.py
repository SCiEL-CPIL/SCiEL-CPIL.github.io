import numpy as np
import time
import itertools
import random

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 580  # Safety margin

    # 1. Generate Baseline: Norm 2 vectors (220 points)
    candidates_2 = []
    for combo in itertools.combinations(range(d), 2):
        for signs in itertools.product([1, -1], repeat=2):
            v = np.zeros(d, dtype=np.int64)
            v[list(combo)] = signs
            candidates_2.append(v)
    
    base_solution = np.array(candidates_2, dtype=np.int64)
    best_points = base_solution
    best_k = len(base_solution)
    
    # 2. Generate Norm 4 vectors
    candidates_4 = []
    for combo in itertools.combinations(range(d), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(d, dtype=np.int64)
            v[list(combo)] = signs
            candidates_4.append(v)
            
    if not candidates_4:
        return best_points
        
    V4 = np.array(candidates_4, dtype=np.int64)
    N4 = len(V4)
    
    # 3. Build Conflict Graph
    # Conflict if dot > 2
    dots = V4 @ V4.T
    conflict_adj = [None] * N4
    for i in range(N4):
        if time.time() > deadline - 10:
            return best_points
        neighbors = np.where(dots[i] > 2)[0]
        conflict_adj[i] = neighbors
    del dots

    # 4. Improved Search: Greedy + Local Search
    # The previous program used random greedy restarts.
    # We will use a greedy start, then apply 2-swap local search.
    
    def greedy_independent_set(order):
        active = np.ones(N4, dtype=bool)
        selected = []
        for v in order:
            if not active[v]:
                continue
            selected.append(v)
            active[v] = False
            if len(conflict_adj[v]) > 0:
                active[conflict_adj[v]] = False
        return selected

    def local_search(initial_set):
        # Represent set as a boolean mask for fast membership
        in_set = np.zeros(N4, dtype=bool)
        for v in initial_set:
            in_set[v] = True
        
        improved = True
        current_set = list(initial_set)
        
        # Iterate until no improvement
        while improved and time.time() < deadline - 5:
            improved = False
            # Shuffle candidates to explore
            candidates = np.where(~in_set)[0]
            random.shuffle(candidates)
            
            # Try adding each candidate
            for v in candidates:
                if in_set[v]:
                    continue
                
                # Find neighbors of v that are in the set
                neighbors_in_set = []
                for u in conflict_adj[v]:
                    if in_set[u]:
                        neighbors_in_set.append(u)
                
                # If we can remove a small number of neighbors to add v, and gain > 0
                # Try removing 1 neighbor
                if len(neighbors_in_set) == 1:
                    u = neighbors_in_set[0]
                    # Check if removing u allows adding v
                    # We add v, remove u. Size change: +1 - 1 = 0. No gain.
                    # But maybe u conflicts with others? 
                    # Actually, if we remove u, we might be able to add others?
                    # Simple 1-swap doesn't increase size.
                    pass
                
                # Try removing 2 neighbors
                if len(neighbors_in_set) == 2:
                    u1, u2 = neighbors_in_set
                    # Remove u1, u2, add v. Size change: -1. Bad.
                    # We need to add v and remove FEWER than 1 neighbor? Impossible.
                    # Wait, if we add v, we MUST remove all its neighbors in the set.
                    # So size change = 1 - len(neighbors_in_set).
                    # To increase size, we need len(neighbors_in_set) = 0.
                    # But v was not in set, so if it had 0 neighbors in set, greedy would have picked it (if it was considered).
                    # Since we are in local search, maybe v was skipped due to order.
                    # If len(neighbors_in_set) == 0, we can add v!
                    if len(neighbors_in_set) == 0:
                        in_set[v] = True
                        current_set.append(v)
                        improved = True
                        continue
                        
                    # If len(neighbors_in_set) >= 1, we can't increase size by simple add.
                    # What if we remove a neighbor u, and then add v, AND add another node w that was blocked by u?
                    # This is complex.
                    
                    # Let's try a different move: 
                    # Pick a node u in the set. Remove it.
                    # This opens up its neighbors.
                    # Greedily add neighbors.
                    
            if not improved:
                # Try removing a random node from the set and re-greedy-ing its neighborhood
                if len(current_set) > 0 and random.random() < 0.1:
                    u_to_remove = random.choice(current_set)
                    in_set[u_to_remove] = False
                    current_set.remove(u_to_remove)
                    
                    # Re-add neighbors of u_to_remove
                    for v in conflict_adj[u_to_remove]:
                        if not in_set[v] and random.random() < 0.5:
                            # Check if v conflicts with existing set
                            ok = True
                            for w in conflict_adj[v]:
                                if in_set[w]:
                                    ok = False
                                    break
                            if ok:
                                in_set[v] = True
                                current_set.append(v)
                    
                    if len(current_set) > len(initial_set):
                        improved = True
                        initial_set = current_set
                    else:
                        # Revert
                        in_set[u_to_remove] = True
                        current_set.append(u_to_remove)
                        
        return current_set

    best_indices = []
    
    # Run several restarts
    while time.time() < deadline - 5:
        order = np.random.permutation(N4)
        current = greedy_independent_set(order)
        
        # Local search
        improved_current = local_search(current)
        
        if len(improved_current) > len(best_indices):
            best_indices = improved_current
            if len(best_indices) > best_k:
                new_points = V4[best_indices].astype(np.int64)
                best_points = new_points
                best_k = len(best_indices)

    return best_points
