import time
import random
import sys

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Precompute conflicts using bitmasks for speed
    # conflict_mask[i] is a bitmask of all j where (i+1)*(j+1) % (i+1+j+1) == 0
    conflict_mask = [0] * N
    
    for i in range(N):
        a = i + 1
        mask = 0
        for j in range(N):
            if i != j:
                b = j + 1
                if (a * b) % (a + b) == 0:
                    mask |= (1 << j)
        conflict_mask[i] = mask
    
    # Greedy initialization: start with all odds, then add compatible evens
    def greedy_init(order):
        S = set(i for i in range(0, N, 2))  # all odds (indices 0,2,4,...)
        for e in order:
            if e not in S:
                # Check if e conflicts with any in S
                can_add = True
                for y in S:
                    if (conflict_mask[e] >> y) & 1:
                        can_add = False
                        break
                if can_add:
                    S.add(e)
        return S
    
    # Branch and bound for maximum independent set
    def max_independent_set(vertices, current_set, current_size):
        """
        vertices: list of candidate vertices to consider
        current_set: set of vertices already chosen
        current_size: size of current_set
        Returns (best_set, best_size)
        """
        if time.time() > deadline:
            return current_set, current_size
        
        if not vertices:
            return current_set, current_size
        
        # Upper bound: current_size + len(vertices)
        upper_bound = current_size + len(vertices)
        if upper_bound <= best_size[0]:
            return current_set, current_size
        
        # Choose vertex with maximum degree in remaining graph
        # For speed, just pick the first one or use a heuristic
        v = vertices[0]
        
        # Option 1: exclude v
        rest = vertices[1:]
        set1, size1 = max_independent_set(rest, current_set, current_size)
        
        # Option 2: include v
        # Remove v and all its neighbors from candidates
        new_vertices = []
        for u in rest:
            if not ((conflict_mask[v] >> u) & 1):
                new_vertices.append(u)
        
        new_set = current_set | {v}
        set2, size2 = max_independent_set(new_vertices, new_set, current_size + 1)
        
        if size1 >= size2:
            return set1, size1
        else:
            return set2, size2
    
    # Better branch and bound with bitmasks for speed
    def branch_and_bound(candidates_mask, chosen_mask, chosen_size):
        """
        candidates_mask: bitmask of vertices still to consider
        chosen_mask: bitmask of vertices already chosen
        chosen_size: count of chosen vertices
        Returns (best_chosen_mask, best_size)
        """
        if time.time() > deadline:
            return chosen_mask, chosen_size
        
        # Count candidates
        cand_count = bin(candidates_mask).count('1')
        if chosen_size + cand_count <= best_size[0]:
            return chosen_mask, chosen_size
        
        if candidates_mask == 0:
            return chosen_mask, chosen_size
        
        # Choose a vertex to branch on - use the one with fewest conflicts in candidates
        # For simplicity, pick the lowest set bit
        # Better: pick vertex that maximizes pruning
        
        # Find the vertex in candidates with minimum degree within candidates
        best_v = -1
        min_degree = N + 1
        temp_mask = candidates_mask
        while temp_mask:
            v = (temp_mask & -temp_mask).bit_length() - 1
            # Count neighbors of v within candidates
            neighbors = conflict_mask[v] & candidates_mask
            deg = bin(neighbors).count('1')
            if deg < min_degree:
                min_degree = deg
                best_v = v
            temp_mask &= temp_mask - 1
        
        if best_v == -1:
            return chosen_mask, chosen_size
        
        # Option 1: exclude best_v
        new_cand1 = candidates_mask & ~(1 << best_v)
        set1, size1 = branch_and_bound(new_cand1, chosen_mask, chosen_size)
        
        # Option 2: include best_v
        # Remove best_v and all its neighbors from candidates
        neighbors = conflict_mask[best_v]
        new_cand2 = candidates_mask & ~(1 << best_v) & ~neighbors
        new_chosen = chosen_mask | (1 << best_v)
        set2, size2 = branch_and_bound(new_cand2, new_chosen, chosen_size + 1)
        
        if size1 >= size2:
            return set1, size1
        else:
            return set2, size2
    
    # Global best
    best_size = [0]
    best_mask = 0
    
    # Greedy initialization
    evens = list(range(1, N, 2))
    random.shuffle(evens)
    
    S = greedy_init(evens)
    init_mask = 0
    for idx in S:
        init_mask |= (1 << idx)
    init_size = len(S)
    
    best_size[0] = init_size
    best_mask = init_mask
    
    # Try multiple random orderings for greedy
    for trial in range(50):
        if time.time() > deadline:
            break
        random.shuffle(evens)
        S = greedy_init(evens)
        mask = 0
        for idx in S:
            mask |= (1 << idx)
        size = len(S)
        if size > best_size[0]:
            best_size[0] = size
            best_mask = mask
    
    # Now try to improve with local search / branch and bound on the complement
    # The candidates are vertices not in best_mask
    candidates_mask = 0
    for i in range(N):
        if not (best_mask >> i) & 1:
            candidates_mask |= (1 << i)
    
    # Try branch and bound from the current best
    if time.time() < deadline - 10:
        # We want to find if we can add more vertices
        # This is hard, so let's try a different approach:
        # For each vertex not in best, check if it can be added by removing some vertices
        
        # Simple local search: try adding one vertex at a time
        # If adding v requires removing k vertices, and k < 1, skip
        # Actually, we can only add v if it has no conflicts with current set
        
        # Try to find vertices that can be added
        for v in range(N):
            if time.time() > deadline:
                break
            if (best_mask >> v) & 1:
                continue
            # Check if v conflicts with any in best
            conflicts_with_best = conflict_mask[v] & best_mask
            if conflicts_with_best == 0:
                # Can add v
                best_mask |= (1 << v)
                best_size[0] += 1
    
    # Convert back to list
    result = []
    for i in range(N):
        if (best_mask >> i) & 1:
            result.append(i + 1)
    
    return result
