import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    target_n = 43
    
    def has_independent_set_of_size_k(n, adj_masks, k, time_limit):
        """
        Returns True if there is an independent set of size k.
        Returns False if no such set exists or if time runs out (treat as invalid to be safe? 
        No, if time runs out, we don't know. Better to discard the graph).
        """
        def dfs(remaining_mask, current_size, k, start_t):
            if time.time() - start_t > time_limit:
                return None # Timeout
            
            if current_size == k:
                return True
            
            # Pruning: if not enough vertices left
            if current_size + bin(remaining_mask).count('1') < k:
                return False
            
            if remaining_mask == 0:
                return False
            
            # Pick a vertex. Heuristic: pick one with max degree in remaining?
            # For simplicity, pick lowest.
            # Better: pick a vertex, branch on include/exclude.
            # Standard MIS algorithm:
            # Pick vertex v.
            # Branch 1: Include v. Remove v and N(v).
            # Branch 2: Exclude v. Remove v.
            
            # Get vertices
            vertices = []
            m = remaining_mask
            while m:
                v = (m & -m).bit_length() - 1
                vertices.append(v)
                m &= m - 1
            
            # Heuristic: sort vertices by degree in remaining graph (descending)
            # This helps prune faster.
            def get_degree(v):
                return bin(adj_masks[v] & remaining_mask).count('1')
            
            vertices.sort(key=get_degree, reverse=True)
            
            for v in vertices:
                # Branch 1: Include v
                new_mask = remaining_mask & ~(1 << v) & ~adj_masks[v]
                res = dfs(new_mask, current_size + 1, k, start_t)
                if res is not None:
                    return res
            
            # If we looped through all and didn't return True, 
            # does that mean False?
            # Yes, if we tried all vertices and none led to size k.
            return False

        full_mask = (1 << n) - 1
        res = dfs(full_mask, 0, k, start_time)
        return res

    attempts = 0
    while time.time() < deadline and attempts < 50:
        attempts += 1
        n = target_n
        adj_masks = [0] * n
        edges = []
        
        possible_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
        random.shuffle(possible_edges)
        
        for u, v in possible_edges:
            common = adj_masks[u] & adj_masks[v]
            if bin(common).count('1') <= 1:
                adj_masks[u] |= (1 << v)
                adj_masks[v] |= (1 << u)
                edges.append([u, v])
        
        # Check validity
        # We want NO independent set of size 11.
        # So we check if there IS one.
        # If yes, invalid.
        # If no, valid.
        # If timeout, invalid (discard).
        
        is_valid = True
        # Time limit for check: 2 seconds
        check_start = time.time()
        # We check for independent set of size 11.
        # If found, is_valid = False.
        # If not found within time, is_valid = False (conservative).
        
        # Optimization: If greedy independent set size < 11, it's likely valid.
        # But we need exactness.
        
        res = has_independent_set_of_size_k(n, adj_masks, 11, 2.0)
        
        if res is None:
            # Timeout
            is_valid = False
        elif res:
            # Found independent set of size 11
            is_valid = False
        else:
            # No independent set of size 11
            is_valid = True
        
        if is_valid:
            return {'n': n, 'edges': edges}

    # Fallback
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}
