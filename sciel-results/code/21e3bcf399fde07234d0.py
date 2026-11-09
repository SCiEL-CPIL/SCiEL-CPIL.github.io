import time
import random

def construct():
    start_time = time.time()
    # The evaluator kills after 150s. We use a 135s deadline to be safe.
    deadline = start_time + 135
    
    # Fallback: Andrásfai graph And(9) on 26 vertices
    # It is triangle-free and has independence number 9.
    n_seed = 26
    edges_seed = [[i, j] for i in range(n_seed) for j in range(i + 1, n_seed) if (j - i) % n_seed % 3 == 1]
    best_result = {'n': n_seed, 'edges': edges_seed}
    
    # Helper to extract edges from adjacency bitmasks
    def edges_from_adj(adj, n):
        edges = []
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                mask ^= lsb
        return edges

    # Helper to check if the graph has an independent set of size 10
    # If it does NOT, the graph is valid (alpha < 10).
    def has_independent_set_10(adj, n):
        all_mask = (1 << n) - 1
        # Precompute non-neighbor masks for faster lookups in DFS
        # non_neighbors[i] contains bits for j != i such that (i,j) is NOT an edge
        non_neighbors = [0] * n
        for i in range(n):
            # all_except_i is all bits set except i
            # ~adj[i] has 0s where edges exist, 1s where they don't (and padding)
            # We mask with all_except_i to remove i itself and padding
            non_neighbors[i] = (all_mask ^ (1 << i)) & ~adj[i]
            
        # Recursive DFS with pruning
        def dfs(mask, size):
            if size == 10:
                return True
            if mask == 0:
                return False
            # Pruning: if remaining candidates < needed, impossible
            if mask.bit_count() < 10 - size:
                return False
            
            # Heuristic: pick vertex with fewest non-neighbors in mask
            # This helps prune the "include" branch faster if the vertex is very connected
            # or the "exclude" branch if it is isolated?
            # Actually, for finding an independent set, we want to pick a vertex that
            # restricts the most if included, or is most likely to be in the set?
            # Standard heuristic: pick v that maximizes degree in the induced subgraph?
            # Here, "degree" in complement is non-neighbor count.
            # If we pick a vertex with many non-neighbors, the "include" branch keeps many candidates.
            # If we pick a vertex with few non-neighbors, the "include" branch prunes a lot.
            # So picking the one with MINIMUM non-neighbors in mask is good for pruning the include branch.
            
            min_cnt = n + 1
            best_v = -1
            temp_mask = mask
            while temp_mask:
                lsb = temp_mask & -temp_mask
                v = lsb.bit_length() - 1
                cnt = (non_neighbors[v] & mask).bit_count()
                if cnt < min_cnt:
                    min_cnt = cnt
                    best_v = v
                temp_mask ^= lsb
            
            if best_v == -1:
                return False
                
            v_bit = 1 << best_v
            
            # Branch 1: Include best_v
            new_mask_include = non_neighbors[best_v] & mask
            if dfs(new_mask_include, size + 1):
                return True
                
            # Branch 2: Exclude best_v
            new_mask_exclude = mask & ~v_bit
            if dfs(new_mask_exclude, size):
                return True
                
            return False
            
        return dfs(all_mask, 0)

    # Try to find a larger graph
    # R(3, 10) <= 41, so max n is 40
    for n in range(40, 26, -1):
        if time.time() > deadline:
            break
        
        # Try multiple random restarts for this n
        restarts = 0
        while time.time() < deadline - 5:  # Keep 5s buffer
            restarts += 1
            if restarts > 5000: # Cap restarts
                break
                
            # Generate a random maximal triangle-free graph
            # adj[i] is a bitmask of neighbors of i
            adj = [0] * n
            
            # Generate all pairs and shuffle
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(pairs)
            
            for u, v in pairs:
                # Check if adding (u, v) creates a triangle
                # Triangle exists if there is a common neighbor
                common = adj[u] & adj[v]
                if common == 0:
                    # Add edge
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
            
            # Check if alpha < 10
            # We use a branch and bound to see if an independent set of size 10 exists.
            # If it does NOT exist, the graph is valid.
            if not has_independent_set_10(adj, n):
                # Valid graph found
                edges = edges_from_adj(adj, n)
                return {'n': n, 'edges': edges}
        
    return best_result
