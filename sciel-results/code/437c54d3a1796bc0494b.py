import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    target_n = 39
    
    def is_c4_free(adj, n, u, v):
        common = adj[u] & adj[v]
        if bin(common).count('1') > 1:
            return False
        return True

    def find_independent_set(adj, n, size):
        # Exact search for independent set of size 'size'
        # Returns the set or None
        # Optimization: Use bitmasks for remaining vertices?
        # n=39, so bitmask is 64-bit int.
        
        # State: remaining_mask, current_set, depth
        # To prune, we need to know if popcount(remaining_mask) >= size - depth
        
        # We will use recursion with memoization? No, state space too big.
        # Just DFS.
        
        # To speed up, we can use a heuristic to pick the next vertex.
        # Pick vertex with max degree in remaining?
        
        def dfs(rem_mask, cur_set, depth):
            if depth == size:
                return cur_set[:]
            
            # Pruning: if not enough vertices left
            if bin(rem_mask).count('1') < size - depth:
                return None
                
            # Pick a vertex
            # Heuristic: pick vertex with max degree in rem_mask
            # To get max degree, we need to calculate it.
            # Or just pick the lowest index?
            # Let's pick the lowest index for determinism and simplicity.
            # But max degree is better for pruning.
            
            # Let's try max degree.
            best_v = -1
            max_deg = -1
            temp_mask = rem_mask
            while temp_mask:
                # Get lowest set bit
                lsb = temp_mask & -temp_mask
                v = lsb.bit_length() - 1
                # Degree of v in rem_mask
                # Neighbors of v in rem_mask: adj[v] & rem_mask
                deg = bin(adj[v] & rem_mask).count('1')
                if deg > max_deg:
                    max_deg = deg
                    best_v = v
                temp_mask ^= lsb
                
            if best_v == -1:
                return None
                
            v = best_v
            new_rem_mask = rem_mask & ~(1 << v)
            
            # Option 1: Include v
            # Remove v and its neighbors from rem_mask
            new_rem_include = new_rem_mask & ~(adj[v])
            res = dfs(new_rem_include, cur_set + [v], depth + 1)
            if res:
                return res
                
            # Option 2: Exclude v
            res = dfs(new_rem_mask, cur_set, depth)
            if res:
                return res
                
            return None
            
        initial_mask = (1 << n) - 1
        return dfs(initial_mask, [], 0)

    # Main logic
    best_result = None
    
    for trial in range(50):
        if time.time() > deadline:
            break
            
        n = target_n
        adj = [0] * n
        
        # Initialize with 13 triangles
        for t in range(13):
            a, b, c = 3*t, 3*t+1, 3*t+2
            adj[a] |= (1 << b) | (1 << c)
            adj[b] |= (1 << a) | (1 << c)
            adj[c] |= (1 << a) | (1 << b)
            
        # Randomly add edges to break C4-free constraint? No, must maintain C4-free.
        # We will add edges randomly if valid.
        
        random.seed(100 + trial)
        
        # List of potential edges (inter-triangle)
        potential = []
        for i in range(n):
            for j in range(i+1, n):
                if not (adj[i] >> j) & 1:
                    potential.append((i, j))
                    
        random.shuffle(potential)
        
        # Add edges greedily
        for u, v in potential:
            if is_c4_free(adj, n, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                
        # Now repair alpha
        # Loop while alpha >= 11
        for _ in range(100): # Max repair steps
            if time.time() > deadline:
                break
                
            indep_set = find_independent_set(adj, n, 11)
            if indep_set is None:
                # Valid graph found
                edges = []
                for i in range(n):
                    for j in range(i+1, n):
                        if (adj[i] >> j) & 1:
                            edges.append([i, j])
                best_result = {'n': n, 'edges': edges}
                break
            else:
                # Try to add an edge within indep_set to break it
                # Check all pairs in indep_set
                added = False
                # Sort pairs by potential to create C4?
                # Just try all
                for k in range(len(indep_set)):
                    for l in range(k+1, len(indep_set)):
                        u = indep_set[k]
                        v = indep_set[l]
                        if is_c4_free(adj, n, u, v):
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            added = True
                            break
                    if added:
                        break
                if not added:
                    # Cannot repair this set, break loop
                    break
                    
        if best_result:
            break
            
    if best_result:
        return best_result
        
    # Fallback to seed
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}
