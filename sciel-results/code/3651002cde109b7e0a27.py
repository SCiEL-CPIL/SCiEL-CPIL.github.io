"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55 second safety margin
    
    n = P['n']
    edge_set = set()
    for e in P['edges']:
        edge_set.add(tuple(sorted(e)))
    
    # Precompute adjacency bitmask for each vertex
    adj = [0] * n
    for u, v in edge_set:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    # Precompute for each subset whether it's a clique or independent set
    num_subsets = 1 << n
    valid_clique = [False] * num_subsets
    valid_indep = [False] * num_subsets
    valid_clique[0] = True
    valid_indep[0] = True
    
    for mask in range(1, num_subsets):
        lsb = mask & (-mask)
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        if valid_clique[rest] and (adj[v] & rest) == rest:
            valid_clique[mask] = True
        if valid_indep[rest] and (adj[v] & rest) == 0:
            valid_indep[mask] = True
    
    # Precompute list of valid masks (non-empty, clique or independent)
    valid_masks = []
    for mask in range(1, num_subsets):
        if valid_clique[mask] or valid_indep[mask]:
            valid_masks.append(mask)
    
    def popcount(x):
        return bin(x).count('1')
    
    # Sort by popcount descending to try larger parts first
    valid_masks.sort(key=popcount, reverse=True)
    
    full_mask = (1 << n) - 1
    
    # DFS with path tracking and deadline checking
    best_path = None
    search_state = {'found': False, 'path': [], 'k': 0}
    
    def dfs(remaining, k, path):
        if time.time() > deadline:
            return False
        if remaining == 0:
            if k == 0:
                search_state['found'] = True
                search_state['path'] = list(path)
                return True
            return False
        if k == 0:
            return False
        
        for vm in valid_masks:
            if (vm & remaining) == vm:  # vm is subset of remaining
                path.append(vm)
                if dfs(remaining ^ vm, k - 1, path):
                    return True
                path.pop()
        
        return False
    
    # Iterative deepening: try k=1, 2, 3, ...
    for k in range(1, n + 1):
        if time.time() > deadline:
            break
        
        search_state['found'] = False
        search_state['path'] = []
        search_state['k'] = k
        
        if dfs(full_mask, k, []):
            best_path = list(search_state['path'])
            break
    
    if best_path is None:
        # Fallback: each vertex as its own part
        best_path = [1 << i for i in range(n)]
    
    # Convert path to list of vertex lists
    result = []
    for mask in best_path:
        vertices = [i for i in range(n) if (mask >> i) & 1]
        result.append(vertices)
    
    # Ensure all elements are Python ints
    final_result = []
    for part in result:
        final_result.append([int(v) for v in part])
    
    return final_result

# EVOLVE-BLOCK-END
