import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Adjacency matrix as bitmasks
    # adj[i] is a bitmask of neighbors of i
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        
    full_mask = (1 << n) - 1
    
    # Precompute validity of all subsets
    # valid[mask] = True if mask is a clique or independent set
    valid = [False] * (1 << n)
    valid[0] = True
    valid[1] = True # Single vertex is both
    
    # Helper to check if a mask is a clique
    def is_clique(mask):
        if mask == 0: return True
        # Check all pairs
        # Iterate through set bits
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            # Check if all other vertices in mask are neighbors of v
            others = mask ^ lsb
            if others & ~adj[v] != 0:
                return False
        return True

    # Helper to check if a mask is an independent set
    def is_independent(mask):
        if mask == 0: return True
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            m ^= lsb
            # Check if no other vertex in mask is a neighbor of v
            others = mask ^ lsb
            if others & adj[v] != 0:
                return False
        return True

    # Precompute valid subsets
    # We can optimize this, but 2^14 is small
    for mask in range(1, 1 << n):
        # Optimization: if mask is not clique and not independent, it's invalid
        # We can check clique first, then independent
        if is_clique(mask):
            valid[mask] = True
        else:
            if is_independent(mask):
                valid[mask] = True
                
    # Find max size of a valid part to use for lower bound pruning
    max_valid_size = 0
    for mask in range(1, 1 << n):
        if valid[mask]:
            sz = bin(mask).count('1')
            if sz > max_valid_size:
                max_valid_size = sz
                
    best_partition = None
    best_count = n + 1 # Upper bound
    
    # Start time
    start_time = time.time()
    timeout = 55.0 # Safety margin
    
    # DFS with branch and bound
    # state: remaining_mask, current_parts (list of masks)
    # To avoid permutations, we always pick the lowest set bit in remaining_mask
    # and require it to be in the next part we choose.
    
    def dfs(remaining, current_parts):
        nonlocal best_partition, best_count
        
        # Time check
        if time.time() - start_time > timeout:
            return
            
        if remaining == 0:
            if len(current_parts) < best_count:
                best_count = len(current_parts)
                best_partition = list(current_parts)
            return
            
        # Pruning: if current parts + lower bound >= best_count, prune
        # Lower bound: ceil(popcount(remaining) / max_valid_size)
        rem_count = bin(remaining).count('1')
        lb = (rem_count + max_valid_size - 1) // max_valid_size
        if len(current_parts) + lb >= best_count:
            return
            
        # Pick the lowest set bit vertex
        lsb = remaining & -remaining
        v = lsb.bit_length() - 1
        
        # Iterate through all subsets S of remaining that contain v
        # and are valid
        # We can iterate subsets of (remaining ^ lsb) and add lsb
        rest = remaining ^ lsb
        
        # Generate subsets of rest
        sub = rest
        while True:
            S = sub | lsb
            if valid[S]:
                dfs(remaining ^ S, current_parts + [S])
            if sub == 0:
                break
            sub = (sub - 1) & rest
            
    dfs(full_mask, [])
    
    # If no solution found (should not happen), fallback to greedy or singletons
    if best_partition is None:
        # Fallback: each vertex is its own part
        best_partition = [[i] for i in range(n)]
        
    # Convert bitmasks to lists of vertices
    result = []
    for mask in best_partition:
        part = []
        m = mask
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            part.append(int(v))
            m ^= lsb
        result.append(part)
        
    return result

# EVOLVE-BLOCK-END
