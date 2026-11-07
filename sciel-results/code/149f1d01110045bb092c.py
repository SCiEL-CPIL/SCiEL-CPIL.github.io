"""Cochromatic partition solver with backtracking and deadline guard."""
import time
import itertools

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency structure
    adj = [set() for _ in range(n)]
    non_adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    for u in range(n):
        for v in range(n):
            if u != v and v not in adj[u]:
                non_adj[u].add(v)
    
    # Precompute clique and independent set checks for subsets
    # We'll use bitmasks for efficiency
    # For each vertex, its adjacency bitmask
    adj_mask = [0] * n
    for u in range(n):
        for v in adj[u]:
            adj_mask[u] |= (1 << v)
    
    # Function to check if a set (bitmask) is a clique
    def is_clique(mask):
        # A set is a clique if every pair is connected
        # Equivalent: for each vertex v in mask, (mask without v) subset of adj[v]
        # Or: mask & adj[v] contains all other vertices in mask
        temp = mask
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            # Check if all other vertices in mask are adjacent to v
            if (mask ^ lsb) & ~adj_mask[v]:
                return False
            temp ^= lsb
        return True
    
    # Function to check if a set (bitmask) is an independent set
    def is_independent(mask):
        # A set is independent if no two vertices are adjacent
        # Equivalent: for each vertex v in mask, mask & adj[v] == 0 (excluding v itself)
        temp = mask
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            if mask & adj_mask[v]:
                return False
            temp ^= lsb
        return True
    
    # Determine for each subset whether it's a clique, independent, or neither
    # We'll cache these
    clique_cache = {}
    indep_cache = {}
    
    def is_valid_part(mask):
        if mask in clique_cache:
            return clique_cache[mask]
        if mask in indep_cache:
            return indep_cache[mask]
        if is_clique(mask):
            clique_cache[mask] = True
            return True
        if is_independent(mask):
            indep_cache[mask] = True
            return True
        clique_cache[mask] = False
        indep_cache[mask] = False
        return False
    
    # Start with a greedy solution to get an upper bound
    # Greedy: assign each vertex to an existing part if possible, else create new
    def greedy_partition():
        parts = []  # list of bitmasks
        for v in range(n):
            vmask = 1 << v
            placed = False
            for i, part in enumerate(parts):
                trial = part | vmask
                if is_valid_part(trial):
                    parts[i] = trial
                    placed = True
                    break
            if not placed:
                parts.append(vmask)
        return parts
    
    initial_parts = greedy_partition()
    best_num_parts = len(initial_parts)
    best_partition = initial_parts[:]
    
    # Try to find a better partition using backtracking
    # We'll try to find a partition with k parts for k < best_num_parts
    
    deadline = time.time() + 85  # 85 seconds, leaving 5s margin
    
    # Order vertices by degree (descending) for better pruning
    vertex_order = sorted(range(n), key=lambda v: -len(adj[v]))
    
    # Backtracking search for cochromatic number
    def backtrack_search(k_max):
        """Try to find a valid partition with at most k_max parts."""
        # parts[i] is a bitmask for part i, or 0 if not yet assigned
        parts = [0] * k_max
        # Track which vertices are assigned
        assigned_mask = 0
        
        def search(idx):
            if time.time() > deadline:
                return None
            if idx == len(vertex_order):
                # All vertices assigned
                # Convert to list of lists
                result = []
                for p in parts:
                    if p:
                        verts = []
                        temp = p
                        while temp:
                            lsb = temp & (-temp)
                            verts.append(lsb.bit_length() - 1)
                            temp ^= lsb
                        result.append(verts)
                return result
            
            v = vertex_order[idx]
            vmask = 1 << v
            
            # Try placing v in each existing part
            for i in range(k_max):
                if parts[i] == 0:
                    # Empty part
                    parts[i] = vmask
                    res = search(idx + 1)
                    if res is not None:
                        return res
                    parts[i] = 0
                else:
                    trial = parts[i] | vmask
                    if is_valid_part(trial):
                        parts[i] = trial
                        res = search(idx + 1)
                        if res is not None:
                            return res
                        parts[i] = trial ^ vmask
            
            # Try creating a new part (if we haven't used all k_max parts)
            # Find the first empty part slot
            for i in range(k_max):
                if parts[i] == 0:
                    parts[i] = vmask
                    res = search(idx + 1)
                    if res is not None:
                        return res
                    parts[i] = 0
                    break
            
            return None
        
        return search(0)
    
    # Try smaller numbers of parts
    for k in range(best_num_parts - 1, 0, -1):
        if time.time() > deadline:
            break
        result = backtrack_search(k)
        if result is not None:
            best_partition = result
            best_num_parts = k
            break
    
    # Convert best_partition to list of lists of ints
    if isinstance(best_partition, list) and best_partition and isinstance(best_partition[0], int):
        # It's a list of bitmasks
        final_parts = []
        for p in best_partition:
            verts = []
            temp = p
            while temp:
                lsb = temp & (-temp)
                verts.append(lsb.bit_length() - 1)
                temp ^= lsb
            final_parts.append(verts)
    else:
        # It's already a list of lists
        final_parts = best_partition
    
    # Ensure types are correct: Python int, list
    final_result = []
    for part in final_parts:
        final_result.append([int(v) for v in part])
    
    return final_result

# EVOLVE-BLOCK-END
