"""Cochromatic partition solver with backtracking, bitmask checks, and deadline guard."""
import time

# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency bitmasks
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    # Precompute for each vertex: which vertices are NOT adjacent (excluding self)
    non_adj_mask = [0] * n
    for v in range(n):
        non_adj_mask[v] = ((1 << n) - 1) ^ (1 << v) ^ adj_mask[v]
    
    # Check if a bitmask represents a clique
    def is_clique(mask):
        temp = mask
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            others = mask ^ lsb
            if others & ~adj_mask[v]:
                return False
            temp ^= lsb
        return True
    
    # Check if a bitmask represents an independent set
    def is_independent(mask):
        temp = mask
        while temp:
            lsb = temp & (-temp)
            v = lsb.bit_length() - 1
            if mask & adj_mask[v]:
                return False
            temp ^= lsb
        return True
    
    # Cache for validity checks
    valid_cache = {}
    
    def is_valid_part(mask):
        if mask in valid_cache:
            return valid_cache[mask]
        result = is_clique(mask) or is_independent(mask)
        valid_cache[mask] = result
        return result
    
    # Greedy partition to get initial upper bound
    def greedy_partition():
        parts = []
        for v in range(n):
            vmask = 1 << v
            placed = False
            for i in range(len(parts)):
                trial = parts[i] | vmask
                if is_valid_part(trial):
                    parts[i] = trial
                    placed = True
                    break
            if not placed:
                parts.append(vmask)
        return parts
    
    initial_parts = greedy_partition()
    best_k = len(initial_parts)
    best_partition = initial_parts[:]
    
    # Deadline: 85 seconds (5s margin from 90s limit)
    deadline = time.time() + 85.0
    
    # Vertex ordering: sort by degree descending for better pruning
    vertex_order = sorted(range(n), key=lambda v: -len(adj_mask[v] & ((1 << n) - 1)))
    
    # Backtracking search to find partition with exactly k parts
    def search_with_k(k):
        """Try to find a valid cochromatic partition with exactly k parts."""
        parts = [0] * k
        assigned = 0
        # Track which part indices have been used (for symmetry breaking)
        # We enforce that parts are filled in order: if part[i] is empty, all parts[j] for j>i must also be empty
        
        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == n:
                # All vertices assigned
                # Verify all parts are non-empty
                for p in parts:
                    if p == 0:
                        return False
                return True
            
            v = vertex_order[idx]
            vmask = 1 << v
            
            # Try placing v in each existing non-empty part
            for i in range(k):
                if parts[i] == 0:
                    # If this part is empty, by symmetry we should only try the first empty part
                    break
                trial = parts[i] | vmask
                if is_valid_part(trial):
                    parts[i] = trial
                    if backtrack(idx + 1):
                        return True
                    parts[i] = trial ^ vmask
            
            # Try placing v in the first empty part (symmetry breaking: only first empty)
            for i in range(k):
                if parts[i] == 0:
                    parts[i] = vmask
                    if backtrack(idx + 1):
                        return True
                    parts[i] = 0
                    break
            
            return False
        
        return backtrack(0)
    
    # Try to find a better (smaller k) partition
    for k in range(best_k - 1, 1, -1):
        if time.time() > deadline:
            break
        if search_with_k(k):
            # Extract the partition
            best_k = k
            best_partition = parts[:]
            break
    
    # Convert bitmask partition to list of lists
    final_parts = []
    for p in best_partition:
        if p == 0:
            continue
        verts = []
        temp = p
        while temp:
            lsb = temp & (-temp)
            verts.append(lsb.bit_length() - 1)
            temp ^= lsb
        final_parts.append(verts)
    
    # Ensure all vertices are present exactly once
    all_verts = set()
    for part in final_parts:
        for v in part:
            all_verts.add(v)
    
    # If something went wrong, fall back to greedy
    if len(all_verts) != n:
        final_parts = []
        for p in initial_parts:
            verts = []
            temp = p
            while temp:
                lsb = temp & (-temp)
                verts.append(lsb.bit_length() - 1)
                temp ^= lsb
            final_parts.append(verts)
    
    # Ensure correct types: Python int, list
    result = []
    for part in final_parts:
        result.append([int(v) for v in part])
    
    return result

# EVOLVE-BLOCK-END
