"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def construct():
    n = P['n']
    edges = P['edges']
    deadline = time.time() + 80  # 80s deadline, 10s safety margin
    
    # Build adjacency masks
    adj_mask = [0] * n
    for u, v in edges:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
        
    # Precompute validity of all subsets
    valid = [False] * (1 << n)
    valid[0] = True
    max_part_size = 0
    
    for mask in range(1, 1 << n):
        is_clique = True
        is_ind = True
        m = mask
        while m:
            l = m & -m
            v = l.bit_length() - 1
            neighbors_in_mask = adj_mask[v] & mask
            if neighbors_in_mask != (mask ^ l):
                is_clique = False
            if neighbors_in_mask != 0:
                is_ind = False
            if not is_clique and not is_ind:
                break
            m ^= l
        if is_clique or is_ind:
            valid[mask] = True
            sz = mask.bit_count()
            if sz > max_part_size:
                max_part_size = sz
    
    full_mask = (1 << n) - 1
    best_result = None
    
    def dfs(remaining, parts, k):
        nonlocal best_result
        if time.time() > deadline:
            return False
        
        if remaining == 0:
            result = []
            for p in parts:
                verts = []
                m = p
                while m:
                    l = m & -m
                    v = l.bit_length() - 1
                    verts.append(v)
                    m ^= l
                result.append(verts)
            best_result = result
            return True
        
        # Lower bound pruning
        rem_count = remaining.bit_count()
        if len(parts) + math.ceil(rem_count / max_part_size) > k:
            return False
        
        # Select vertex with highest degree in remaining
        best_v = -1
        max_deg = -1
        temp = remaining
        while temp:
            l = temp & -temp
            v = l.bit_length() - 1
            deg = (adj_mask[v] & remaining).bit_count()
            if deg > max_deg:
                max_deg = deg
                best_v = v
            temp ^= l
        
        v_mask = 1 << best_v
        new_remaining = remaining ^ v_mask
        
        # Try adding to existing parts with symmetry breaking
        # Only allow adding to part i if it's the first part of its "size class"
        # to reduce symmetric solutions
        for i in range(len(parts)):
            new_part = parts[i] | v_mask
            if valid[new_part]:
                parts[i] = new_part
                # Maintain canonical ordering: sort parts by bitmask
                parts.sort()
                if dfs(new_remaining, parts, k):
                    return True
                parts[i] ^= v_mask
                parts.sort()
                
        # Try starting a new part
        if len(parts) < k:
            parts.append(v_mask)
            parts.sort()
            if dfs(new_remaining, parts, k):
                return True
            parts.pop()
            
        return False
    
    # Search for minimum k
    for k in range(1, n + 1):
        if time.time() > deadline:
            break
        parts = []
        if dfs(full_mask, parts, k):
            if best_result is not None:
                # Ensure all values are Python ints in lists
                result = [[int(v) for v in part] for part in best_result]
                return result
            break
    
    # Fallback: trivial partition
    return [[i] for i in range(n)]

# EVOLVE-BLOCK-END
