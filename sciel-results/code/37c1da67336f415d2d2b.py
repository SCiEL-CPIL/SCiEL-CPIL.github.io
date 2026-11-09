"""Rewritten program using bitmask backtracking for cochromatic partition."""
# EVOLVE-BLOCK-START
import itertools

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

N = P['n']
adj_mask = [0] * N
non_adj_mask = [0] * N
full_mask = (1 << N) - 1

for i in range(N):
    non_adj_mask[i] = full_mask & ~(1 << i)

for u, v in P['edges']:
    adj_mask[u] |= (1 << v)
    adj_mask[v] |= (1 << u)
    non_adj_mask[u] &= ~(1 << v)
    non_adj_mask[v] &= ~(1 << u)

def get_vertices(mask):
    verts = []
    while mask:
        lsb = mask & -mask
        i = lsb.bit_length() - 1
        verts.append(i)
        mask ^= lsb
    return verts

best_k = N + 1
best_parts = None

def dfs(rem, parts):
    global best_k, best_parts
    
    if rem == 0:
        if len(parts) < best_k:
            best_k = len(parts)
            best_parts = [p for p in parts]
        return

    # Prune if we can't beat current best
    if len(parts) >= best_k:
        return

    # Select vertex with max degree in remaining subgraph
    v = -1
    max_deg = -1
    temp_rem = rem
    while temp_rem:
        lsb = temp_rem & -temp_rem
        u = lsb.bit_length() - 1
        deg = (adj_mask[u] & rem).bit_count()
        if deg > max_deg:
            max_deg = deg
            v = u
        temp_rem ^= lsb
    
    if v == -1:
        return

    v_bit = 1 << v
    new_rem = rem & ~v_bit

    # Try adding v to existing parts
    for i in range(len(parts)):
        p_type, p_mask = parts[i]
        if p_type == 0: # Clique
            if (p_mask & adj_mask[v]) == p_mask:
                parts[i] = (0, p_mask | v_bit)
                dfs(new_rem, parts)
                parts[i] = (0, p_mask)
        else: # Independent
            if (p_mask & non_adj_mask[v]) == p_mask:
                parts[i] = (1, p_mask | v_bit)
                dfs(new_rem, parts)
                parts[i] = (1, p_mask)
        
        # If we found a solution with best_k == len(parts), we can't do better by adding to existing?
        # Actually, if best_k is updated to len(parts), we can prune all further branches from this level?
        # No, because we are at level len(parts). We found a solution with len(parts) parts.
        # Any other branch from this level will result in >= len(parts) parts.
        # So if best_k == len(parts), we can return immediately.
        if best_k == len(parts):
            return

    # Try adding v to a new part
    # We need at least 1 more part, so if len(parts) + 1 >= best_k, we can't improve
    if len(parts) + 1 < best_k:
        # New Clique
        parts.append((0, v_bit))
        dfs(new_rem, parts)
        parts.pop()
        
        # If best_k == len(parts) + 1, we can't do better with a new part?
        # If we found a solution with len(parts)+1 parts, we can't find one with < len(parts)+1 from here?
        # We are trying to add a new part, so the count becomes len(parts)+1.
        # If best_k is len(parts)+1, we matched the best, but didn't beat it.
        # We can prune if best_k <= len(parts) + 1.
        if best_k <= len(parts) + 1:
            return

        # New Independent
        parts.append((1, v_bit))
        dfs(new_rem, parts)
        parts.pop()

def construct():
    global best_k, best_parts
    best_k = N + 1
    best_parts = None
    
    dfs(full_mask, [])
    
    if best_parts is None:
        return [[i] for i in range(N)]
        
    result = []
    for p_type, p_mask in best_parts:
        verts = get_vertices(p_mask)
        result.append(verts)
    return result

# EVOLVE-BLOCK-END
