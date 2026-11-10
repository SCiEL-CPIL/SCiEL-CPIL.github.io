"""Rewritten program: exact cochromatic number via branch-and-bound / DP."""
# EVOLVE-BLOCK-START
P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

import time

def _is_clique(vertices, adj_masks):
    """Bit-parallel clique check using adjacency bitmasks."""
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            if not (adj_masks[vertices[i]] >> vertices[j]) & 1:
                return False
    return True

def _is_independent(vertices, adj_masks):
    """Bit-parallel independent-set check using adjacency bitmasks."""
    for i in range(len(vertices)):
        for j in range(i + 1, len(vertices)):
            if (adj_masks[vertices[i]] >> vertices[j]) & 1:
                return False
    return True

def _is_valid_part(vertices, adj_masks):
    if not vertices:
        return False
    return _is_clique(vertices, adj_masks) or _is_independent(vertices, adj_masks)

def _cochromatic_number(n, adj_masks, deadline):
    """Exact cochromatic number via DP over subsets with bit-parallel checks."""
    
    # Precompute valid subsets (cliques or independent sets) using bit-parallel checks
    valid_subsets = {}
    for mask in range(1, 1 << n):
        if time.time() > deadline:
            return None
        vertices = [i for i in range(n) if mask & (1 << i)]
        if _is_valid_part(vertices, adj_masks):
            valid_subsets[mask] = vertices
    
    full_mask = (1 << n) - 1
    
    # DP: minimum number of valid subsets to cover all vertices
    # dp[mask] = min number of parts needed to cover vertices in mask
    INF = float('inf')
    dp = [INF] * (1 << n)
    dp[0] = 0
    
    # Also track the actual partition for reconstruction
    parent = [None] * (1 << n)
    
    for mask in range(1, 1 << n):
        if time.time() > deadline:
            return None
        # Try all valid subsets that are subsets of mask
        sub = mask
        while sub:
            if sub in valid_subsets:
                prev = mask ^ sub
                if dp[prev] + 1 < dp[mask]:
                    dp[mask] = dp[prev] + 1
                    parent[mask] = sub
            sub = (sub - 1) & mask
    
    if dp[full_mask] == INF:
        return None
    
    # Reconstruct partition
    parts = []
    mask = full_mask
    while mask:
        sub = parent[mask]
        if sub is None:
            return None
        parts.append(valid_subsets[sub])
        mask ^= sub
    
    return parts

def _greedy_cochromatic(n, adj_masks):
    """Greedy heuristic: assign vertices to existing parts or create new ones."""
    parts = []
    for v in range(n):
        placed = False
        for part in parts:
            trial = part + [v]
            if _is_valid_part(trial, adj_masks):
                part.append(v)
                placed = True
                break
        if not placed:
            parts.append([v])
    return parts

def construct():
    start = time.time()
    deadline = start + 85  # 85s safety margin within 90s limit
    
    n = P['n']
    # Build adjacency as bitmasks for fast bit-parallel checks
    adj_masks = [0] * n
    for u, v in P['edges']:
        adj_masks[u] |= (1 << v)
        adj_masks[v] |= (1 << u)
    
    # Try exact method
    result = _cochromatic_number(n, adj_masks, deadline)
    if result is not None:
        # Convert to list of lists of Python ints
        return [list(part) for part in result]
    
    # Fallback: greedy
    return _greedy_cochromatic(n, adj_masks)
# EVOLVE-BLOCK-END
