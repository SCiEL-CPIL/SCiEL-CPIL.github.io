"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from functools import lru_cache
from itertools import combinations

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def _build_adjacency_bitmasks():
    n = P['n']
    adj = [0] * n
    for u, v in P['edges']:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    return adj

def _is_clique(mask, adj):
    # A set is a clique if for every vertex v in the set, all other vertices in the set are in adj[v]
    # Equivalently, the intersection of adj[v] for all v in mask should contain (mask ^ (1<<v))
    # Or simpler: for all v in mask, (mask & ~adj[v] & ~(1<<v)) == 0
    # Actually, adj[v] does not contain v. So we need (mask & ~adj[v]) to only have v itself.
    # So (mask & ~adj[v] & ~(1<<v)) must be 0.
    m = mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        # Check if all other vertices in mask are adjacent to v
        others = mask ^ lsb
        if (others & ~adj[v]) != 0:
            return False
        m ^= lsb
    return True

def _is_independent(mask, adj):
    # A set is independent if for every vertex v in the set, no other vertex in the set is in adj[v]
    # So (mask & adj[v]) must be 0 for all v in mask? No, adj[v] contains neighbors.
    # So (mask & adj[v]) should be 0.
    m = mask
    while m:
        lsb = m & -m
        v = lsb.bit_length() - 1
        if (mask & adj[v]) != 0:
            return False
        m ^= lsb
    return True

def construct():
    start_time = time.time()
    n = P['n']
    full_mask = (1 << n) - 1
    adj = _build_adjacency_bitmasks()
    
    # Precompute valid subsets
    # valid_subs is a list of masks that are cliques or independent sets
    valid_subs = []
    for mask in range(1, 1 << n):
        if _is_clique(mask, adj) or _is_independent(mask, adj):
            valid_subs.append(mask)
    
    # DP to find minimum number of parts
    # dp[mask] = min parts to cover mask
    # parent[mask] = the valid subset chosen for the last step in the optimal solution for mask
    # We use -1 as infinity
    INF = n + 1
    dp = [INF] * (1 << n)
    parent = [0] * (1 << n)
    dp[0] = 0
    
    # We iterate masks from 1 to full_mask
    # For each mask, we pick the least significant set bit (lsb) to break symmetry
    # We only consider valid subsets S such that:
    # 1. S is a subset of mask
    # 2. S contains the lsb
    # This ensures that we build the partition in a canonical order (always taking the part containing the smallest available vertex first).
    
    # Optimization: Group valid_subs by their lsb?
    # Or just iterate all valid_subs for each mask?
    # Number of masks: 2^14 = 16384
    # Number of valid_subs: likely a few thousand.
    # 16384 * ~1000 = 16M operations. This is fine in Python if optimized, but might be slow.
    
    # Better: Pre-filter valid_subs by lsb.
    # valid_subs_by_lsb[lsb_index] = list of masks in valid_subs that have lsb_index as their least significant bit.
    valid_subs_by_lsb = [[] for _ in range(n)]
    for s in valid_subs:
        lsb_idx = (s & -s).bit_length() - 1
        valid_subs_by_lsb[lsb_idx].append(s)
        
    # DP
    for mask in range(1, 1 << n):
        # Find the least significant set bit in mask
        lsb = mask & -mask
        lsb_idx = lsb.bit_length() - 1
        
        # Try all valid subsets S that contain lsb and are subsets of mask
        # S must be from valid_subs_by_lsb[lsb_idx]
        # And (S & mask) == S
        best_cost = INF
        best_s = 0
        
        for s in valid_subs_by_lsb[lsb_idx]:
            if (s & mask) == s:
                # s is a valid subset of mask containing lsb
                prev_mask = mask ^ s
                cost = dp[prev_mask] + 1
                if cost < best_cost:
                    best_cost = cost
                    best_s = s
        
        dp[mask] = best_cost
        parent[mask] = best_s
        
    # Reconstruct the partition
    parts = []
    current_mask = full_mask
    while current_mask:
        s = parent[current_mask]
        if s == 0:
            # This should not happen if a solution exists
            break
        # Convert bitmask s to list of vertices
        part_vertices = []
        m = s
        while m:
            lsb = m & -m
            v = lsb.bit_length() - 1
            part_vertices.append(v)
            m ^= lsb
        parts.append(part_vertices)
        current_mask ^= s
        
    # The parts are found in reverse order of selection (largest remaining mask first).
    # The order of parts in the list doesn't matter for the validity, only the content.
    # However, we must ensure we return a list of lists of ints.
    
    # Verify result
    if dp[full_mask] >= INF:
        # Fallback: should not happen for this graph, but just in case
        # Return each vertex as a separate part
        return [[i] for i in range(n)]
        
    return parts

# EVOLVE-BLOCK-END
