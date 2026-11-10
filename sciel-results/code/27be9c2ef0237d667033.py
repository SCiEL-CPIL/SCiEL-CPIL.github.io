import sys
import time

def construct():
    # Define the graph parameters
    P = {
        'n': 14,
        'edges': [
            [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12],
            [1, 4], [1, 9], [1, 11],
            [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13],
            [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13],
            [4, 5], [4, 7], [4, 13],
            [5, 8], [5, 9], [5, 10], [5, 12], [5, 13],
            [6, 8], [6, 11], [6, 13],
            [7, 12],
            [8, 9], [8, 13],
            [9, 11], [9, 12],
            [11, 13]
        ]
    }
    
    n = P['n']
    edges = P['edges']
    
    # Deadline guard: 60 seconds limit, use 55 seconds to be safe
    start_time = time.time()
    deadline = start_time + 55.0
    
    # Build adjacency bitmasks
    adj = [0] * n
    for u, v in edges:
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
    
    full_mask = (1 << n) - 1
    
    # Precompute popcount is not strictly needed for logic, but useful for debugging or specific optimizations.
    # We will precompute is_clique and is_independent for all masks.
    
    is_clique = [False] * (1 << n)
    is_independent = [False] * (1 << n)
    is_clique[0] = True
    is_independent[0] = True
    
    # is_valid_part[m] is True if m is a clique or an independent set
    is_valid_part = [False] * (1 << n)
    
    # We compute properties iteratively.
    # For a mask to be a clique, if v is in mask, then all other vertices in mask must be adjacent to v.
    # Let lsb be the lowest set bit in mask, and rest = mask ^ lsb.
    # mask is clique iff rest is clique AND all vertices in rest are adjacent to v.
    # i.e., (rest & ~adj[v]) == 0
    
    # Similarly for independent set:
    # mask is independent iff rest is independent AND no vertices in rest are adjacent to v.
    # i.e., (rest & adj[v]) == 0
    
    for mask in range(1, 1 << n):
        lsb = mask & -mask
        v = lsb.bit_length() - 1
        rest = mask ^ lsb
        
        if is_clique[rest] and (rest & ~adj[v]) == 0:
            is_clique[mask] = True
        if is_independent[rest] and (rest & adj[v]) == 0:
            is_independent[mask] = True
            
        if is_clique[mask] or is_independent[mask]:
            is_valid_part[mask] = True

    # DP to find minimum number of parts
    # dp[mask] = minimum number of valid parts to partition mask
    # best_sub[mask] = the valid part (submask) chosen in the optimal solution for mask
    INF = 10**9
    dp = [INF] * (1 << n)
    best_sub = [0] * (1 << n)
    dp[0] = 0
    
    # To speed up, we can precompute a list of valid submasks for each mask?
    # No, iterating submasks on the fly is fine for N=14.
    
    for mask in range(1, 1 << n):
        # Check time
        if time.time() > deadline:
            # Fallback: return a greedy solution
            return _greedy_construct(P)
            
        lsb = mask & -mask
        sub = mask
        while sub:
            if (sub & lsb) != 0:
                if is_valid_part[sub]:
                    rest = mask ^ sub
                    candidate = 1 + dp[rest]
                    if candidate < dp[mask]:
                        dp[mask] = candidate
                        best_sub[mask] = sub
            sub = (sub - 1) & mask
            
    # Reconstruct the partition
    partition = []
    current_mask = full_mask
    while current_mask:
        sub = best_sub[current_mask]
        if sub == 0:
            # This shouldn't happen if dp was computed correctly, but handle gracefully
            # If stuck, break and add remaining as a single part (might be invalid if >1 vertex and not valid)
            # But since we found a solution, this shouldn't occur.
            break
        verts = []
        m = sub
        while m:
            l = m & -m
            verts.append(l.bit_length() - 1)
            m ^= l
        partition.append(verts)
        current_mask ^= sub
        
    # If the loop didn't cover everything (e.g., due to break), append remaining
    if current_mask:
        verts = []
        m = current_mask
        while m:
            l = m & -m
            verts.append(l.bit_length() - 1)
            m ^= l
        partition.append(verts)

    # Convert to list of lists of ints
    result = []
    for part in partition:
        result.append([int(v) for v in part])
        
    return result

def _greedy_construct(P):
    """Fallback greedy constructor."""
    n = P['n']
    edge_set = set()
    for u, v in P['edges']:
        edge_set.add((min(u, v), max(u, v)))
    
    parts = []
    # Sort vertices by degree descending for better greedy packing
    degrees = [0] * n
    for u, v in P['edges']:
        degrees[u] += 1
        degrees[v] += 1
    vertices = sorted(range(n), key=lambda x: degrees[x], reverse=True)
    
    for v in vertices:
        placed = False
        # Try to place v in an existing part
        for part in parts:
            is_cli = True
            is_ind = True
            for u in part:
                e = (min(u, v), max(u, v))
                if e in edge_set:
                    is_ind = False
                else:
                    is_cli = False
            if is_cli or is_ind:
                part.append(v)
                placed = True
                break
        if not placed:
            parts.append([v])
    
    # Return list of lists of ints
    return [[int(v) for v in part] for part in parts]

# EVOLVE-BLOCK-END
