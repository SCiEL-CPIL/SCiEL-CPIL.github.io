"""Precomputed lookup table over all 2^17 masks to maximize distinct degrees in induced subgraph."""
# EVOLVE-BLOCK-START
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}


def construct():
    n = P['n']
    edges = P['edges']
    minv = P['minimum_vertices']
    deadline = time.time() + 85  # safety margin before 90s kill

    # Precompute adjacency as integer bitmasks
    adj = [0] * n
    for a, b in edges:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)

    # Precompute popcount for all 17-bit values (0 to 2^17 - 1)
    # This is fast: 131072 entries
    pc = [0] * (1 << n)
    for i in range(1, 1 << n):
        pc[i] = pc[i >> 1] + (i & 1)

    # For each mask, we need the set of degrees of vertices in the mask
    # degree of vertex v in mask = popcount(adj[v] & mask)
    # We want to maximize the number of distinct degrees among vertices in the mask,
    # subject to popcount(mask) >= minv

    # Strategy: iterate over all masks, compute distinct degree count, track best
    # To speed up: for each mask, the degrees are pc[adj[v] & mask] for v in mask
    # We can use a bitset of seen degrees (max degree is 16, so 17 bits)

    best_count = -1
    best_mask = 0

    # Iterate over all masks
    for mask in range(1, 1 << n):
        if time.time() > deadline:
            break
        # Check popcount of mask
        if pc[mask] < minv:
            continue
        # Compute distinct degrees
        deg_bits = 0  # bit i set if degree i appears
        m = mask
        while m:
            v = (m & -m).bit_length() - 1  # lowest set bit index
            d = pc[adj[v] & mask]
            deg_bits |= (1 << d)
            m &= m - 1
        c = pc[deg_bits]  # number of distinct degrees
        if c > best_count:
            best_count = c
            best_mask = mask

    # Convert best_mask to list of vertex indices
    out = []
    m = best_mask
    while m:
        v = (m & -m).bit_length() - 1
        out.append(int(v))
        m &= m - 1

    return out
# EVOLVE-BLOCK-END
