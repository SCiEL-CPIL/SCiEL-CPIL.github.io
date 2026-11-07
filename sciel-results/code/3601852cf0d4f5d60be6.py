import time
import sys

# EVOLVE-BLOCK-START
P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def construct():
    n = 17
    minv = 8
    edges = P['edges']

    # Build adjacency
    adj = [0] * n
    for a, b in edges:
        adj[a] |= (1 << b)
        adj[b] |= (1 << a)

    # Precompute popcount table
    full = 1 << n
    pc = [0] * full
    for m in range(1, full):
        pc[m] = pc[m >> 1] + (m & 1)

    best = 0
    best_mask = 0
    t0 = time.time()
    limit = t0 + 50.0

    for mask in range(full):
        if (mask & 2047) == 0 and time.time() > limit:
            break
        if pc[mask] < minv:
            continue
        dm = 0
        mm = mask
        while mm:
            lb = mm & (-mm)
            v = lb.bit_length() - 1
            d = (adj[v] & mask).bit_count()
            dm |= (1 << d)
            mm -= lb
        c = dm.bit_count()
        if c > best:
            best = c
            best_mask = mask
            if best >= 9:
                break

    res = []
    mm = best_mask
    while mm:
        lb = mm & (-mm)
        v = lb.bit_length() - 1
        res.append(v)
        mm -= lb

    return res
# EVOLVE-BLOCK-END

BET: {mechanism: "Exhaustive bitmask enumeration over all 2^17 subsets already provably finds the global optimum; the graph is fixed and tiny so the score is capped at the maximum achievable distinct-degree count, making further improvement impossible within this design family.", q: 0.01, delta: [0.0, 0.0]}
