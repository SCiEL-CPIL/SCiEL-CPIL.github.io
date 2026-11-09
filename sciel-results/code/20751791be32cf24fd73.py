"""Rewritten program: vectorized bitmask search for max distinct degrees in induced subgraph."""
import numpy as np
from itertools import combinations

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

def _popcount(x):
    x = x.astype(np.uint32)
    x = x - ((x >> 1) & 0x55555555)
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)
    x = (x + (x >> 4)) & 0x0F0F0F0F
    return ((x * 0x01010101) >> 24).astype(np.uint8)

def construct():
    n = int(P['n'])
    edges = P['edges']
    masks = np.zeros(n, dtype=np.uint32)
    for a, b in edges:
        masks[a] |= (1 << b)
        masks[b] |= (1 << a)

    all_vertices = list(range(n))
    best_deg = -1
    best_subset = None

    for k in range(n, int(P['minimum_vertices']) - 1, -1):
        if best_deg >= k:
            continue
            
        combos = list(combinations(all_vertices, k))
        bitmasks = np.array([sum(1 << v for v in c) for c in combos], dtype=np.uint32)
        n_sub = len(bitmasks)

        deg_matrix = np.empty((n_sub, n), dtype=np.uint8)
        active = np.empty((n_sub, n), dtype=bool)
        for v in range(n):
            deg_matrix[:, v] = _popcount(bitmasks & masks[v])
            active[:, v] = ((bitmasks >> v) & 1) == 1

        distinct = np.zeros(n_sub, dtype=np.uint8)
        for d in range(n):
            distinct += (((deg_matrix == d) & active).any(axis=1)).astype(np.uint8)

        idx = int(np.argmax(distinct))
        if int(distinct[idx]) > best_deg:
            best_deg = int(distinct[idx])
            best_subset = [int(v) for v in combos[idx]]

    return best_subset
