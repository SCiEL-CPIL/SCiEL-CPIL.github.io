"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

# Precompute adjacency as bitmasks
N = P['n']
adj = [0] * N
for u, v in P['edges']:
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)

# Precompute popcount for all 17-bit masks
# We can use a lookup table for 17 bits: 2^17 = 131072 entries
_popcount_table = [0] * (1 << N)
for i in range(1, 1 << N):
    _popcount_table[i] = _popcount_table[i >> 1] + (i & 1)

def _distinct_degrees(subset_mask):
    """Compute the number of distinct degrees in the induced subgraph on subset_mask."""
    verts = [i for i in range(N) if (subset_mask >> i) & 1]
    k = len(verts)
    if k < 2:
        return len(set([0] * k)) if k > 0 else 0
    
    # For each vertex in the subset, compute its degree within the induced subgraph
    degrees = []
    for v in verts:
        # Degree of v in induced subgraph = popcount(adj[v] & subset_mask)
        d = _popcount_table[adj[v] & subset_mask]
        degrees.append(d)
    
    return len(set(degrees))

# Precompute the best subset by doing exhaustive search at module load time
# This is feasible: sum(C(17,k) for k=8..17) = 131072 - sum(C(17,k) for k=0..7)
# sum(C(17,k) for k=0..7) = 1 + 17 + 136 + 680 + 2380 + 6188 + 12376 + 19448 = 41226
# So we need to check 131072 - 41226 = 89846 subsets, which is very fast.

def _precompute_best():
    """Find the subset of size >= 8 with maximum distinct degrees."""
    best_count = 0
    best_mask = 0
    
    for size in range(8, N + 1):
        for combo in itertools.combinations(range(N), size):
            mask = 0
            for v in combo:
                mask |= (1 << v)
            
            d = _distinct_degrees(mask)
            if d > best_count:
                best_count = d
                best_mask = mask
    
    return best_mask, best_count

# Do the precomputation at module load time
_BEST_MASK, _BEST_COUNT = _precompute_best()

def construct():
    """Return a list of at least 8 distinct vertices maximizing distinct degrees in induced subgraph."""
    # Convert the best mask to a list of vertices
    result = [i for i in range(N) if (_BEST_MASK >> i) & 1]
    return result

# EVOLVE-BLOCK-END
