"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

N = P['n']
MIN_V = P['minimum_vertices']
edges = P['edges']

# Precompute adjacency bitmask for each vertex
adj = [0] * N
for u, v in edges:
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)

# Precompute popcount for all 2^17 masks
popcount = [0] * (1 << N)
for m in range(1, 1 << N):
    popcount[m] = popcount[m >> 1] + (m & 1)

# Precompute degree-in-subset for each vertex and each subset mask
# deg_in_subset[v][mask] = number of neighbors of v that are in mask
deg_in_subset = [[0] * (1 << N) for _ in range(N)]
for v in range(N):
    a = adj[v]
    for mask in range(1 << N):
        deg_in_subset[v][mask] = popcount[a & mask]

# Precompute list of vertices in each mask (for quick extraction)
# This is expensive in memory (131072 * avg ~8.5 verts), but we can compute on the fly
# Instead, precompute for each mask the sorted list of vertices
mask_vertices = [[] for _ in range(1 << N)]
for mask in range(1 << N):
    m = mask
    verts = []
    while m:
        v = (m & -m).bit_length() - 1
        verts.append(v)
        m &= m - 1
    mask_vertices[mask] = verts

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin under 90s
    
    # Best solution tracking
    best_count = 0
    best_mask = 0
    best_vertices = None
    
    # Iterate all masks with popcount >= MIN_V
    # Optimization: if we find a mask with popcount = p and distinct degrees = p,
    # that's the maximum possible for that size. But a larger mask could have more.
    # Max possible distinct degrees overall is N=17, but practically bounded.
    
    for mask in range(1 << N):
        if time.time() > deadline:
            break
        
        p = popcount[mask]
        if p < MIN_V:
            continue
        
        # Upper bound: distinct degrees <= p
        if p <= best_count:
            continue
        
        # Compute distinct degrees for this mask
        degs = set()
        for v in mask_vertices[mask]:
            degs.add(deg_in_subset[v][mask])
        
        cnt = len(degs)
        if cnt > best_count:
            best_count = cnt
            best_mask = mask
            best_vertices = mask_vertices[mask]
            
            # If we achieved the theoretical max for this subset size,
            # we can't do better with same-size subsets, but larger ones might
            # The global max is at most N
            if best_count == N:
                break
    
    if best_vertices is None:
        # Fallback: first MIN_V vertices
        best_vertices = list(range(MIN_V))
    
    # Ensure types are correct: list of Python ints
    return [int(v) for v in best_vertices]

# EVOLVE-BLOCK-END
