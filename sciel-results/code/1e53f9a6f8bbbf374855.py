"""Suite task: maximize distinct degrees in an induced subgraph of the fixed 17-vertex host."""
# EVOLVE-BLOCK-START
import time
import itertools

P = {'n': 17, 'edges': [[0, 1], [0, 2], [0, 4], [0, 8], [0, 9], [0, 13], [0, 15], [0, 16], [1, 2], [1, 3], [1, 5], [1, 9], [1, 10], [1, 14], [1, 16], [2, 3], [2, 4], [2, 6], [2, 10], [2, 11], [2, 15], [3, 4], [3, 5], [3, 7], [3, 11], [3, 12], [3, 16], [4, 5], [4, 6], [4, 8], [4, 12], [4, 13], [5, 6], [5, 7], [5, 9], [5, 13], [5, 14], [6, 7], [6, 8], [6, 10], [6, 14], [6, 15], [7, 8], [7, 9], [7, 11], [7, 15], [7, 16], [8, 9], [8, 10], [8, 12], [8, 16], [9, 10], [9, 11], [9, 13], [10, 11], [10, 12], [10, 14], [11, 12], [11, 13], [11, 15], [12, 13], [12, 14], [12, 16], [13, 14], [13, 15], [14, 15], [14, 16], [15, 16]], 'homogeneous_cap': 4, 'minimum_vertices': 8}

N = P['n']
MIN_V = P['minimum_vertices']
EDGES = P['edges']

# Precompute full adjacency as a list of bitmasks for speed.
_ADJ = [0] * N
for a, b in EDGES:
    _ADJ[a] |= (1 << b)
    _ADJ[b] |= (1 << a)

# Precompute the set of edges as a bitmask pair for quick counting.
_EDGE_LIST = EDGES


def _distinct_degrees_in_subset(subset_mask):
    """Return the number of distinct degrees among vertices in subset_mask."""
    # subset_mask is an int bitmask over N vertices.
    degs = []
    m = subset_mask
    while m:
        lsb = m & (-m)
        v = lsb.bit_length() - 1
        # degree within the induced subgraph = popcount(adj[v] & subset_mask)
        d = bin(_ADJ[v] & subset_mask).count('1')
        degs.append(d)
        m ^= lsb
    return len(set(degs))


def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin before 90s kill

    best_mask = None
    best_score = 0

    # Build list of all vertex indices.
    vertices = list(range(N))

    # We must pick at least MIN_V distinct vertices.
    # To maximize distinct degrees, we want a subset where degrees spread out.
    # Strategy: enumerate subsets of size MIN_V up to N, but that's C(17,8)+... which is large.
    # C(17,8)=24310, C(17,9)=11628, C(17,10)=11628, ... total over all sizes 8..17 is 2^17 - (sum C(17,k) for k<8)
    # sum_{k=0}^{7} C(17,k) = 1+17+136+680+2380+6188+12376+19448 = 41226
    # 2^17 = 131072, so subsets of size >=8 = 131072-41226 = 89846. That's feasible.

    # Enumerate all subsets of size >= MIN_V via bitmask iteration.
    # Faster: iterate masks from 0 to 2^N, but filter by popcount >= MIN_V.
    # 2^17 = 131072 masks, each with a popcount check and degree computation.
    # Degree computation for a mask of size k takes O(k) with bit ops.
    # Total work ~ 131072 * avg_size ~ 131072 * 8.5 ~ 1.1M ops. Very fast.

    for mask in range(1, 1 << N):
        if time.time() > deadline:
            break
        # Quick popcount check
        pc = bin(mask).count('1')
        if pc < MIN_V:
            continue
        score = _distinct_degrees_in_subset(mask)
        if score > best_score:
            best_score = score
            best_mask = mask
            # Theoretical max distinct degrees for k vertices is k (all different).
            # But degrees range 0..k-1 in induced subgraph, so max distinct is k.
            # For k=17, max is 17. For k=8, max is 8.
            # If we reach pc distinct degrees, that's optimal for this size.
            # But we could do better with larger subsets. Keep searching.

    # Fallback: if we didn't find anything (shouldn't happen), return a default.
    if best_mask is None:
        # Return first MIN_V vertices.
        best_mask = (1 << MIN_V) - 1
        best_score = _distinct_degrees_in_subset(best_mask)

    # Extract vertices from best_mask.
    result = []
    m = best_mask
    while m:
        lsb = m & (-m)
        v = lsb.bit_length() - 1
        result.append(int(v))
        m ^= lsb
    result.sort()

    # Ensure it's a list of Python ints.
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
