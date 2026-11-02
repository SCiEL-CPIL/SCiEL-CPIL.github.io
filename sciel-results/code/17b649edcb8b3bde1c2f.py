"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

D = 11

# EVOLVE-BLOCK-START
import itertools
import time
import numpy as np

_DEADLINE = time.time() + 115


def construct():
    start = time.time()

    # All vectors of squared norm 4 in R^11 with integer entries:
    # Type A: 2*e_i, -2*e_i  -> 22 vectors
    # Type B: e_i + e_j, e_i - e_j, -e_i + e_j, -e_i - e_j for i<j -> 4*C(11,2)=220
    # Type C: 4 entries of +/-1 -> C(11,4)*16 = 330*16 = 5280
    # Total: 22 + 220 + 5280 = 5522 vectors

    # Generate all candidates
    candidates = []
    # Type A
    for i in range(D):
        v = [0] * D
        v[i] = 2
        candidates.append(v)
        v[i] = -2
        candidates.append(v)

    # Type B
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                candidates.append(v)

    # Type C
    for quad in itertools.combinations(range(D), 4):
        for signs in itertools.product((1, -1), repeat=4):
            v = [0] * D
            for idx, s in zip(quad, signs):
                v[idx] = s
            candidates.append(v)

    n = len(candidates)
    # n should be 5522

    # Compute dot product matrix (n x n)
    # Two vectors v, w of norm^2=4 conflict if ||v-w||^2 < 4, i.e., 8 - 2*v.w < 4, i.e., v.w > 2
    # So they are compatible if v.w <= 2
    # We want max independent set where edge = compatible (v.w <= 2)
    # Equivalently, max clique in the compatibility graph.

    # Build dot product matrix
    mat = np.array(candidates, dtype=np.int8)
    # Compute all pairwise dot products in chunks
    # dot[i][j] = mat[i] . mat[j]
    # For efficiency, compute mat @ mat.T
    dot = mat.astype(np.int32) @ mat.astype(np.int32).T
    # Compatibility: dot[i][j] <= 2 for i != j
    # Self: dot[i][i] = 4

    # Build adjacency as Python big integers (bitset per vertex)
    # adj[i] is a big int with bit j set if j is compatible with i (and j != i)
    adj = [0] * n
    for i in range(n):
        # Compatible with j if dot[i][j] <= 2 and j != i
        comp = dot[i] <= 2
        comp[i] = False  # exclude self
        # Convert boolean array to big int
        bits = 0
        for j in range(n):
            if comp[j]:
                bits |= (1 << j)
        adj[i] = bits

    # Now solve Maximum Clique on this graph using Tomita BB-MCS with bitsets
    # We use the standard branch-and-bound:
    # expand(R, P) where R is current clique size, P is candidate set (as bitset)
    # Order vertices by degree in P for better pruning

    # Precompute degree in the full graph
    deg = [bin(adj[i]).count('1') for i in range(n)]

    # Sort vertices by degree (ascending for the search order)
    order = list(range(n))
    order.sort(key=lambda x: deg[x])

    # Remap: new_index[v] = position in sorted order
    new_index = [0] * n
    for pos, v in enumerate(order):
        new_index[v] = pos

    # Rebuild adjacency in sorted order
    adj2 = [0] * n
    for i in range(n):
        old_i = order[i]
        bits = adj[old_i]
        new_bits = 0
        for j in range(n):
            if (bits >> j) & 1:
                new_bits |= (1 << new_index[j])
        adj2[i] = new_bits

    best = [0]
    best_set = [0]  # bitset of best clique found

    # Tomita BB-MCS: expand(R, P)
    # R: current clique size
    # P: candidate set as bitset (vertices that can be added)
    # We process P in order (by index in sorted array)

    def expand(R, P):
        if R + P.bit_count() <= best[0]:
            return
        if P == 0:
            if R > best[0]:
                best[0] = R
                best_set[0] = 0  # we'll track separately
            return

        # Process candidates in order from high index to low (or use the sorted order)
        # Standard approach: pick vertex v from P, branch on include/exclude
        # Use the ordering: process in reverse sorted order (highest degree first in P)

        while P != 0:
            if R + P.bit_count() <= best[0]:
                return

            # Pick the last set bit (highest index = highest degree in original)
            v = P.bit_length() - 1
            P &= ~(1 << v)

            # Include v: new candidates are P & adj2[v]
            P_new = P & adj2[v]
            expand(R + 1, P_new)

            # Exclude v: P already has v removed
            # Continue loop

    # Initial call: all vertices are candidates
    full_mask = (1 << n) - 1
    expand(0, full_mask)

    # best[0] is the max clique size
    # Now we need to actually extract the clique. Let's redo with tracking.
    # Actually, for the score we just need the count. But we need to return the vectors.
    # Let me re-implement to track the actual set.

    # Actually, let's just use the count for now and verify.
    # But we need to return actual vectors. Let me track the best clique.

    # Re-do with proper tracking
    best2 = [0]
    best_clique = [0]  # bitset

    def expand2(R, P, current_clique):
        if R + P.bit_count() <= best2[0]:
            return
        if P == 0:
            if R > best2[0]:
                best2[0] = R
                best_clique[0] = current_clique
            return

        while P != 0:
            if R + P.bit_count() <= best2[0]:
                return

            v = P.bit_length() - 1
            P &= ~(1 << v)

            P_new = P & adj2[v]
            expand2(R + 1, P_new, current_clique | (1 << v))

    # This might be too slow for 5522 vertices. Let me check time.
    if time.time() - start < 100:
        expand2(0, full_mask, 0)
    else:
        # Fallback: use greedy
        best2[0] = 0
        best_clique[0] = 0

    # Extract vectors from best_clique
    if best2[0] > 0:
        result = []
        for i in range(n):
            if (best_clique[0] >> i) & 1:
                result.append(candidates[order[i]])
        result_arr = np.array(result, dtype=np.int64)
        # Verify
        if len(result_arr) >= 2:
            # Check constraint
            norms2 = np.sum(result_arr * result_arr, axis=1)
            max_n2 = int(np.max(norms2))
            # Min pairwise distance
            n_r = len(result_arr)
            min_d2 = float('inf')
            chunk = 256
            for i in range(0, n_r, chunk):
                a = result_arr[i:i+chunk]
                for j in range(i+1, n_r):
                    diff = a - result_arr[j]
                    d2 = np.sum(diff * diff, axis=1)
                    m = int(np.min(d2))
                    if m < min_d2:
                        min_d2 = m
            if min_d2 >= max_n2:
                return result_arr
        # If verification fails, fall through

    # Fallback: greedy on the compatibility graph
    # Greedy: repeatedly pick vertex with max degree in remaining P
    selected = []
    remaining = set(range(n))
    # Use the sorted order for efficiency
    # Greedy: pick highest degree vertex, add it, remove incompatible vertices
    deg_remaining = {i: deg[i] for i in range(n)}
    # This is O(n^2) which might be slow for 5522. Let's use a simpler greedy.

    # Simpler greedy: iterate in order, add if compatible with all selected
    # But this gives suboptimal results. Let's try a few random orderings.
    import random
    best_greedy = 0
    best_greedy_set = []

    for trial in range(5):
        if time.time() - start > 105:
            break
        perm = list(range(n))
        random.shuffle(perm)
        sel = []
        sel_set = set()
        for v in perm:
            # Check compatibility with all selected
            ok = True
            for s in sel:
                if dot[v][s] > 2:
                    ok = False
                    break
            if ok:
                sel.append(v)
                sel_set.add(v)
        if len(sel) > best_greedy:
            best_greedy = len(sel)
            best_greedy_set = sel[:]

    if best_greedy > 0:
        result = [candidates[v] for v in best_greedy_set]
        result_arr = np.array(result, dtype=np.int64)
        # Verify
        if len(result_arr) >= 2:
            norms2 = np.sum(result_arr * result_arr, axis=1)
            max_n2 = int(np.max(norms2))
            n_r = len(result_arr)
            min_d2 = float('inf')
            chunk = 256
            for i in range(0, n_r, chunk):
                a = result_arr[i:i+chunk]
                for j in range(i+1, n_r):
                    diff = a - result_arr[j]
                    d2 = np.sum(diff * diff, axis=1)
                    m = int(np.min(d2))
                    if m < min_d2:
                        min_d2 = m
            if min_d2 >= max_n2:
                return result_arr

    # Ultimate fallback: the 220 weight-2 vectors
    base_vectors = []
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                base_vectors.append(v)
    return np.array(base_vectors, dtype=np.int64)
# EVOLVE-BLOCK-END
