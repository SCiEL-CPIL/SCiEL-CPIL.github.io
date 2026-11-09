"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def synthesize(matrix, depth_limit):
    start_time = time.time()
    DEADLINE = start_time + 85  # safety margin before 90s kill

    mat = np.array(matrix, dtype=np.int32)
    n_rows, n_cols = mat.shape
    n = n_cols
    zero_wire = n  # constant-zero wire

    # For each row, get the set of input indices where value is 1
    row_sets = []
    for i in range(n_rows):
        cols = np.nonzero(mat[i])[0]
        row_sets.append(frozenset(cols.tolist()))

    # Gate management
    gates = []
    next_wire = n + 1
    wire_cache = {}  # frozenset -> wire index
    # Also track singletons
    for i in range(n):
        wire_cache[frozenset([i])] = i

    def get_wire(subset):
        """Get wire for XOR of all elements in subset, building balanced tree with cache reuse."""
        nonlocal next_wire
        if len(subset) == 0:
            return zero_wire
        if len(subset) == 1:
            return next(iter(subset))
        if subset in wire_cache:
            return wire_cache[subset]

        elems = sorted(subset)
        k = len(elems)

        # Try to find a split that maximizes cache reuse
        # Prefer splits where both halves are cached or have small estimated cost
        best_w1 = None
        best_w2 = None
        best_cost = float('inf')

        for mid in range(1, k):
            s1 = frozenset(elems[:mid])
            s2 = frozenset(elems[mid:])

            c1 = s1 in wire_cache
            c2 = s2 in wire_cache

            # Cost estimate: 0 if cached, else popcount - 1 for chain
            cost1 = 0 if c1 else max(0, len(s1) - 1)
            cost2 = 0 if c2 else max(0, len(s2) - 1)
            total_cost = cost1 + cost2

            if total_cost < best_cost:
                best_cost = total_cost
                best_w1 = s1
                best_w2 = s2

            if c1 and c2:
                break

        s1 = best_w1
        s2 = best_w2

        w1 = get_wire(s1)
        w2 = get_wire(s2)

        if w1 == w2:
            result = zero_wire
        else:
            gates.append([w1, w2])
            result = next_wire
            next_wire += 1

        wire_cache[subset] = result
        return result

    # Phase 1: Synthesize all non-zero rows sorted by popcount ascending
    # This builds small sub-expressions first, maximizing cache reuse for larger sets
    non_zero_rows = [i for i in range(n_rows) if len(row_sets[i]) > 0]
    non_zero_rows.sort(key=lambda i: len(row_sets[i]))
    
    for i in non_zero_rows:
        s = row_sets[i]
        if s not in wire_cache:
            get_wire(s)

    # Build outputs
    outputs = []
    for i in range(n_rows):
        s = row_sets[i]
        if len(s) == 0:
            outputs.append(zero_wire)
        elif s in wire_cache:
            outputs.append(wire_cache[s])
        elif len(s) == 1:
            outputs.append(next(iter(s)))
        else:
            outputs.append(get_wire(s))

    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
