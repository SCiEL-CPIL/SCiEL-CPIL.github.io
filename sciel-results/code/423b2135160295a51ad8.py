"""XOR-circuit synthesis: minimum-cost arborescence via Dijkstra on subset hypergraph."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import heapq
from collections import defaultdict

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

    # Dijkstra-based global greedy: we want to build all required subsets
    # with minimum total gate count, sharing sub-expressions globally.
    # 
    # Strategy: Process subsets in order of "difficulty" (popcount), and for each
    # subset, find the cheapest way to express it as XOR of two already-available
    # (cached or input) subsets. This is a global greedy that tends to maximize
    # sharing.
    #
    # More sophisticated: Use a priority queue where we expand from known wires
    # to produce new wires, tracking cost.

    # Collect all unique non-trivial subsets needed
    all_needed = set()
    for s in row_sets:
        if len(s) > 1:
            all_needed.add(s)

    # We'll use a Dijkstra-like approach:
    # - Start with all singletons (cost 0) and empty set (cost 0)
    # - For each pair of known wires, their XOR is a new candidate wire
    # - We want to reach all needed subsets with minimum total gate count
    #
    # But the state space is 2^n which is too large. Instead, we use a 
    # greedy approach: process subsets in increasing popcount order, and for
    # each subset, find the best split into two cached subsets.

    # Sort needed subsets by popcount
    needed_sorted = sorted(all_needed, key=lambda s: len(s))

    def get_wire_greedy(subset):
        """Get wire for XOR of all elements in subset, using greedy split with cache."""
        nonlocal next_wire
        if len(subset) == 0:
            return zero_wire
        if len(subset) == 1:
            return next(iter(subset))
        if subset in wire_cache:
            return wire_cache[subset]

        elems = sorted(subset)
        k = len(elems)

        # Find best split: minimize total cost of sub-parts
        # Cost of a part: 0 if cached, else estimated (popcount - 1)
        best_w1 = None
        best_w2 = None
        best_cost = float('inf')
        best_depth = float('inf')

        for mid in range(1, k):
            s1 = frozenset(elems[:mid])
            s2 = frozenset(elems[mid:])

            c1 = s1 in wire_cache
            c2 = s2 in wire_cache

            # Cost: 0 if cached, else len-1 for balanced tree
            cost1 = 0 if c1 else max(0, len(s1) - 1)
            cost2 = 0 if c2 else max(0, len(s2) - 1)
            total_cost = cost1 + cost2

            # Depth estimate: 1 + max(depth1, depth2)
            d1 = 0 if c1 else max(1, int(np.ceil(np.log2(len(s1)))) if len(s1) > 1 else 0)
            d2 = 0 if c2 else max(1, int(np.ceil(np.log2(len(s2)))) if len(s2) > 1 else 0)
            total_depth = 1 + max(d1, d2)

            # Prefer lower cost, then lower depth
            if total_cost < best_cost or (total_cost == best_cost and total_depth < best_depth):
                best_cost = total_cost
                best_depth = total_depth
                best_w1 = s1
                best_w2 = s2

            # Early exit if both cached
            if c1 and c2:
                break

        s1 = best_w1
        s2 = best_w2

        w1 = get_wire_greedy(s1)
        w2 = get_wire_greedy(s2)

        if w1 == w2:
            result = zero_wire
        else:
            gates.append([w1, w2])
            result = next_wire
            next_wire += 1

        wire_cache[subset] = result
        return result

    # Global greedy with priority queue:
    # We process all needed subsets. For each, we try to find a representation
    # using already-computed wires. We use a priority queue to process subsets
    # in order of their "readiness" (how much of their structure is already cached).

    # Alternative: Two-pass approach
    # Pass 1: Compute all subsets of size 2 (all pairs from inputs that appear in rows)
    # Pass 2: Use those to build larger subsets

    # Let's use a more systematic approach:
    # 1. Identify all subsets that appear as rows
    # 2. Process them in increasing size
    # 3. For each, find the best decomposition using cached wires

    # First, let's pre-compute all size-2 subsets that are needed (directly or indirectly)
    # Actually, the greedy recursive approach above should work well.
    # Let's enhance it with a global view: before computing a large subset,
    # check if it can be formed from two smaller cached subsets.

    # Enhanced: For each needed subset, try all possible splits, not just
    # contiguous splits. This is expensive but more thorough.
    # For subsets of size up to ~20, we can try all 2^(k-1)-1 splits.

    def get_wire_enhanced(subset):
        """Get wire for subset using enhanced split search."""
        nonlocal next_wire
        if len(subset) == 0:
            return zero_wire
        if len(subset) == 1:
            return next(iter(subset))
        if subset in wire_cache:
            return wire_cache[subset]

        elems = sorted(subset)
        k = len(elems)

        # NEW: Attempt to compose 'subset' from two already-cached subsets.
        # This maximizes sharing and reduces gate count by avoiding redundant sub-tree construction.
        best_cached_split = None
        min_split_cost = float('inf')
        
        # Iterate through existing cached subsets to find a partition
        # Complexity: O(CacheSize * k). For n=16, CacheSize < 2^16, this is fast enough.
        for cached_sub in wire_cache.keys():
            if cached_sub == subset or len(cached_sub) == 0:
                continue
            # Check if cached_sub is a proper subset of 'subset'
            if cached_sub.issubset(subset):
                remainder = subset - cached_sub
                # Check if the remainder is also cached
                if remainder in wire_cache:
                    # Valid 1-gate composition found
                    # Heuristic: Prefer splits that balance the sizes or have lower depth
                    # Using size balance as a proxy for depth
                    cost_metric = max(len(cached_sub), len(remainder))
                    if cost_metric < min_split_cost:
                        min_split_cost = cost_metric
                        best_cached_split = (cached_sub, remainder)

        if best_cached_split:
            s1, s2 = best_cached_split
            w1 = wire_cache[s1]
            w2 = wire_cache[s2]
            if w1 == w2:
                result = zero_wire
            else:
                gates.append([w1, w2])
                result = next_wire
                next_wire += 1
            wire_cache[subset] = result
            return result

        if k <= 12:
            # Try all possible splits (all 2^(k-1)-1 ways to partition)
            best_w1 = None
            best_w2 = None
            best_cost = float('inf')
            best_depth = float('inf')

            # Represent subset as bitmask for efficiency
            # For k <= 12, 2^k = 4096, manageable
            elem_set = set(elems)

            for mask in range(1, (1 << k) - 1):
                # Skip masks where both halves are empty (mask=0) or one is full (mask=all)
                if mask == 0 or mask == (1 << k) - 1:
                    continue

                s1 = frozenset(elems[i] for i in range(k) if (mask >> i) & 1)
                s2 = frozenset(elems[i] for i in range(k) if not ((mask >> i) & 1))

                if len(s1) == 0 or len(s2) == 0:
                    continue

                c1 = s1 in wire_cache
                c2 = s2 in wire_cache

                cost1 = 0 if c1 else max(0, len(s1) - 1)
                cost2 = 0 if c2 else max(0, len(s2) - 1)
                total_cost = cost1 + cost2

                d1 = 0 if c1 else (0 if len(s1) == 1 else int(np.ceil(np.log2(len(s1)))))
                d2 = 0 if c2 else (0 if len(s2) == 1 else int(np.ceil(np.log2(len(s2)))))
                total_depth = 1 + max(d1, d2)

                if total_cost < best_cost or (total_cost == best_cost and total_depth < best_depth):
                    best_cost = total_cost
                    best_depth = total_depth
                    best_w1 = s1
                    best_w2 = s2

                # Early exit if both cached
                if c1 and c2:
                    break

            if best_w1 is None:
                # Fallback: contiguous split
                mid = k // 2
                best_w1 = frozenset(elems[:mid])
                best_w2 = frozenset(elems[mid:])
        else:
            # For larger subsets, use contiguous splits only (too many to try all)
            best_w1 = None
            best_w2 = None
            best_cost = float('inf')
            best_depth = float('inf')

            for mid in range(1, k):
                s1 = frozenset(elems[:mid])
                s2 = frozenset(elems[mid:])

                c1 = s1 in wire_cache
                c2 = s2 in wire_cache

                cost1 = 0 if c1 else max(0, len(s1) - 1)
                cost2 = 0 if c2 else max(0, len(s2) - 1)
                total_cost = cost1 + cost2

                d1 = 0 if c1 else (0 if len(s1) == 1 else int(np.ceil(np.log2(len(s1)))))
                d2 = 0 if c2 else (0 if len(s2) == 1 else int(np.ceil(np.log2(len(s2)))))
                total_depth = 1 + max(d1, d2)

                if total_cost < best_cost or (total_cost == best_cost and total_depth < best_depth):
                    best_cost = total_cost
                    best_depth = total_depth
                    best_w1 = s1
                    best_w2 = s2

                if c1 and c2:
                    break

            if best_w1 is None:
                mid = k // 2
                best_w1 = frozenset(elems[:mid])
                best_w2 = frozenset(elems[mid:])

        w1 = get_wire_enhanced(best_w1)
        w2 = get_wire_enhanced(best_w2)

        if w1 == w2:
            result = zero_wire
        else:
            gates.append([w1, w2])
            result = next_wire
            next_wire += 1

        wire_cache[subset] = result
        return result

    # Process non-zero rows sorted by popcount ascending
    non_zero_rows = [i for i in range(n_rows) if len(row_sets[i]) > 0]
    non_zero_rows.sort(key=lambda i: len(row_sets[i]))

    for i in non_zero_rows:
        s = row_sets[i]
        if s not in wire_cache and len(s) > 1:
            get_wire_enhanced(s)

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
            outputs.append(get_wire_enhanced(s))

    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
