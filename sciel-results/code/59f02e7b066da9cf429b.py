import numpy as np
import time
import itertools

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin before 150s limit

    # 1. Generate Type A vectors (one non-zero entry +/- 2)
    type_a_vectors = []
    for i in range(D):
        for sign in (2, -2):
            v = np.zeros(D, dtype=np.int64)
            v[i] = sign
            type_a_vectors.append(v)
    type_a_vectors = np.array(type_a_vectors)

    # 2. Generate Type B vectors (four non-zero entries +/- 1)
    type_b_vectors = []
    for support in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = np.zeros(D, dtype=np.int64)
            for idx, sign in zip(support, signs):
                v[idx] = sign
            type_b_vectors.append(v)
    type_b_vectors = np.array(type_b_vectors)

    n_b = len(type_b_vectors)

    # 3. Compute incompatibility graph for Type B
    dot_products = type_b_vectors @ type_b_vectors.T
    adjacency = (dot_products >= 3)
    np.fill_diagonal(adjacency, False)

    # Convert adjacency to bitsets for fast operations
    # Use Python's arbitrary-precision integers as bitsets
    n_bits = n_b
    adj_bitsets = [0] * n_b
    for i in range(n_bits):
        mask = 0
        for j in range(n_bits):
            if adjacency[i, j]:
                mask |= (1 << j)
        adj_bitsets[i] = mask

    # Precompute degree for each vertex
    degrees = [bin(adj_bitsets[i]).count('1') for i in range(n_bits)]

    # Bitset-based branch-and-bound for Maximum Independent Set
    # Tomita-style: color bound + branching on min-degree vertex
    def bitset_mis(adj_bitsets, n, time_limit, start_time):
        """
        Find a large independent set using bitset branch-and-bound.
        Returns the set of selected indices.
        """
        best_set = []
        best_size = 0

        def popcount(x):
            return bin(x).count('1')

        def lowest_bit(x):
            if x == 0:
                return -1
            return (x & -x).bit_length() - 1

        def solve(P, Q, current_set, current_size):
            nonlocal best_set, best_size
            if time.time() > time_limit:
                return

            if P == 0:
                if current_size > best_size:
                    best_size = current_size
                    best_set = current_set[:]
                return

            # Upper bound: current_size + popcount(P)
            if current_size + popcount(P) <= best_size:
                return

            # Color bound: greedy coloring of P to get upper bound
            # We'll use a simple greedy coloring
            remaining = P
            color_classes = []
            while remaining:
                # Pick a vertex with minimum degree in remaining
                # For speed, just pick the lowest bit
                v = lowest_bit(remaining)
                if v < 0:
                    break
                # Color class: all vertices in remaining not adjacent to v
                # Actually for color bound, we want to find a proper coloring
                # Simple approach: greedy sequential coloring
                color_classes.append(0)
                remaining &= ~(1 << v)

            # Better color bound: use greedy coloring
            # Let's implement a proper greedy coloring
            def greedy_color_bound(P):
                """Returns number of color classes needed (upper bound on independence number)"""
                uncolored = P
                colors = 0
                while uncolored:
                    # Find a maximal independent set in uncolored
                    # Use greedy: pick lowest bit, remove it and its neighbors
                    cis = 0
                    temp = uncolored
                    while temp:
                        v = lowest_bit(temp)
                        if v < 0:
                            break
                        cis |= (1 << v)
                        # Remove v and all neighbors of v from temp
                        temp &= ~(1 << v)
                        temp &= ~adj_bitsets[v]
                    uncolored &= ~cis
                    colors += 1
                return colors

            color_bound = greedy_color_bound(P)
            if current_size + color_bound <= best_size:
                return

            # Branch: pick vertex with minimum degree in P
            # For efficiency, pick lowest bit
            v = lowest_bit(P)
            if v < 0:
                return

            # Branch 1: include v
            new_P = P & ~(1 << v) & ~adj_bitsets[v]
            current_set.append(v)
            solve(new_P, Q, current_set, current_size + 1)
            current_set.pop()

            # Branch 2: exclude v
            P &= ~(1 << v)
            solve(P, Q, current_set, current_size)

        # Initial call
        all_vertices = (1 << n) - 1
        solve(all_vertices, all_vertices, [], 0)
        return best_set

    # Try multiple strategies to find a good MIS
    best_mis_indices = []
    best_mis_size = 0

    # Strategy 1: Static min-degree greedy (fast baseline)
    available = np.ones(n_b, dtype=bool)
    selected_indices = []
    while np.any(available):
        if time.time() > deadline:
            break
        avail_indices = np.where(available)[0]
        if len(avail_indices) == 0:
            break
        avail_degrees = degrees[avail_indices]
        min_deg_idx_in_avail = np.argmin(avail_degrees)
        chosen_global_idx = avail_indices[min_deg_idx_in_avail]
        selected_indices.append(chosen_global_idx)
        available[chosen_global_idx] = False
        neighbors = np.where(adjacency[chosen_global_idx])[0]
        available[neighbors] = False

    if len(selected_indices) > best_mis_size:
        best_mis_size = len(selected_indices)
        best_mis_indices = selected_indices[:]

    # Strategy 2: Bitset branch-and-bound with time limit
    if time.time() < deadline - 10:
        bb_start = time.time()
        bb_limit = deadline - 5
        bb_result = bitset_mis(adj_bitsets, n_b, bb_limit, bb_start)
        if len(bb_result) > best_mis_size:
            best_mis_size = len(bb_result)
            best_mis_indices = bb_result[:]

    # Strategy 3: Try a few random orderings with greedy
    rng = np.random.default_rng(42)
    for trial in range(20):
        if time.time() > deadline - 5:
            break
        order = rng.permutation(n_b)
        selected = set()
        for idx in order:
            if time.time() > deadline - 5:
                break
            # Check if idx is compatible with all selected
            compatible = True
            for sel in selected:
                if adjacency[idx, sel]:
                    compatible = False
                    break
            if compatible:
                selected.add(idx)
        if len(selected) > best_mis_size:
            best_mis_size = len(selected)
            best_mis_indices = list(selected)

    # Convert selected indices to vectors
    selected_b_vectors = type_b_vectors[best_mis_indices]

    # Combine with Type A
    all_vectors = np.vstack([type_a_vectors, selected_b_vectors])

    return all_vectors
