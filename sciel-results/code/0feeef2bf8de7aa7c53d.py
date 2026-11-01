"""Kissing number in dimension 11: D11 lattice shortest vectors (squared norm 4, even coordinate sum)."""
import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s

    # Generate all shortest vectors of the D11 lattice:
    # - Type A: exactly 4 nonzero entries, each ±1, sum of all coordinates is even
    #   (4 entries of ±1: sum = (#plus) - (#minus) = 2k - 4 for k plus signs; always even)
    #   So ALL such vectors have even coordinate sum.
    # - Type B: exactly 1 nonzero entry = ±2, sum = ±2 (even)
    #   So ALL such vectors have even coordinate sum.
    #
    # Total: C(11,4)*2^4 + 11*2 = 330*16 + 22 = 5280 + 22 = 5302
    # But wait: the problem says "nonzero integer vectors in R^11 whose minimum pairwise
    # distance is at least the maximum vector norm."
    #
    # For norm^2 = 4 vectors: max norm = 2. Min pairwise distance >= 2 means
    # ||v_i - v_j|| >= 2, i.e., ||v_i - v_j||^2 >= 4.
    # ||v_i - v_j||^2 = ||v_i||^2 + ||v_j||^2 - 2*v_i·v_j = 8 - 2*v_i·v_j >= 4
    # => v_i·v_j <= 2.
    #
    # For D11 lattice shortest vectors (norm^2=4), the dot product between any two
    # distinct vectors is at most 2. Let's verify:
    # - Two Type A vectors: they share at most 3 positions. If they share 3, the dot
    #   product is at most 3 (if all shared signs agree). Wait, that's > 2!
    #
    # Actually, let me reconsider. The D11 lattice is {x in Z^11 : sum(x_i) is even}.
    # The shortest vectors have norm^2 = 4. The kissing number of D11 is 2304.
    # Wait, that doesn't match. Let me recalculate.
    #
    # D_n has shortest vectors of norm^2 = 2: vectors with two ±1 entries.
    # For D_11: C(11,2)*4 = 55*4 = 220 vectors of norm^2 = 2.
    # The next shortest are norm^2 = 4: vectors with four ±1 entries (C(11,4)*16 = 5280)
    # and vectors with one ±2 entry (11*2 = 22). Total = 5302.
    #
    # But the kissing number is the max number of norm-1 vectors with mutual distance >= 2.
    # If we use norm^2 = 4 vectors, we need mutual distance^2 >= 4, i.e., dot <= 2.
    #
    # For two Type A vectors (4 nonzero ±1 entries):
    # They can share 0, 1, 2, 3, or 4 positions.
    # - Share 0: dot = 0 ✓
    # - Share 1: dot = ±1 ✓
    # - Share 2: dot = -2, 0, or 2 ✓ (all ≤ 2)
    # - Share 3: dot = -3, -1, 1, or 3 ✗ (dot=3 violates)
    # - Share 4: same vector or dot = 4, 2, 0, -2, -4
    #
    # So NOT all Type A vectors are mutually compatible! We need to find a subset.
    #
    # The actual kissing number of D11 is known to be 2304? No, that's not right either.
    # Let me just use the constructive approach: generate all 5302 candidates and find
    # the maximum independent set in the conflict graph (conflict if dot > 2).
    #
    # Actually, re-reading the task: "minimum pairwise distance is at least the maximum
    # vector norm." If all vectors have norm 2, then min distance >= 2.
    # This is exactly what we need.
    #
    # Let me just generate all candidates and do local search, but with a better
    # initial solution using the algebraic structure.

    # Generate Type A vectors (4 nonzero ±1 entries)
    v4_list = []
    for positions in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for pos, s in zip(positions, signs):
                v[pos] = s
            v4_list.append(v)
    n4 = len(v4_list)  # 5280

    # Generate Type B vectors (1 nonzero ±2 entry)
    v1_list = []
    for pos in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[pos] = 2 * sign
            v1_list.append(v)
    n1 = len(v1_list)  # 22

    n_total = n4 + n1
    all_vectors = np.array(v4_list + v1_list, dtype=np.int64)

    # Compute conflict matrix: conflict if dot > 2
    dots = all_vectors @ all_vectors.T
    conflicts = (dots > 2).astype(np.int32)
    np.fill_diagonal(conflicts, 0)

    # Convert to bitsets for fast operations
    conflict_bits = []
    for i in range(n_total):
        bits = 0
        row = conflicts[i]
        for j in range(n_total):
            if row[j]:
                bits |= (1 << j)
        conflict_bits.append(bits)

    def bit_count(x):
        return bin(x).count('1')

    # Algebraic initial solution: use sign symmetry
    # For each position set (4 positions), we can pick at most a limited number of sign patterns.
    # Key insight: for a fixed set of 4 positions, two sign patterns conflict iff they agree
    # on at least 3 positions (dot >= 3). So we can pick sign patterns that differ in at least
    # 2 positions. This is equivalent to a code of length 4, distance >= 2, over {+1,-1}.
    # The max size of such a code is 8 (take all patterns with even number of minus signs,
    # or equivalently, fix one sign). Actually, we can take all 16 patterns? No.
    # Two patterns of length 4 agree in k positions iff they differ in 4-k positions.
    # They conflict iff they agree in >= 3 positions, i.e., differ in <= 1 position.
    # So we need a code of length 4 with minimum Hamming distance >= 2.
    # The max size is 8 (e.g., all patterns with first sign = +1, or even-weight patterns).
    #
    # But different position sets also conflict with each other.
    # Let's just use a greedy approach with the algebraic hint.

    # Better: use a structured initial solution
    # For each of the C(11,4)=330 position sets, pick 8 sign patterns (even parity)
    # Then check compatibility and remove conflicts.
    # This gives at most 330*8 = 2640 Type A vectors + up to 22 Type B vectors.

    # Actually, let's just do a fast greedy + local search with multiple restarts.
    # The key improvement: better initial solutions using the algebraic structure.

    rng = np.random.default_rng(42)

    def greedy_min_degree():
        """Greedy: pick vertex with min degree among alive, remove it and its neighbors."""
        in_set = 0
        alive = (1 << n_total) - 1
        # Precompute degrees
        degrees = np.zeros(n_total, dtype=np.int32)
        for i in range(n_total):
            degrees[i] = bit_count(conflict_bits[i] & alive)

        while alive:
            # Find min degree alive vertex
            min_deg = n_total + 1
            best_v = -1
            mask = alive
            while mask:
                v = (mask & (-mask)).bit_length() - 1
                if degrees[v] < min_deg:
                    min_deg = degrees[v]
                    best_v = v
                mask &= mask - 1

            if best_v == -1:
                break

            in_set |= (1 << best_v)
            alive &= ~(1 << best_v)

            # Remove neighbors
            nbrs = conflict_bits[best_v]
            alive &= ~nbrs

            # Update degrees
            mask = alive
            while mask:
                i = (mask & (-mask)).bit_length() - 1
                degrees[i] -= bit_count(conflict_bits[i] & nbrs)
                mask &= mask - 1

        return in_set

    def local_search(init_bits, max_time=1.0):
        """Simulated annealing local search."""
        current = init_bits
        current_size = bit_count(current)
        best = current
        best_size = current_size
        ls_deadline = time.time() + max_time
        T = 0.5
        T_min = 0.01
        cooling = 0.999

        while time.time() < ls_deadline and T > T_min:
            improved = False

            # Add all compatible vertices
            not_in_set = ((1 << n_total) - 1) ^ current
            mask = not_in_set
            while mask:
                v = (mask & (-mask)).bit_length() - 1
                if (conflict_bits[v] & current) == 0:
                    current |= (1 << v)
                    current_size += 1
                    improved = True
                mask &= mask - 1

            if current_size > best_size:
                best_size = current_size
                best = current

            # Try swaps: remove one, add up to two
            if current_size > 0 and time.time() < ls_deadline - 0.05:
                # Collect set vertices
                set_verts = []
                mask = current
                while mask:
                    v = (mask & (-mask)).bit_length() - 1
                    set_verts.append(v)
                    mask &= mask - 1

                if set_verts:
                    # Pick a random vertex to remove
                    u = set_verts[rng.integers(len(set_verts))]
                    # Find vertices that conflict only with u (among set)
                    # and are not in set
                    not_in_set = ((1 << n_total) - 1) ^ current
                    candidates = []
                    mask = not_in_set
                    while mask:
                        v = (mask & (-mask)).bit_length() - 1
                        cc = conflict_bits[v] & current
                        if cc == (1 << u):
                            candidates.append(v)
                        mask &= mask - 1

                    if candidates:
                        v = candidates[rng.integers(len(candidates))]
                        new_set = (current & ~(1 << u)) | (1 << v)
                        if rng.random() < T:
                            current = new_set
                            improved = True

            if not improved:
                # Perturb: remove 2 random vertices
                set_verts = []
                mask = current
                while mask:
                    v = (mask & (-mask)).bit_length() - 1
                    set_verts.append(v)
                    mask &= mask - 1
                if len(set_verts) >= 2:
                    to_remove = rng.choice(len(set_verts), size=2, replace=False)
                    for idx in to_remove:
                        current &= ~(1 << set_verts[idx])
                        current_size -= 1

            T *= cooling

        # Final polish
        not_in_set = ((1 << n_total) - 1) ^ current
        mask = not_in_set
        while mask:
            v = (mask & (-mask)).bit_length() - 1
            if (conflict_bits[v] & current) == 0:
                current |= (1 << v)
                current_size += 1
            mask &= mask - 1
        if current_size > best_size:
            best_size = current_size
            best = current
        return best, best_size

    # Main search
    best_bits = 0
    best_size = 0

    # Initial greedy
    init_bits = greedy_min_degree()
    init_size = bit_count(init_bits)
    if init_size > best_size:
        best_size = init_size
        best_bits = init_bits

    # Local search from greedy
    remaining = deadline - 5 - time.time()
    if remaining > 0.5:
        ls_result, ls_size = local_search(init_bits, max_time=min(remaining * 0.4, 3.0))
        if ls_size > best_size:
            best_size = ls_size
            best_bits = ls_result

    # Multiple restarts
    n_restarts = 0
    while time.time() < deadline - 3 and n_restarts < 200:
        n_restarts += 1

        # Randomized greedy
        in_set = 0
        alive = (1 << n_total) - 1
        degrees = np.zeros(n_total, dtype=np.int32)
        for i in range(n_total):
            degrees[i] = bit_count(conflict_bits[i] & alive)

        while alive:
            # Find min degree
            min_deg = n_total + 1
            cands = []
            mask = alive
            while mask:
                v = (mask & (-mask)).bit_length() - 1
                if degrees[v] < min_deg:
                    min_deg = degrees[v]
                    cands = [v]
                elif degrees[v] == min_deg:
                    cands.append(v)
                mask &= mask - 1

            if not cands:
                break
            chosen = cands[rng.integers(len(cands))]
            in_set |= (1 << chosen)
            alive &= ~(1 << chosen)
            nbrs = conflict_bits[chosen]
            alive &= ~nbrs
            mask = alive
            while mask:
                i = (mask & (-mask)).bit_length() - 1
                degrees[i] -= bit_count(conflict_bits[i] & nbrs)
                mask &= mask - 1

        init_size = bit_count(in_set)
        if init_size > best_size:
            best_size = init_size
            best_bits = in_set

        remaining = deadline - 3 - time.time()
        if remaining < 0.3:
            break
        ls_time = min(remaining * 0.15, 1.5)
        if ls_time < 0.2:
            ls_time = 0.2
        ls_result, ls_size = local_search(in_set, max_time=ls_time)
        if ls_size > best_size:
            best_size = ls_size
            best_bits = ls_result

    # Convert to index array
    final_indices = []
    mask = best_bits
    while mask:
        v = (mask & (-mask)).bit_length() - 1
        final_indices.append(v)
        mask &= mask - 1

    if not final_indices:
        return np.array([], dtype=np.int64)

    return all_vectors[final_indices]
