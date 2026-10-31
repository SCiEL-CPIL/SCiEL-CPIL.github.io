import numpy as np
import itertools
import time

D = 11

def construct():
    start_time = time.time()
    deadline = start_time + 145

    # Generate all norm-4 integer vectors
    pool = []
    for i in range(D):
        for sign in [1, -1]:
            v = [0] * D
            v[i] = 2 * sign
            pool.append(v)
    for comb in itertools.combinations(range(D), 4):
        for signs in itertools.product([1, -1], repeat=4):
            v = [0] * D
            for idx, sign in zip(comb, signs):
                v[idx] = sign
            pool.append(v)

    pool = np.array(pool, dtype=np.int64)
    n = len(pool)

    dots = pool @ pool.T
    conflict = (dots > 2)
    np.fill_diagonal(conflict, False)

    # Type 1 (0-21) all compatible with each other and with type 2
    # Solve MIS on type-2 subgraph (indices 22 to 5301)
    n2 = n - 22
    t2_conf = conflict[22:, 22:]

    # Convert to uint64 bitsets for fast operations
    # n2 = 5280, need ceil(5280/64) = 83 uint64 words per row
    nwords = (n2 + 63) // 64
    conf_bits = np.zeros((n2, nwords), dtype=np.uint64)
    for i in range(n2):
        row = t2_conf[i]
        for j in np.where(row)[0]:
            conf_bits[i, j // 64] |= (1 << (j % 64))

    # Greedy coloring upper bound using bitsets
    def color_bound(remaining_words):
        """Upper bound via greedy coloring. remaining_words: (nwords,) uint64 array"""
        if not np.any(remaining_words):
            return 0

        colors = 0
        rem = remaining_words.copy()

        while np.any(rem):
            # Find vertex with min degree in rem
            # For speed, just pick first available vertex
            # Find first set bit
            found = False
            start_w = 0
            for w in range(nwords):
                if rem[w]:
                    # Find lowest set bit in this word
                    word = rem[w]
                    # Lowest set bit position
                    lb = (word & (-word)).bit_length() - 1
                    v = w * 64 + lb
                    if v < n2:
                        found = True
                        break
            if not found:
                break

            # Remove v and its neighbors from rem
            rem &= ~conf_bits[v]
            # Also clear v's own bit
            rem[v // 64] &= ~(1 << (v % 64))

            colors += 1

        return colors

    # Branch and bound with time limit
    best_size = 0
    best_mask = np.zeros(nwords, dtype=np.uint64)

    # Initial greedy solution
    rem = np.ones(nwords, dtype=np.uint64)
    # Clear bits beyond n2
    if n2 % 64 != 0:
        rem[-1] &= (1 << (n2 % 64)) - 1

    greedy_mask = np.zeros(nwords, dtype=np.uint64)
    greedy_size = 0
    rem_g = rem.copy()
    while np.any(rem_g):
        if time.time() > deadline:
            break
        # Pick vertex with min degree
        best_v = -1
        best_deg = n2 + 1
        for w in range(nwords):
            word = rem_g[w]
            while word:
                lb = (word & (-word)).bit_length() - 1
                v = w * 64 + lb
                if v < n2:
                    # Count degree
                    deg = 0
                    for ww in range(nwords):
                        deg += (rem_g[ww] & conf_bits[v, ww]).bit_count()
                    if deg < best_deg:
                        best_deg = deg
                        best_v = v
                    word &= word - 1
        if best_v < 0:
            break
        greedy_mask[best_v // 64] |= (1 << (best_v % 64))
        greedy_size += 1
        rem_g &= ~conf_bits[best_v]
        rem_g[best_v // 64] &= ~(1 << (best_v % 64))

    best_size = greedy_size
    best_mask = greedy_mask.copy()

    # B&B
    def b_and_b(rem, curr_size, curr_mask):
        nonlocal best_size, best_mask

        if time.time() > deadline:
            return

        if not np.any(rem):
            if curr_size > best_size:
                best_size = curr_size
                best_mask = curr_mask.copy()
            return

        ub = color_bound(rem)
        if curr_size + ub <= best_size:
            return

        # Pick branching vertex: max degree in rem
        best_v = -1
        best_deg = -1
        for w in range(nwords):
            word = rem[w]
            while word:
                lb = (word & (-word)).bit_length() - 1
                v = w * 64 + lb
                if v < n2:
                    deg = 0
                    for ww in range(nwords):
                        deg += (rem[ww] & conf_bits[v, ww]).bit_count()
                    if deg > best_deg:
                        best_deg = deg
                        best_v = v
                word &= word - 1

        if best_v < 0:
            return

        # Branch 1: include best_v
        new_rem = rem & ~conf_bits[best_v]
        new_rem[best_v // 64] &= ~(1 << (best_v % 64))
        new_mask = curr_mask.copy()
        new_mask[best_v // 64] |= (1 << (best_v % 64))
        b_and_b(new_rem, curr_size + 1, new_mask)

        # Branch 2: exclude best_v
        new_rem = rem.copy()
        new_rem[best_v // 64] &= ~(1 << (best_v % 64))
        b_and_b(new_rem, curr_size, curr_mask)

    initial_rem = np.ones(nwords, dtype=np.uint64)
    if n2 % 64 != 0:
        initial_rem[-1] &= (1 << (n2 % 64)) - 1

    b_and_b(initial_rem, 0, np.zeros(nwords, dtype=np.uint64))

    # Extract selected vertices
    selected = []
    for w in range(nwords):
        word = best_mask[w]
        while word:
            lb = (word & (-word)).bit_length() - 1
            v = w * 64 + lb
            if v < n2:
                selected.append(v + 22)
            word &= word - 1

    all_indices = list(range(22)) + selected
    result = pool[all_indices]
    return result
