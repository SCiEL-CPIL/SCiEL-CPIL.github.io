import time
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    total_words = q ** n

    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # Precompute neighbors (Hamming distance <= radius)
    neigh = [[] for _ in range(total_words)]
    ball_masks = [0] * total_words

    for idx in range(total_words):
        w = all_words[idx]
        neigh[idx].append(idx)
        ball_masks[idx] |= (1 << idx)

        w_list = list(w)
        for pos in range(n):
            orig_val = w_list[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                w_list[pos] = new_val
                n_idx = word_to_idx[tuple(w_list)]
                neigh[idx].append(n_idx)
                ball_masks[idx] |= (1 << n_idx)
            w_list[pos] = orig_val

    # Precompute distance-2 adjacency for structural initialization
    dist2_adj = [set() for _ in range(total_words)]
    for idx in range(total_words):
        w = all_words[idx]
        w_list = list(w)
        for i in range(n):
            for j in range(i + 1, n):
                orig_i = w_list[i]
                orig_j = w_list[j]
                for new_i in range(q):
                    if new_i == orig_i:
                        continue
                    for new_j in range(q):
                        if new_j == orig_j:
                            continue
                        w_list[i] = new_i
                        w_list[j] = new_j
                        n_idx = word_to_idx[tuple(w_list)]
                        if n_idx != idx:
                            dist2_adj[idx].add(n_idx)
                            dist2_adj[n_idx].add(idx)
                    w_list[i] = orig_i
                    w_list[j] = orig_j

    def build_counts(code_list):
        counts = [0] * total_words
        for c in code_list:
            m = ball_masks[c]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                m ^= lsb
        return counts

    def prune_redundant(code_list, counts):
        changed = True
        while changed:
            changed = False
            for i in range(len(code_list) - 1, -1, -1):
                c_idx = code_list[i]
                redundant = True
                m = ball_masks[c_idx]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    if counts[idx] <= 1:
                        redundant = False
                        break
                    m ^= lsb
                if redundant:
                    code_list.pop(i)
                    m = ball_masks[c_idx]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        m ^= lsb
                    changed = True
        return code_list

    def greedy_cover_from_seed(seed_words, rng):
        code_set = set(seed_words)
        code_list = list(seed_words)

        counts = [0] * total_words
        for c in code_list:
            for u in neigh[c]:
                counts[u] += 1

        uncovered_mask = 0
        for i in range(total_words):
            if counts[i] == 0:
                uncovered_mask |= (1 << i)

        while uncovered_mask:
            if time.time() > deadline - 5.0:
                break
            best_idx = -1
            best_gain = 0

            for i in range(total_words):
                if i in code_set:
                    continue
                gain = (uncovered_mask & ball_masks[i]).bit_count()
                if gain > best_gain:
                    best_gain = gain
                    best_idx = i

            if best_idx == -1 or best_gain == 0:
                break

            code_set.add(best_idx)
            code_list.append(best_idx)

            m = ball_masks[best_idx]
            while m:
                lsb = m & -m
                idx = lsb.bit_length() - 1
                counts[idx] += 1
                if counts[idx] == 1:
                    uncovered_mask ^= (1 << idx)
                m ^= lsb

        return code_list, counts

    def get_greedy_independent_set(rng):
        candidates = list(range(total_words))
        rng.shuffle(candidates)

        in_set = [False] * total_words
        code_set = set()

        for v in candidates:
            if in_set[v]:
                continue
            is_conflict = False
            for u in dist2_adj[v]:
                if in_set[u]:
                    is_conflict = True
                    break
            if not is_conflict:
                code_set.add(v)
                in_set[v] = True

        return list(code_set)

    def simulated_annealing(code_list, counts, rng, time_budget):
        """
        SA with cost = len(code) + penalty for uncovered words.
        Moves: (1) remove a codeword, (2) add a codeword, (3) swap one codeword,
        (4) 2-opt: swap two codewords simultaneously.
        """
        if not code_list:
            return code_list, counts

        sa_end = time.time() + time_budget
        T = 2.0  # initial temperature
        T_min = 0.01
        cooling = 0.9995

        best_code = code_list[:]
        best_count = counts[:]
        best_size = len(code_list)

        in_code = [False] * total_words
        for c in code_list:
            in_code[c] = True

        while time.time() < sa_end:
            if T < T_min:
                break

            # Pick a random move type
            move_type = rng.random()
            old_size = len(code_list)
            old_uncovered = sum(1 for i in range(total_words) if counts[i] == 0)
            old_cost = old_size + 10.0 * old_uncovered

            new_cost = old_cost
            accepted = False

            if move_type < 0.15 and len(code_list) > 1:
                # Remove a random codeword
                i = rng.randint(0, len(code_list) - 1)
                c_old = code_list[i]
                m = ball_masks[c_old]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    counts[idx] -= 1
                    m ^= lsb
                code_list.pop(i)
                in_code[c_old] = False

                new_uncovered = sum(1 for i in range(total_words) if counts[i] == 0)
                new_cost = len(code_list) + 10.0 * new_uncovered
                delta = new_cost - old_cost

                if delta <= 0 or rng.random() < min(1.0, (-delta) / T):
                    accepted = True
                else:
                    # Revert
                    code_list.insert(i, c_old)
                    in_code[c_old] = True
                    m = ball_masks[c_old]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        counts[idx] += 1
                        m ^= lsb

            elif move_type < 0.30:
                # Add a random codeword
                c_new = rng.randint(0, total_words - 1)
                if in_code[c_new]:
                    T *= cooling
                    continue
                code_list.append(c_new)
                in_code[c_new] = True
                m = ball_masks[c_new]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    counts[idx] += 1
                    m ^= lsb

                new_uncovered = sum(1 for i in range(total_words) if counts[i] == 0)
                new_cost = len(code_list) + 10.0 * new_uncovered
                delta = new_cost - old_cost

                if delta <= 0 or rng.random() < min(1.0, (-delta) / T):
                    accepted = True
                else:
                    # Revert
                    code_list.pop()
                    in_code[c_new] = False
                    m = ball_masks[c_new]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        m ^= lsb

            elif move_type < 0.70:
                # Swap one codeword
                if len(code_list) == 0:
                    T *= cooling
                    continue
                i = rng.randint(0, len(code_list) - 1)
                c_old = code_list[i]
                c_new = rng.randint(0, total_words - 1)
                if c_new == c_old or in_code[c_new]:
                    T *= cooling
                    continue

                # Remove c_old
                m = ball_masks[c_old]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    counts[idx] -= 1
                    m ^= lsb
                in_code[c_old] = False

                # Add c_new
                m = ball_masks[c_new]
                while m:
                    lsb = m & -m
                    idx = lsb.bit_length() - 1
                    counts[idx] += 1
                    m ^= lsb
                in_code[c_new] = True
                code_list[i] = c_new

                new_uncovered = sum(1 for i in range(total_words) if counts[i] == 0)
                new_cost = len(code_list) + 10.0 * new_uncovered
                delta = new_cost - old_cost

                if delta <= 0 or rng.random() < min(1.0, (-delta) / T):
                    accepted = True
                else:
                    # Revert
                    m = ball_masks[c_new]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        m ^= lsb
                    in_code[c_new] = False

                    m = ball_masks[c_old]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        counts[idx] += 1
                        m ^= lsb
                    in_code[c_old] = True
                    code_list[i] = c_old

            else:
                # 2-opt: swap two codewords
                if len(code_list) < 2:
                    T *= cooling
                    continue
                i1 = rng.randint(0, len(code_list) - 1)
                i2 = rng.randint(0, len(code_list) - 1)
                if i1 == i2:
                    T *= cooling
                    continue
                c_old1 = code_list[i1]
                c_old2 = code_list[i2]
                c_new1 = rng.randint(0, total_words - 1)
                c_new2 = rng.randint(0, total_words - 1)

                # Ensure new words are distinct and not in code
                if c_new1 == c_new2 or in_code[c_new1] or in_code[c_new2]:
                    T *= cooling
                    continue
                if c_new1 == c_old1 or c_new1 == c_old2 or c_new2 == c_old1 or c_new2 == c_old2:
                    T *= cooling
                    continue

                # Remove both old
                for c in (c_old1, c_old2):
                    m = ball_masks[c]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        counts[idx] -= 1
                        m ^= lsb
                    in_code[c] = False

                # Add both new
                for c in (c_new1, c_new2):
                    m = ball_masks[c]
                    while m:
                        lsb = m & -m
                        idx = lsb.bit_length() - 1
                        counts[idx] += 1
                        m ^= lsb
                    in_code[c] = True
                code_list[i1] = c_new1
                code_list[i2] = c_new2

                new_uncovered = sum(1 for i in range(total_words) if counts[i] == 0)
                new_cost = len(code_list) + 10.0 * new_uncovered
                delta = new_cost - old_cost

                if delta <= 0 or rng.random() < min(1.0, (-delta) / T):
                    accepted = True
                else:
                    # Revert
                    for c in (c_new1, c_new2):
                        m = ball_masks[c]
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            counts[idx] -= 1
                            m ^= lsb
                        in_code[c] = False

                    for c in (c_old1, c_old2):
                        m = ball_masks[c]
                        while m:
                            lsb = m & -m
                            idx = lsb.bit_length() - 1
                            counts[idx] += 1
                            m ^= lsb
                        in_code[c] = True
                    code_list[i1] = c_old1
                    code_list[i2] = c_old2

            # Track best
            if accepted and len(code_list) < best_size and all(counts[i] > 0 for i in range(total_words)):
                best_size = len(code_list)
                best_code = code_list[:]
                best_count = counts[:]

            T *= cooling

        # Restore best
        for c in code_list:
            in_code[c] = False
        for c in best_code:
            in_code[c] = True
        code_list[:] = best_code
        counts[:] = best_count

        return code_list, counts

    best_code = []
    best_size = float('inf')

    num_starts = 80

    for seed in range(num_starts):
        if time.time() > deadline - 10.0:
            break

        rng = random.Random(seed)

        if seed % 2 == 0:
            seed_words = get_greedy_independent_set(rng)
            code, counts = greedy_cover_from_seed(seed_words, rng)
        else:
            uncovered = (1 << total_words) - 1
            code = []
            while uncovered:
                best_idx = -1
                best_gain = -1
                for i in range(total_words):
                    gain = (uncovered & ball_masks[i]).bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i
                if best_idx == -1:
                    break
                code.append(best_idx)
                uncovered &= ~ball_masks[best_idx]
            counts = build_counts(code)

        # Prune first
        code, counts = prune_redundant(code, counts)

        # Simulated annealing
        remaining = deadline - 15.0 - time.time()
        if remaining > 1.0:
            code, counts = simulated_annealing(code, counts, rng, remaining)

        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]

        if best_size <= 73:
            break

    if best_code:
        counts = build_counts(best_code)
        best_code = prune_redundant(best_code, counts)

    result = [list(all_words[c]) for c in best_code]
    return result
