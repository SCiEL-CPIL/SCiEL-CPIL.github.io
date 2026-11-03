"""Ternary covering code K_3(8,1) via structured init + vectorized local search."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    total_words = q ** n
    if total_words == 1:
        return [[0] * n]

    # Precompute neighbors table: neighbors[w] is a list/array of indices at Hamming dist <= radius
    # For radius=1, this is w itself + all words differing in exactly one position
    n_neighbors = 1 + n * (q - 1)
    neighbors = np.zeros((total_words, n_neighbors), dtype=np.int32)

    # Precompute digit extraction for speed
    q_powers = np.array([q ** i for i in range(n)], dtype=np.int64)
    # q_powers[i] = q^i, so digit at position i (0-indexed from right) is (w // q^i) % q
    # But our index convention: idx = sum(digit_i * q^i) where digit_0 is least significant
    # Let's define: word w has digits d_0, d_1, ..., d_{n-1} where w = sum d_i * q^i
    # Position i corresponds to q^i

    for w in range(total_words):
        neighbors[w, 0] = w
        idx = 1
        for pos in range(n):
            digit = (w // (q ** pos)) % q
            for v in range(q):
                if v == digit:
                    continue
                diff = (v - digit) * (q ** pos)
                neighbors[w, idx] = w + diff
                idx += 1

    # Precompute neighbor sets as Python sets for fast membership
    neighbor_sets = [set(neighbors[w].tolist()) for w in range(total_words)]

    # Greedy initialization with gain tracking
    def greedy_init(seed_offset=0):
        rng = random.Random(seed_offset)
        covered = np.zeros(total_words, dtype=bool)
        gain = np.full(total_words, n_neighbors, dtype=np.int32)
        codewords = []
        remaining = total_words

        while remaining > 0:
            if time.time() > deadline - 20.0:
                break
            # Pick the word with highest gain (covers most uncovered words)
            best_word = int(np.argmax(gain))
            best_gain = gain[best_word]

            if best_gain == 0:
                # Fallback: pick first uncovered
                uncovered = np.where(~covered)[0]
                if len(uncovered) == 0:
                    break
                best_word = int(uncovered[0])

            codewords.append(best_word)

            for v in neighbors[best_word]:
                if not covered[v]:
                    covered[v] = True
                    remaining -= 1
                    for w in neighbors[v]:
                        gain[w] -= 1

        return codewords

    # Local search: remove redundant, swap, perturb
    def local_search(cw, time_limit):
        cw = cw[:]
        best_cw = cw[:]
        best_size = len(cw)

        count = np.zeros(total_words, dtype=np.int32)
        in_code = np.zeros(total_words, dtype=bool)

        for ci in cw:
            in_code[ci] = True
            for u in neighbors[ci]:
                count[u] += 1

        while time.time() < time_limit:
            improved = False

            # Pass 1: Remove redundant codewords
            indices = list(range(len(cw)))
            random.shuffle(indices)

            for i in indices:
                if i >= len(cw):
                    continue
                ci = cw[i]
                # Check if all neighbors have count > 1
                if np.min(count[neighbors[ci]]) > 1:
                    for u in neighbors[ci]:
                        count[u] -= 1
                    in_code[ci] = False
                    cw.pop(i)
                    improved = True
                    if len(cw) < best_size:
                        best_size = len(cw)
                        best_cw = cw[:]

            # Pass 2: Swaps
            if not improved:
                indices = list(range(len(cw)))
                random.shuffle(indices)

                for i in indices:
                    if i >= len(cw):
                        continue
                    ci = cw[i]

                    critical_mask = count[neighbors[ci]] == 1
                    critical = neighbors[ci][critical_mask]

                    if len(critical) == 0:
                        # Can just remove
                        for u in neighbors[ci]:
                            count[u] -= 1
                        in_code[ci] = False
                        cw.pop(i)
                        improved = True
                        if len(cw) < best_size:
                            best_size = len(cw)
                            best_cw = cw[:]
                        break

                    if len(critical) > 0:
                        first_crit = critical[0]

                        if len(critical) == 1:
                            # Find a candidate that covers first_crit and is not already in code
                            valid_cands = []
                            for cand in neighbor_sets[first_crit]:
                                if cand == ci or in_code[cand]:
                                    continue
                                valid_cands.append(cand)

                            if valid_cands:
                                cand = random.choice(valid_cands)
                                for u in neighbors[ci]:
                                    count[u] -= 1
                                for u in neighbors[cand]:
                                    count[u] += 1
                                in_code[ci] = False
                                in_code[cand] = True
                                cw[i] = cand
                                improved = True
                                if len(cw) < best_size:
                                    best_size = len(cw)
                                    best_cw = cw[:]
                                break
                        else:
                            # Multiple critical: need candidate in intersection of all critical neighbors' neighbor sets
                            candidate_set = neighbor_sets[first_crit].copy()
                            for u in critical[1:]:
                                candidate_set &= neighbor_sets[u]
                                if not candidate_set:
                                    break

                            if candidate_set:
                                valid_cands = [c for c in candidate_set if c != ci and not in_code[c]]
                                if valid_cands:
                                    cand = random.choice(valid_cands)
                                    for u in neighbors[ci]:
                                        count[u] -= 1
                                    for u in neighbors[cand]:
                                        count[u] += 1
                                    in_code[ci] = False
                                    in_code[cand] = True
                                    cw[i] = cand
                                    improved = True
                                    if len(cw) < best_size:
                                        best_size = len(cw)
                                        best_cw = cw[:]
                                    break

            if not improved:
                # Perturbation: randomly swap a few codewords to escape local minimum
                if len(cw) > 2 and random.random() < 0.5:
                    num_swaps = random.randint(1, 3)
                    for _ in range(num_swaps):
                        idx1, idx2 = random.sample(range(len(cw)), 2)
                        c1, c2 = cw[idx1], cw[idx2]

                        # Check if swapping maintains coverage
                        temp_count = count.copy()
                        for u in neighbors[c1]:
                            temp_count[u] -= 1
                        for u in neighbors[c2]:
                            temp_count[u] += 1

                        if np.min(temp_count) >= 1:
                            for u in neighbors[c1]:
                                count[u] -= 1
                            for u in neighbors[c2]:
                                count[u] += 1
                            in_code[c1] = False
                            in_code[c2] = True
                            cw[idx1] = c2
                            improved = True
                            if len(cw) < best_size:
                                best_size = len(cw)
                                best_cw = cw[:]

        return best_cw

    # Main: try multiple random seeds for greedy init, then local search
    best_code = None
    best_size = float('inf')

    # Try a few different greedy inits with different randomization
    for seed in range(5):
        if time.time() > deadline - 30:
            break
        cw = greedy_init(seed)
        if len(cw) >= best_size:
            continue
        # Run local search
        remaining_time = deadline - time.time() - 5
        if remaining_time < 5:
            break
        cw = local_search(cw, deadline - 10)
        if len(cw) < best_size:
            best_size = len(cw)
            best_code = cw

    if best_code is None:
        # Fallback: just do one greedy
        best_code = greedy_init(0)

    # Convert indices to tuples
    def idx_to_tuple(idx):
        t = []
        for _ in range(n):
            t.append(idx % q)
            idx //= q
        return t

    result = [idx_to_tuple(c) for c in best_code]
    return result

# EVOLVE-BLOCK-END
