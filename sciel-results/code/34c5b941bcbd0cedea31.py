import time
import random
import itertools
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s limit

    if q != 3 or n != 6 or radius != 1:
        return _greedy_set_cover(q, n, radius, deadline)

    space_size = q ** n  # 729

    # Precompute index <-> word conversions
    def word_to_idx(w):
        idx = 0
        for val in w:
            idx = idx * q + val
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for i in range(n - 1, -1, -1):
            w[i] = idx % q
            idx //= q
        return w

    # Precompute neighbors (distance <= radius) for every word
    neighbors = []
    for i in range(space_size):
        w = idx_to_word(i)
        nbs = [i]  # include self
        for pos in range(n):
            orig = w[pos]
            for new_val in range(q):
                if new_val != orig:
                    w[pos] = new_val
                    nbs.append(word_to_idx(w))
            w[pos] = orig
        neighbors.append(nbs)

    # Build neighbor sets for fast membership
    neighbor_sets = [set(nb) for nb in neighbors]

    # Build numpy array of neighbor indices
    neighbors_np = [np.array(nb, dtype=np.int32) for nb in neighbors]

    # Build adjacency matrix for vectorized operations
    # adj[i, j] = 1 if word j is within distance 1 of word i
    adj = np.zeros((space_size, space_size), dtype=np.int32)
    for i in range(space_size):
        for j in neighbors[i]:
            adj[i, j] = 1

    # Tetracode direct-sum initial construction: 9 * 3^(6-4) = 81 words
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    tails = list(itertools.product(range(q), repeat=n - 4))
    initial_codewords = [h + list(tail) for h in tetracode for tail in tails]

    def build_counts(cw_indices):
        """Build coverage count array from a list of codeword indices."""
        counts = np.zeros(space_size, dtype=np.int32)
        for ci in cw_indices:
            counts[neighbors[ci]] += 1
        return counts

    def is_valid(counts):
        """Check if all words are covered."""
        return np.all(counts >= 1)

    def greedy_remove(cw_indices, counts, deadline):
        """Iteratively remove redundant codewords."""
        current = list(cw_indices)
        cnt = counts.copy()
        current_set = set(current)

        while True:
            if time.time() > deadline:
                break
            removed = False
            for i in range(len(current) - 1, -1, -1):
                if time.time() > deadline:
                    break
                ci = current[i]
                can_remove = True
                for nb in neighbors[ci]:
                    if cnt[nb] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for nb in neighbors[ci]:
                        cnt[nb] -= 1
                    current.pop(i)
                    current_set.discard(ci)
                    removed = True
            if not removed:
                break
        return current, cnt

    def local_search_swap(cw_indices, counts, deadline, max_swaps=50000):
        """Local search: try swaps to enable further deletions."""
        current = list(cw_indices)
        cnt = counts.copy()
        current_set = set(current)
        swaps_made = 0

        while swaps_made < max_swaps and time.time() < deadline:
            progress = False

            # Phase 1: try simple removals
            for i in range(len(current) - 1, -1, -1):
                if time.time() > deadline:
                    return current, cnt
                ci = current[i]
                can_remove = True
                for nb in neighbors[ci]:
                    if cnt[nb] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for nb in neighbors[ci]:
                        cnt[nb] -= 1
                    current.pop(i)
                    current_set.discard(ci)
                    progress = True
            if progress:
                continue

            # Phase 2: try swaps
            for i in range(len(current)):
                if time.time() > deadline:
                    return current, cnt
                ci = current[i]
                essential = [nb for nb in neighbors[ci] if cnt[nb] == 1]
                if not essential:
                    for nb in neighbors[ci]:
                        cnt[nb] -= 1
                    current.pop(i)
                    current_set.discard(ci)
                    progress = True
                    break

                tried = 0
                max_tries = 100
                while tried < max_tries and time.time() < deadline:
                    tried += 1
                    cand = random.randint(0, space_size - 1)
                    if cand in current_set:
                        continue
                    cand_nbs = neighbors[cand]
                    covers = all(e in neighbor_sets[cand] for e in essential)
                    if not covers:
                        continue
                    valid = True
                    for nb in neighbors[ci]:
                        if cnt[nb] == 1 and nb not in neighbor_sets[cand]:
                            valid = False
                            break
                    if not valid:
                        continue
                    for nb in neighbors[ci]:
                        cnt[nb] -= 1
                    for nb in cand_nbs:
                        cnt[nb] += 1
                    current[i] = cand
                    current_set.discard(ci)
                    current_set.add(cand)
                    progress = True
                    swaps_made += 1
                    break
                if progress:
                    break

            if not progress:
                break

        return current, cnt

    def two_swap_search(cw_indices, counts, deadline, max_iters=20000):
        """Try removing two codewords at a time if valid."""
        current = list(cw_indices)
        cnt = counts.copy()
        current_set = set(current)
        iters = 0

        while iters < max_iters and time.time() < deadline:
            iters += 1
            progress = False

            # First try single removals
            for i in range(len(current) - 1, -1, -1):
                if time.time() > deadline:
                    return current, cnt
                ci = current[i]
                can_remove = True
                for nb in neighbors[ci]:
                    if cnt[nb] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for nb in neighbors[ci]:
                        cnt[nb] -= 1
                    current.pop(i)
                    current_set.discard(ci)
                    progress = True
            if progress:
                continue

            # Try two-removal
            found_two = False
            for i in range(len(current)):
                if time.time() > deadline:
                    return current, cnt
                ci = current[i]
                for j in range(i + 1, len(current)):
                    if time.time() > deadline:
                        return current, cnt
                    cj = current[j]
                    valid = True
                    for nb in neighbors[ci]:
                        if cnt[nb] == 1 and nb not in neighbor_sets[cj]:
                            valid = False
                            break
                    if valid:
                        for nb in neighbors[cj]:
                            if cnt[nb] == 1 and nb not in neighbor_sets[ci]:
                                valid = False
                                break
                    if valid:
                        for nb in neighbors[ci]:
                            cnt[nb] -= 1
                        for nb in neighbors[cj]:
                            cnt[nb] -= 1
                        current.pop(j)
                        current.pop(i)
                        current_set.discard(ci)
                        current_set.discard(cj)
                        found_two = True
                        break
                if found_two:
                    break
            if found_two:
                continue

            if not progress and not found_two:
                break

        return current, cnt

    def try_randomized_removal_order(cw_indices, counts, deadline, seed):
        """Try removal in a randomized order, then swap local search."""
        rng = random.Random(seed)
        current = list(cw_indices)
        cnt = counts.copy()

        for _ in range(20):
            if time.time() > deadline:
                break
            rng.shuffle(current)
            removed = False
            for ci in current[:]:
                if time.time() > deadline:
                    break
                can_remove = True
                for nb in neighbors[ci]:
                    if cnt[nb] == 1:
                        can_remove = False
                        break
                if can_remove:
                    for nb in neighbors[ci]:
                        cnt[nb] -= 1
                    current.remove(ci)
                    removed = True
            if not removed:
                break

        current, cnt = local_search_swap(current, cnt, deadline, max_swaps=10000)
        return current, cnt

    def simulated_annealing(initial_indices, deadline, sa_time_budget=25.0):
        """
        Simulated annealing on a binary inclusion vector.
        """
        sa_deadline = start_time + sa_time_budget
        rng = random.Random(42)

        x = np.zeros(space_size, dtype=np.int8)
        for ci in initial_indices:
            x[ci] = 1

        coverage = np.zeros(space_size, dtype=np.int32)
        for ci in initial_indices:
            coverage[neighbors_np[ci]] += 1

        uncovered_count = int(np.sum(coverage == 0))
        num_codewords = int(np.sum(x))

        penalty = 100
        current_obj = num_codewords + penalty * uncovered_count

        T_start = 15.0
        T_end = 0.005

        best_x = x.copy()
        best_obj = current_obj
        best_uncovered = uncovered_count
        best_num_cw = num_codewords

        best_valid_x = None
        best_valid_num_cw = num_codewords if uncovered_count == 0 else float('inf')

        iteration = 0
        max_iterations = 1500000

        while time.time() < sa_deadline and iteration < max_iterations:
            progress = iteration / max_iterations
            T = T_start * (T_end / T_start) ** progress

            pos = rng.randint(0, space_size - 1)

            if x[pos] == 1:
                nbs = neighbors_np[pos]
                newly_uncovered = int(np.sum(coverage[nbs] == 1))
                delta_obj = -1 + penalty * newly_uncovered

                if delta_obj <= 0 or (T > 0 and rng.random() < np.exp(-delta_obj / T)):
                    x[pos] = 0
                    coverage[nbs] -= 1
                    num_codewords -= 1
                    uncovered_count += newly_uncovered
                    current_obj += delta_obj

                    if current_obj < best_obj:
                        best_obj = current_obj
                        best_x = x.copy()
                        best_uncovered = uncovered_count
                        best_num_cw = num_codewords

                    if uncovered_count == 0 and num_codewords < best_valid_num_cw:
                        best_valid_num_cw = num_codewords
                        best_valid_x = x.copy()

            else:
                nbs = neighbors_np[pos]
                newly_covered = int(np.sum(coverage[nbs] == 0))
                delta_obj = 1 - penalty * newly_covered

                if delta_obj <= 0 or (T > 0 and rng.random() < np.exp(-delta_obj / T)):
                    x[pos] = 1
                    coverage[nbs] += 1
                    num_codewords += 1
                    uncovered_count -= newly_covered
                    current_obj += delta_obj

                    if current_obj < best_obj:
                        best_obj = current_obj
                        best_x = x.copy()
                        best_uncovered = uncovered_count
                        best_num_cw = num_codewords

                    if uncovered_count == 0 and num_codewords < best_valid_num_cw:
                        best_valid_num_cw = num_codewords
                        best_valid_x = x.copy()

            iteration += 1

        if best_valid_x is not None:
            result_indices = [i for i in range(space_size) if best_valid_x[i] == 1]
        else:
            result_indices = [i for i in range(space_size) if best_x[i] == 1]

        return result_indices

    # Main search loop
    best_size = float('inf')
    best_code = None

    # Initial: deterministic greedy remove + swap
    cw_indices = [word_to_idx(w) for w in initial_codewords]
    counts = build_counts(cw_indices)
    code, counts = greedy_remove(cw_indices, counts, deadline - 30)
    code, counts = local_search_swap(code, counts, deadline - 25, max_swaps=20000)
    if len(code) < best_size:
        best_size = len(code)
        best_code = code.copy()

    # Two-swap search to escape local minima
    code2, counts2 = two_swap_search(best_code, build_counts(best_code), deadline - 20, max_iters=15000)
    if len(code2) < best_size:
        best_size = len(code2)
        best_code = code2.copy()

    # Simulated annealing phase
    sa_budget = 30.0
    if time.time() < deadline - sa_budget - 10:
        sa_indices = simulated_annealing(cw_indices, deadline, sa_time_budget=sa_budget)

        sa_counts = build_counts(sa_indices)
        sa_valid = np.all(sa_counts >= 1)

        if sa_valid and len(sa_indices) < best_size:
            best_size = len(sa_indices)
            best_code = sa_indices.copy()
            polished, _ = greedy_remove(best_code, sa_counts, deadline - 5)
            polished, _ = local_search_swap(polished, build_counts(polished), deadline - 2, max_swaps=10000)
            if len(polished) < best_size:
                best_size = len(polished)
                best_code = polished.copy()

    # Multiple randomized restarts with remaining time
    remaining_deadline = deadline - 5
    restarts = 0
    while time.time() < remaining_deadline and best_size > 73:
        restarts += 1
        seed = restarts * 12345 + 678
        code_r, counts_r = try_randomized_removal_order(cw_indices, counts, remaining_deadline, seed)
        if len(code_r) < best_size:
            best_size = len(code_r)
            best_code = code_r.copy()

    # Final polish
    if best_code is not None and time.time() < deadline - 2:
        final_counts = build_counts(best_code)
        final_code, final_counts = greedy_remove(best_code, final_counts, deadline - 1)
        final_code, final_counts = local_search_swap(final_code, final_counts, deadline, max_swaps=10000)
        if len(final_code) < best_size:
            best_code = final_code
            best_size = len(final_code)
        final_code2, _ = two_swap_search(best_code, build_counts(best_code), deadline, max_iters=10000)
        if len(final_code2) < best_size:
            best_code = final_code2
            best_size = len(final_code2)

    result = [idx_to_word(ci) for ci in best_code]
    return result


def _greedy_set_cover(q, n, radius, deadline):
    """Greedy set cover for general parameters."""
    all_words = list(itertools.product(range(q), repeat=n))
    total = q ** n

    def hamming_dist(a, b):
        return sum(1 for x, y in zip(a, b) if x != y)

    coverage = {}
    for i, w in enumerate(all_words):
        covered = set()
        for j, v in enumerate(all_words):
            if hamming_dist(w, v) <= radius:
                covered.add(j)
        coverage[i] = covered

    uncovered = set(range(total))
    code = []

    while uncovered and time.time() < deadline - 5:
        best_i = -1
        best_count = 0
        for i in range(total):
            count = len(uncovered & coverage[i])
            if count > best_count:
                best_count = count
                best_i = i
            if time.time() > deadline - 5:
                break
        if best_i == -1:
            break
        code.append(list(all_words[best_i]))
        uncovered -= coverage[best_i]

    return code
