import itertools
import numpy as np
import time
import random

def construct(q, n, radius):
    # Fallback for non-standard parameters
    if not (q == 3 and n == 8 and radius == 1):
        if q == 3 and n == 4 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [list(word) for word in itertools.product(range(q), repeat=n)]

    start_time = time.time()
    deadline = start_time + 143.0  # Safety margin before 150s hard limit

    total_words = q ** n  # 6561

    # Precompute digits for all words
    digits_of = [None] * total_words
    for i in range(total_words):
        d = []
        x = i
        for _ in range(n):
            d.append(x % q)
            x //= q
        d.reverse()
        digits_of[i] = tuple(d)

    # Precompute neighbor lists using index arithmetic
    pos_powers = [q ** (n - 1 - pos) for pos in range(n)]

    neighbors = [None] * total_words
    for i in range(total_words):
        d = digits_of[i]
        neigh = [i]
        for pos in range(n):
            orig_val = d[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                idx = i + (new_val - orig_val) * pos_powers[pos]
                neigh.append(idx)
        neighbors[i] = np.array(neigh, dtype=np.int32)

    # Neighbor masks as boolean arrays for fast intersection
    neighbor_mask = np.zeros((total_words, total_words), dtype=bool)
    for i in range(total_words):
        neighbor_mask[i, neighbors[i]] = True

    # ---- Greedy construction (vectorized potential) ----
    def greedy_construct(existing_code=None):
        covered = np.zeros(total_words, dtype=bool)
        code_list = []
        if existing_code:
            for c in existing_code:
                covered[neighbors[c]] = True
                code_list.append(c)
        while not covered.all() and time.time() < deadline - 40.0:
            uncovered_idx = np.where(~covered)[0]
            if len(uncovered_idx) == 0:
                break
            potential = np.zeros(total_words, dtype=np.int32)
            batch_size = 1000
            for s in range(0, len(uncovered_idx), batch_size):
                batch = uncovered_idx[s:s+batch_size]
                potential += neighbor_mask[batch].sum(axis=0).astype(np.int32)
            c = int(np.argmax(potential))
            if potential[c] == 0:
                break
            code_list.append(c)
            covered[neighbors[c]] = True
        return code_list

    # ---- Local search: reduce-to-fixpoint + targeted swap pass ----
    def local_search(code, deadline_check):
        current_code = list(code)
        current_size = len(current_code)
        best_code = current_code[:]
        best_size = current_size

        current_cov_count = np.zeros(total_words, dtype=np.int32)
        for c in current_code:
            current_cov_count[neighbors[c]] += 1

        in_code = np.zeros(total_words, dtype=bool)
        for c in current_code:
            in_code[c] = True

        def reduce_fixpoint():
            # Remove any codeword whose neighbors are all covered >=2 times,
            # repeating until no more can be removed.
            improved = True
            while improved and time.time() < deadline_check:
                improved = False
                new_code = []
                for c in current_code:
                    if np.all(current_cov_count[neighbors[c]] > 1):
                        current_cov_count[neighbors[c]] -= 1
                        in_code[c] = False
                        improved = True
                    else:
                        new_code.append(c)
                current_code[:] = new_code

        def swap_pass():
            # When stuck, scan codewords (shuffled) for one with few unique
            # points and swap it for a non-code word covering those points.
            nonlocal current_size, best_size, best_code
            if not current_code:
                return False
            indices = list(range(len(current_code)))
            random.shuffle(indices)
            for idx in indices:
                if time.time() > deadline_check:
                    return False
                c = current_code[idx]
                u = neighbors[c]
                unique_mask = current_cov_count[u] == 1
                unique_points = u[unique_mask]

                if len(unique_points) == 0:
                    # redundant (should have been removed); force it
                    current_cov_count[neighbors[c]] -= 1
                    in_code[c] = False
                    current_code.pop(idx)
                    current_size -= 1
                    if current_size < best_size:
                        best_size = current_size
                        best_code = current_code[:]
                    return True

                if len(unique_points) > 3:
                    continue

                # candidates: intersection of neighbor masks of unique points
                if len(unique_points) == 1:
                    candidate_mask = neighbor_mask[unique_points[0]].copy()
                else:
                    candidate_mask = neighbor_mask[unique_points[0]].copy()
                    for x in unique_points[1:]:
                        candidate_mask &= neighbor_mask[x]
                        if not candidate_mask.any():
                            break

                candidate_mask[c] = False
                candidate_mask[in_code] = False

                valid_candidates = np.where(candidate_mask)[0]
                if len(valid_candidates) == 0:
                    continue

                # pick candidate maximizing new coverage (extra unique points gained)
                c2 = int(valid_candidates[0])
                best_gain = -1
                for cand in valid_candidates[:8]:
                    gain = int(np.sum(current_cov_count[neighbors[cand]] == 0))
                    if gain > best_gain:
                        best_gain = gain
                        c2 = int(cand)

                # apply swap
                current_cov_count[neighbors[c]] -= 1
                in_code[c] = False
                current_code[idx] = c2
                in_code[c2] = True
                current_cov_count[neighbors[c2]] += 1
                return True

            return False

        # Main loop
        while time.time() < deadline_check:
            reduce_fixpoint()
            current_size = len(current_code)
            if current_size < best_size:
                best_size = current_size
                best_code = current_code[:]

            if time.time() > deadline_check:
                break

            # stuck? try a swap
            if not swap_pass():
                # no productive move found this pass; nudge with a neutral random swap
                if not current_code or time.time() > deadline_check - 3.0:
                    break
                c1_idx = random.randint(0, len(current_code) - 1)
                c1 = current_code[c1_idx]
                u = neighbors[c1]
                unique_points = u[current_cov_count[u] == 1]
                if len(unique_points) > 2:
                    continue
                if len(unique_points) == 1:
                    cand = neighbor_mask[unique_points[0]].copy()
                else:
                    cand = neighbor_mask[unique_points[0]].copy()
                    for x in unique_points[1:]:
                        cand &= neighbor_mask[x]
                cand[c1] = False
                cand[in_code] = False
                valid = np.where(cand)[0]
                if len(valid) > 0:
                    c2 = int(random.choice(valid))
                    current_cov_count[neighbors[c1]] -= 1
                    in_code[c1] = False
                    current_code[c1_idx] = c2
                    in_code[c2] = True
                    current_cov_count[neighbors[c2]] += 1
                else:
                    break

        return best_code, best_size

    # ---- Final polish: reduce-to-fixpoint ----
    def final_polish(code):
        final_code = list(code)
        final_cov_count = np.zeros(total_words, dtype=np.int32)
        for idx in final_code:
            final_cov_count[neighbors[idx]] += 1

        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            new_code = []
            for c in final_code:
                if np.all(final_cov_count[neighbors[c]] > 1):
                    final_cov_count[neighbors[c]] -= 1
                    improved = True
                else:
                    new_code.append(c)
            final_code = new_code

        if not np.all(final_cov_count > 0):
            return code
        return final_code

    # ---- Main ----
    best_code_overall = None
    best_size_overall = float('inf')

    # 1. Greedy + local search
    greedy_code = greedy_construct()
    if len(greedy_code) < best_size_overall:
        best_code_overall = greedy_code[:]
        best_size_overall = len(greedy_code)

    ls_deadline = deadline - 30.0
    polished, size = local_search(greedy_code, ls_deadline)
    if size < best_size_overall:
        best_code_overall = polished[:]
        best_size_overall = size

    # 2. Perturbation restarts
    attempt = 0
    while time.time() < deadline - 45.0:
        attempt += 1
        random.seed(1000 + attempt)
        if best_code_overall is None:
            break

        perturbed = best_code_overall[:]
        num_remove = min(len(perturbed), random.randint(8, 20))
        remove_indices = random.sample(range(len(perturbed)), num_remove)
        for ri in sorted(remove_indices, reverse=True):
            perturbed.pop(ri)

        # re-greedy to fill gaps
        perturbed = greedy_construct(perturbed)

        polished, size = local_search(perturbed, deadline - 30.0)
        if size < best_size_overall:
            best_code_overall = polished[:]
            best_size_overall = size

    # 3. Final polish
    if best_code_overall is not None:
        best_code_overall = final_polish(best_code_overall)

    if best_code_overall is None:
        best_code_overall = greedy_code

    return [list(digits_of[i]) for i in best_code_overall]
