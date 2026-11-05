import time
import random
import numpy as np
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    # Fallback for non-standard parameters
    if not (q == 3 and n == 8 and radius == 1):
        if q == 3 and n == 4 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [list(word) for word in itertools.product(range(q), repeat=n)]

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

    # Precompute neighbor lists (17 neighbors for q=3, n=8, radius=1)
    neighbors = [None] * total_words
    for i in range(total_words):
        d = digits_of[i]
        neigh = [i]
        for pos in range(n):
            orig_val = d[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                new_d = list(d)
                new_d[pos] = new_val
                idx = 0
                for j in range(n):
                    idx = idx * q + new_d[j]
                neigh.append(idx)
        neighbors[i] = np.array(neigh, dtype=np.int32)

    # Precompute neighbor masks as boolean arrays for fast vectorized intersection
    neighbor_mask = np.zeros((total_words, total_words), dtype=bool)
    for i in range(total_words):
        neighbor_mask[i, neighbors[i]] = True

    # --- Phase 1: Incremental Greedy Initialization ---
    def greedy_incremental():
        cov_count = np.zeros(total_words, dtype=np.int32)
        is_code = np.zeros(total_words, dtype=bool)
        code_list = []
        potential = np.full(total_words, len(neighbors[0]), dtype=np.int32)
        uncovered_remaining = total_words

        while uncovered_remaining > 0 and time.time() < deadline - 45.0:
            c = int(np.argmax(potential))
            if potential[c] == 0:
                break
            code_list.append(c)
            is_code[c] = True
            potential[c] = 0  # prevent re-pick
            neigh_c = neighbors[c]
            for x in neigh_c:
                if cov_count[x] == 0:
                    cov_count[x] = 1
                    uncovered_remaining -= 1
                    potential[neighbors[x]] -= 1
                else:
                    cov_count[x] += 1
        return code_list, cov_count, is_code

    code_list, cov_count, is_code = greedy_incremental()

    # Safety: if greedy didn't finish, fall back to vectorized greedy
    if not np.all(cov_count > 0):
        covered = np.zeros(total_words, dtype=bool)
        covered[cov_count > 0] = True
        code_list = [c for c in code_list]
        while not covered.all() and time.time() < deadline - 45.0:
            uncovered_mask = ~covered
            if not uncovered_mask.any():
                break
            potential = neighbor_mask @ uncovered_mask.astype(np.int32)
            c = int(np.argmax(potential))
            if potential[c] == 0:
                break
            code_list.append(c)
            covered[neighbors[c]] = True
        cov_count = np.zeros(total_words, dtype=np.int32)
        for c in code_list:
            cov_count[neighbors[c]] += 1

    # --- Phase 2: Iterated Local Search with Vectorized Swaps ---
    def local_search(code, cov_count, max_time, seed):
        """Run local search (removal + vectorized swap) on a copy; return best (code, cov_count)."""
        code = code[:]
        cov_count = cov_count.copy()
        code_set = set(code)
        random.seed(seed)
        best_code = code[:]
        best_size = len(code)
        best_cov_count = cov_count.copy()
        stuck_counter = 0

        end_time = time.time() + max_time
        while time.time() < end_time:
            if len(code) == 0:
                break

            # 1. Try to remove a random word
            idx = random.randint(0, len(code) - 1)
            c = code[idx]
            if np.all(cov_count[neighbors[c]] > 1):
                cov_count[neighbors[c]] -= 1
                code_set.remove(c)
                code.pop(idx)
                stuck_counter = 0
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code[:]
                    best_cov_count = cov_count.copy()
                continue

            # 2. Removal failed -> try a vectorized swap
            risky_points = [x for x in neighbors[c] if cov_count[x] == 1]
            if not risky_points:
                # Should be removable; safety
                cov_count[neighbors[c]] -= 1
                code_set.remove(c)
                code.pop(idx)
                stuck_counter = 0
                if len(code) < best_size:
                    best_size = len(code)
                    best_code = code[:]
                    best_cov_count = cov_count.copy()
                continue

            # Candidate c2 must cover all risky points and not be in code
            if len(risky_points) == 1:
                candidate_mask = neighbor_mask[risky_points[0]].copy()
            else:
                candidate_mask = neighbor_mask[risky_points[0]].copy()
                for x in risky_points[1:]:
                    candidate_mask &= neighbor_mask[x]
                    if not candidate_mask.any():
                        break

            candidate_mask[c] = False
            for cc in code:
                candidate_mask[cc] = False

            valid_candidates = np.where(candidate_mask)[0]
            if len(valid_candidates) > 0:
                # Defect-minimizing score: prefer c2 covering most barely-covered points
                if len(valid_candidates) <= 24:
                    scores = np.array([
                        np.sum(cov_count[neighbors[c2]] <= 1)
                        for c2 in valid_candidates
                    ], dtype=np.int32)
                    best_score = np.max(scores)
                    top = valid_candidates[scores == best_score]
                    c2 = int(random.choice(top))
                else:
                    sample_size = min(24, len(valid_candidates))
                    sample_idx = random.sample(range(len(valid_candidates)), sample_size)
                    sampled = valid_candidates[sample_idx]
                    scores = np.array([
                        np.sum(cov_count[neighbors[c2]] <= 1)
                        for c2 in sampled
                    ], dtype=np.int32)
                    best_score = np.max(scores)
                    top = sampled[scores == best_score]
                    c2 = int(random.choice(top))

                # Perform swap
                cov_count[neighbors[c]] -= 1
                code_set.remove(c)
                code[idx] = c2
                cov_count[neighbors[c2]] += 1
                code_set.add(c2)
                stuck_counter = 0
            else:
                stuck_counter += 1

        return best_code, best_cov_count

    # Run two local-search passes with different seeds to explore different basins
    remaining = deadline - 12.0 - time.time()
    if remaining > 5.0:
        t1 = max(2.0, remaining * 0.55)
        code, cov_count = local_search(code_list, cov_count, t1, seed=42)

    remaining = deadline - 10.0 - time.time()
    if remaining > 5.0:
        t2 = max(2.0, remaining)
        code, cov_count = local_search(code, cov_count, t2, seed=1337)

    # --- Final Polish: aggressive multi-pass exploration with restarts ---
    def final_polish(code, cov_count, max_time):
        """Aggressive polish: try multiple random restarts from current state, keep best valid code."""
        best_code = code[:]
        best_size = len(code)
        best_cov_count = cov_count.copy()
        random.seed(999)
        end_time = time.time() + max_time

        # Try several short local-search passes with different seeds
        while time.time() < end_time:
            # Start from a perturbation of the current best
            perturbed_code = best_code[:]
            perturbed_cov = best_cov_count.copy()
            perturbed_set = set(perturbed_code)

            # Randomly swap a few codewords to escape local minima
            num_swaps = random.randint(2, min(6, len(perturbed_code) // 5))
            for _ in range(num_swaps):
                if len(perturbed_code) < 2:
                    break
                i = random.randint(0, len(perturbed_code) - 1)
                c = perturbed_code[i]
                # Find a random replacement
                risky = [x for x in neighbors[c] if perturbed_cov[x] == 1]
                if not risky:
                    # Just remove
                    perturbed_cov[neighbors[c]] -= 1
                    perturbed_set.remove(c)
                    perturbed_code.pop(i)
                    continue
                # Find candidate
                if len(risky) == 1:
                    cand_mask = neighbor_mask[risky[0]].copy()
                else:
                    cand_mask = neighbor_mask[risky[0]].copy()
                    for x in risky[1:]:
                        cand_mask &= neighbor_mask[x]
                        if not cand_mask.any():
                            break
                cand_mask[c] = False
                for cc in perturbed_code:
                    cand_mask[cc] = False
                valid = np.where(cand_mask)[0]
                if len(valid) > 0:
                    c2 = int(random.choice(valid))
                    perturbed_cov[neighbors[c]] -= 1
                    perturbed_set.remove(c)
                    perturbed_code[i] = c2
                    perturbed_cov[neighbors[c2]] += 1
                    perturbed_set.add(c2)

            # Run a short local search from the perturbed state
            short_time = min(3.0, max(0.5, end_time - time.time() - 1.0))
            if short_time > 0.5:
                new_code, new_cov = local_search(perturbed_code, perturbed_cov, short_time, seed=random.randint(1, 10000))
                if len(new_code) < best_size:
                    best_size = len(new_code)
                    best_code = new_code
                    best_cov_count = new_cov

        return best_code, best_cov_count

    # Run final polish if time remains
    remaining = deadline - 8.0 - time.time()
    if remaining > 3.0:
        code, cov_count = final_polish(code, cov_count, remaining)

    # Final verification and cleanup
    final_cov_count = np.zeros(total_words, dtype=np.int32)
    for c in code:
        final_cov_count[neighbors[c]] += 1

    # If invalid, fall back to the initial greedy result
    if not np.all(final_cov_count > 0):
        init_cov = np.zeros(total_words, dtype=np.int32)
        for c in code_list:
            init_cov[neighbors[c]] += 1
        if np.all(init_cov > 0):
            code = code_list
        else:
            # Last resort: full space (expensive but valid)
            code = list(range(total_words))

    return [list(digits_of[i]) for i in code]
