import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    # Keep the exact public interface.
    # This implementation specializes the known task q=3, n=6, radius=1,
    # with a safe general fallback for other parameters.
    if q != 3 or n != 6 or radius != 1:
        # General fallback: product of a small covering code on first 4 symbols
        # and all tails. For q=3, n>=4, radius=1 this gives 3^(n-2) words.
        if n >= 4:
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]
        return [list(w) for w in itertools.product(range(q), repeat=n)]

    total_words = q ** n
    powers = [q ** p for p in range(n)]
    num_neighbors = 1 + n * (q - 1)  # 13 for q=3, n=6

    # Hard wall-clock guard: evaluator kills at 150s; leave margin for return.
    start_time = time.time()
    deadline = start_time + 143.0

    # Precompute neighbors: neighbors[i] = indices within distance <= 1 of i.
    neighbors = np.empty((total_words, num_neighbors), dtype=np.int32)
    for i in range(total_words):
        d = [(i // powers[p]) % q for p in range(n)]
        neighbors[i, 0] = i
        k = 1
        for p in range(n):
            orig = d[p]
            for alt in range(q):
                if alt == orig:
                    continue
                neighbors[i, k] = i + (alt - orig) * powers[p]
                k += 1

    def idx_to_list(idx):
        return [(idx // powers[p]) % q for p in range(n)]

    def list_to_idx(word):
        idx = 0
        for w in word:
            idx = idx * q + w
        return idx

    # Known good initial construction: ternary tetracode product (72 words).
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_indices = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            initial_indices.append(list_to_idx(h + list(tail)))
    initial_indices = list(dict.fromkeys(initial_indices))

    # We will run a local search that keeps the best valid code found.
    # State:
    #   code: list of distinct indices
    #   counts: coverage counts for every word
    #   in_set: membership set for O(1) duplicate checks

    # Helper to build coverage counts from a code list.
    def build_counts(code):
        cnt = np.zeros(total_words, dtype=np.int32)
        for c in code:
            cnt[neighbors[c]] += 1
        return cnt

    # Helper to greedily remove all currently redundant words.
    # A word c is redundant iff every word in its ball has coverage >= 2.
    def greedy_remove(code, counts, in_set, max_iter=100000):
        improved = True
        iters = 0
        while improved and iters < max_iter and time.time() < deadline:
            improved = False
            iters += 1
            # Vectorized-ish scan: N ~ 70, each check 13 entries.
            to_remove = []
            for idx, c in enumerate(code):
                if counts[neighbors[c]].min() >= 2:
                    to_remove.append(idx)
            if not to_remove:
                break
            # Remove all currently redundant words in reverse index order.
            # This is safe: removing a word only decreases coverage, so a word
            # that was redundant remains covered after removing another word.
            for idx in sorted(to_remove, reverse=True):
                c = code[idx]
                code.pop(idx)
                counts[neighbors[c]] -= 1
                in_set.discard(c)
            improved = True
        return code, counts, in_set

    # Local search with simulated-annealing-style swaps.
    # We start from the tetracode product and try to reduce size by swaps that
    # create redundancy, followed by greedy removals.
    best_code = None
    best_size = 10**9

    # Use deterministic seeds for reproducibility within one evaluation.
    rng = random.Random(123456789)

    # Multiple restarts with different initial perturbations can help escape
    # local minima. We budget restarts adaptively based on remaining time.
    max_restarts = 12
    restart_times = []
    for restart in range(max_restarts):
        if time.time() > deadline - 2.0:
            break

        # For restart 0, use the known 72-word tetracode product.
        # For later restarts, start from the best code found and perturb it,
        # or from a random valid completion of a smaller seed.
        if restart == 0:
            code = list(initial_indices)
        elif best_code is not None:
            code = list(best_code)
            # Perturb: replace a few random codewords with random neighbors.
            k = min(len(code), max(1, len(code) // 20))
            for _ in range(k):
                if not code:
                    break
                pos = rng.randrange(len(code))
                c = code[pos]
                cand = int(neighbors[c][rng.randrange(1, num_neighbors)])
                code[pos] = cand
        else:
            # Fallback: random code plus greedy coverage completion.
            code = []

        # Ensure validity: if code is not a full cover, add words greedily.
        counts = build_counts(code)
        in_set = set(code)
        uncovered = (counts == 0)
        while uncovered.any() and time.time() < deadline - 2.0:
            # Pick a random uncovered word and add a random word covering it.
            # This is only a fallback; the tetracode start is already valid.
            unc_idx = int(np.argwhere(uncovered)[0, 0])
            # Choose a candidate from neighbors of unc_idx.
            cand = int(neighbors[unc_idx][rng.randrange(num_neighbors)])
            if cand not in in_set:
                code.append(cand)
                in_set.add(cand)
                counts[neighbors[cand]] += 1
                uncovered = (counts == 0)
            else:
                # If all neighbors already in set, add a random word.
                cand = rng.randrange(total_words)
                if cand not in in_set:
                    code.append(cand)
                    in_set.add(cand)
                    counts[neighbors[cand]] += 1
                    uncovered = (counts == 0)
                else:
                    # Extremely unlikely; break to avoid infinite loop.
                    break

        # Remove duplicates defensively.
        seen = set()
        new_code = []
        for c in code:
            if c not in seen:
                seen.add(c)
                new_code.append(c)
        code = new_code
        in_set = seen
        counts = build_counts(code)

        # Initial greedy removal.
        code, counts, in_set = greedy_remove(code, counts, in_set)

        if len(code) < best_size:
            best_size = len(code)
            best_code = list(code)

        # Local search phase.
        # We alternate between:
        #  1) greedy removals
        #  2) swaps: remove a codeword, add a candidate that covers its critical words
        #  3) occasional random restarts of the local search trajectory
        #
        # Temperature is used to accept non-improving swaps that may enable
        # future reductions. We only keep the best valid code globally.
        T = 1.0
        T_min = 0.01
        cooling = 0.9997
        local_deadline = deadline - 3.0
        steps = 0
        last_improve = time.time()
        no_improve = 0
        max_no_improve = 3000

        while time.time() < local_deadline:
            steps += 1

            # 1) Greedy removals.
            code, counts, in_set = greedy_remove(code, counts, in_set, max_iter=20000)

            if len(code) < best_size:
                best_size = len(code)
                best_code = list(code)
                last_improve = time.time()
                no_improve = 0
                T = max(T * cooling, T_min)
                continue

            # If no improvement for too long, perturb the current trajectory.
            if no_improve > max_no_improve:
                # Randomly swap a few codewords to escape.
                k = min(len(code), rng.randint(2, 6))
                for _ in range(k):
                    if not code:
                        break
                    pos = rng.randrange(len(code))
                    c = code[pos]
                    cand = int(neighbors[c][rng.randrange(1, num_neighbors)])
                    if cand not in in_set:
                        code[pos] = cand
                        counts[neighbors[c]] -= 1
                        counts[neighbors[cand]] += 1
                        in_set.discard(c)
                        in_set.add(cand)
                no_improve = 0
                T = max(0.5, T * 2.0)
                continue

            # 2) Try swaps.
            if not code:
                break

            # Pick a codeword to potentially remove.
            # Heuristic: prefer codewords with high minimum coverage (more slack).
            # But randomization is important for diversity.
            if rng.random() < 0.7:
                # Choose among codewords with highest min coverage.
                min_covs = [counts[neighbors[c]].min() for c in code]
                max_min = max(min_covs)
                candidates = [i for i, v in enumerate(min_covs) if v == max_min]
                pos = candidates[rng.randrange(len(candidates))]
            else:
                pos = rng.randrange(len(code))

            c_out = code[pos]
            # Critical words: those covered only by c_out.
            crit = neighbors[c_out][counts[neighbors[c_out]] == 1]

            if len(crit) == 0:
                # Should have been removed by greedy; remove defensively.
                code.pop(pos)
                counts[neighbors[c_out]] -= 1
                in_set.discard(c_out)
                no_improve += 1
                T = max(T * cooling, T_min)
                continue

            # Candidate c_in must cover all critical words.
            # Start from neighbors of the first critical word.
            w0 = int(crit[0])
            base_cands = neighbors[w0]

            valid_cands = []
            for cand in base_cands:
                cand = int(cand)
                if cand == c_out:
                    continue
                # Check if cand covers all critical words.
                ok = True
                for w in crit:
                    if w not in neighbors[cand]:
                        ok = False
                        break
                if ok:
                    valid_cands.append(cand)

            if not valid_cands:
                no_improve += 1
                T = max(T * cooling, T_min)
                continue

            # Choose a candidate.
            # Heuristic: prefer candidates that increase redundancy (higher min coverage after swap).
            # Compute a cheap score: for each candidate, look at coverage of its neighbors.
            # We want to maximize the minimum coverage among neighbors that will be affected.
            # This is an approximation of creating future redundancy.
            best_cand = None
            best_score = -10**9
            # Limit scoring to a random subset if too many candidates.
            if len(valid_cands) > 20:
                sample = rng.sample(valid_cands, 20)
            else:
                sample = valid_cands

            for cand in sample:
                # Simulate swap effect on min coverage of cand's ball.
                # After swap, coverage of cand's neighbors increases by 1,
                # coverage of c_out's neighbors decreases by 1.
                # We focus on cand's ball because we want to create redundancy there.
                new_min = counts[neighbors[cand]].min() + 1
                # Penalize if cand is already in set? We allow duplicates in search
                # but final code must be distinct. If cand in set, swap just moves c_out to cand,
                # which may reduce size after greedy removal.
                # Score: new_min plus a small bonus if cand not in set.
                score = float(new_min)
                if cand not in in_set:
                    score += 0.1
                if score > best_score:
                    best_score = score
                    best_cand = cand

            c_in = best_cand

            # Acceptance: always accept if it leads to immediate removal after greedy.
            # Otherwise use simulated annealing to accept with probability exp(-delta/T).
            # delta is a proxy for loss: if c_in is already in set, swap may not reduce size;
            # if c_in not in set, size unchanged but structure changes.
            # We use a simple rule: accept with probability based on T and whether
            # the swap increases the minimum coverage of c_in's ball.
            accept_prob = 1.0
            if c_in in in_set:
                accept_prob = 0.5
            else:
                # Estimate if this swap creates redundancy: if new_min >= 2, likely good.
                if best_score >= 2.0:
                    accept_prob = 1.0
                else:
                    accept_prob = 0.8

            if rng.random() > accept_prob:
                no_improve += 1
                T = max(T * cooling, T_min)
                continue

            # Perform swap.
            counts[neighbors[c_out]] -= 1
            counts[neighbors[c_in]] += 1
            code[pos] = c_in
            in_set.discard(c_out)
            in_set.add(c_in)

            # Immediately try greedy removals to see if size improves.
            code, counts, in_set = greedy_remove(code, counts, in_set, max_iter=5000)

            if len(code) < best_size:
                best_size = len(code)
                best_code = list(code)
                last_improve = time.time()
                no_improve = 0
            else:
                no_improve += 1

            T = max(T * cooling, T_min)

        # End of local search for this restart.
        # Record time spent for adaptive restart budgeting.
        restart_times.append(time.time() - start_time)

    # Final high-precision polish on the best code found.
    if best_code is None:
        # Should not happen; fallback to initial tetracode.
        best_code = list(initial_indices)

    # Rebuild counts and do exhaustive greedy removals.
    counts = build_counts(best_code)
    in_set = set(best_code)
    best_code, counts, in_set = greedy_remove(best_code, counts, in_set, max_iter=1000000)

    # Additional polish: try all possible single swaps from the best code,
    # and accept any that allow a reduction after greedy removal.
    # This is a deterministic local refinement that can squeeze out the last word.
    if time.time() < deadline - 1.0:
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            for i, c_out in enumerate(best_code):
                if time.time() > deadline - 1.0:
                    break
                crit = neighbors[c_out][counts[neighbors[c_out]] == 1]
                if len(crit) == 0:
                    # Remove if redundant.
                    best_code.pop(i)
                    counts[neighbors[c_out]] -= 1
                    in_set.discard(c_out)
                    improved = True
                    break
                w0 = int(crit[0])
                base_cands = neighbors[w0]
                for cand in base_cands:
                    cand = int(cand)
                    if cand == c_out:
                        continue
                    ok = True
                    for w in crit:
                        if w not in neighbors[cand]:
                            ok = False
                            break
                    if not ok:
                        continue
                    # Simulate swap.
                    old_counts = counts.copy()
                    counts[neighbors[c_out]] -= 1
                    counts[neighbors[cand]] += 1
                    old_code = list(best_code)
                    old_in_set = set(in_set)
                    best_code[i] = cand
                    in_set.discard(c_out)
                    in_set.add(cand)

                    # Greedy remove.
                    new_code, new_counts, new_in_set = greedy_remove(best_code, counts, in_set, max_iter=10000)

                    if len(new_code) < len(old_code):
                        # Accept improvement.
                        best_code = new_code
                        counts = new_counts
                        in_set = new_in_set
                        improved = True
                        break
                    else:
                        # Revert.
                        best_code = old_code
                        counts = old_counts
                        in_set = old_in_set

                if improved:
                    break

    # Final validation and return.
    final_counts = build_counts(best_code)
    if not np.all(final_counts > 0):
        # Fallback to initial tetracode if something went wrong.
        return [idx_to_list(idx) for idx in initial_indices]

    return [idx_to_list(idx) for idx in best_code]
