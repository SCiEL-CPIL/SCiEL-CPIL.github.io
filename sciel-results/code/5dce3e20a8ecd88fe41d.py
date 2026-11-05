import numpy as np
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 118.0

    if q == 3 and n == 6 and radius == 1:
        total = 729
        powers = np.array([1, 3, 9, 27, 81, 243], dtype=np.int32)

        # Build neighbor table: neighbors[i] contains i and all words at Hamming distance 1
        neighbors = np.empty((total, 13), dtype=np.int32)
        for i in range(total):
            d = [(i // powers[p]) % 3 for p in range(6)]
            neighbors[i, 0] = i
            k = 1
            for p in range(6):
                for alt in range(3):
                    if alt != d[p]:
                        neighbors[i, k] = i + (alt - d[p]) * int(powers[p])
                        k += 1

        # Build coverage matrix: covmat[i] = number of selected codewords whose ball covers word i
        # We'll track this via a vector that we update incrementally
        # For speed, precompute: for each codeword c, the list of words it covers is neighbors[c]

        def idx_to_list(idx):
            return [(idx // int(powers[p])) % 3 for p in range(6)]

        def greedy_init():
            """Fast greedy set cover initialization."""
            uncovered = np.ones(total, dtype=np.int8)
            uncovered_count = total
            codewords = []

            # Precompute for each word, how many uncovered words it covers
            # Use a lookup: cover_count[c] = number of uncovered words in neighbors[c]
            cover_count = uncovered.astype(np.int32).copy()
            # Actually we need: for each candidate c, how many uncovered words does neighbors[c] cover?
            # That's sum of uncovered over neighbors[c]

            while uncovered_count > 0:
                # Compute coverage for each candidate
                # scores[c] = sum of uncovered[neighbors[c]]
                scores = np.zeros(total, dtype=np.int32)
                for c in range(total):
                    scores[c] = np.sum(uncovered[neighbors[c]])

                # Pick best
                best = int(np.argmax(scores))
                if scores[best] == 0:
                    break

                codewords.append(best)
                newly_covered = neighbors[best]
                # Update uncovered
                still_uncovered = uncovered[newly_covered]
                n_covered = int(np.sum(still_uncovered))
                uncovered[newly_covered] = 0
                uncovered_count -= n_covered

            return codewords

        def local_search(codewords, time_budget):
            """Local search: try removals, swaps, 2-swaps to minimize code size."""
            cw = list(codewords)

            # Build coverage counts: cov_count[w] = number of codewords whose ball covers w
            cov_count = np.zeros(total, dtype=np.int32)
            for c in cw:
                cov_count[neighbors[c]] += 1

            best_size = len(cw)
            best_cw = list(cw)
            best_cov = cov_count.copy()

            def try_removals():
                """Remove all redundant words (those whose removal doesn't break coverage)."""
                nonlocal cw, cov_count, best_size, best_cw, best_cov
                changed = True
                while changed:
                    changed = False
                    for i in range(len(cw) - 1, -1, -1):
                        c = cw[i]
                        # Check if all words in neighbors[c] are still covered by >= 2 codewords
                        if np.min(cov_count[neighbors[c]]) >= 2:
                            cov_count[neighbors[c]] -= 1
                            cw.pop(i)
                            changed = True
                            if len(cw) < best_size:
                                best_size = len(cw)
                                best_cw = list(cw)
                                best_cov = cov_count.copy()
                return changed

            def try_swap():
                """Try 1-for-1 swap: remove one, add another that covers the critical words."""
                nonlocal cw, cov_count
                if len(cw) == 0:
                    return False
                idx_out = random.randint(0, len(cw) - 1)
                c_out = cw[idx_out]
                cov_count[neighbors[c_out]] -= 1
                critical = np.where(cov_count == 0)[0]
                if len(critical) == 0:
                    # Removal was valid, treat as removal
                    cw.pop(idx_out)
                    return True
                # Find a word that covers all critical words
                # A word w covers critical word c if c is in neighbors[w], i.e., w is in neighbors[c]
                # So w must be in intersection of neighbors[c] for all critical c
                valid = np.ones(total, dtype=np.bool_)
                for cr in critical:
                    valid[~np.isin(cr, neighbors, assume_unique=False)] = False
                # Simpler: w must be such that all critical words are in neighbors[w]
                # i.e., for all cr in critical, cr is in neighbors[w]
                # Equivalent: w is in neighbors[cr] for all cr in critical
                valid = np.ones(total, dtype=np.bool_)
                for cr in critical[:min(len(critical), 20)]:  # Limit for speed
                    # w must be in neighbors[cr]
                    in_ball = np.zeros(total, dtype=np.bool_)
                    in_ball[neighbors[cr]] = True
                    valid &= in_ball
                # Exclude existing codewords
                valid[cw] = False
                candidates = np.where(valid)[0]
                if len(candidates) > 0:
                    c_in = int(random.choice(candidates))
                    cw[idx_out] = c_in
                    cov_count[neighbors[c_in]] += 1
                    return True
                # Undo
                cov_count[neighbors[c_out]] += 1
                return False

            def try_2swap():
                """Try 2-for-2 swap."""
                nonlocal cw, cov_count
                if len(cw) < 2:
                    return False
                i1, i2 = random.sample(range(len(cw)), 2)
                c1, c2 = cw[i1], cw[i2]
                cov_count[neighbors[c1]] -= 1
                cov_count[neighbors[c2]] -= 1
                critical = np.where(cov_count == 0)[0]
                if len(critical) == 0:
                    cw.pop(i2)
                    cw.pop(i1)
                    return True
                # Need to find 2 words that cover all critical
                # First word must cover at least one critical
                # Try: pick c3 from neighbors of a critical word, then find c4 covering remaining
                w0 = critical[0]
                c3_candidates = neighbors[w0]
                existing = set(cw)
                for c3 in c3_candidates:
                    if c3 in existing:
                        continue
                    # Check what critical words c3 covers
                    c3_covers = neighbors[c3]
                    remaining = critical[~np.isin(critical, c3_covers)]
                    if len(remaining) == 0:
                        # c3 covers all critical; c4 can be anything not in existing
                        valid4 = np.ones(total, dtype=np.bool_)
                        valid4[list(existing)] = False
                        valid4[c3] = False
                        cands4 = np.where(valid4)[0]
                        if len(cands4) > 0:
                            c4 = int(random.choice(cands4))
                            cw[i1] = c3
                            cw[i2] = c4
                            cov_count[neighbors[c3]] += 1
                            cov_count[neighbors[c4]] += 1
                            return True
                    else:
                        # c4 must cover all remaining critical words
                        valid4 = np.ones(total, dtype=np.bool_)
                        for cr in remaining[:min(len(remaining), 15)]:
                            in_ball = np.zeros(total, dtype=np.bool_)
                            in_ball[neighbors[cr]] = True
                            valid4 &= in_ball
                        valid4[list(existing)] = False
                        valid4[c3] = False
                        cands4 = np.where(valid4)[0]
                        if len(cands4) > 0:
                            c4 = int(random.choice(cands4))
                            cw[i1] = c3
                            cw[i2] = c4
                            cov_count[neighbors[c3]] += 1
                            cov_count[neighbors[c4]] += 1
                            return True
                cov_count[neighbors[c1]] += 1
                cov_count[neighbors[c2]] += 1
                return False

            # Main loop
            stuck = 0
            max_stuck = 50
            while time.time() < time_budget:
                if try_removals():
                    stuck = 0
                    continue
                if try_swap():
                    stuck = 0
                    continue
                if try_2swap():
                    stuck = 0
                    continue
                stuck += 1
                if stuck > max_stuck:
                    stuck = 0
                    max_stuck = random.randint(30, 100)

            # Final removals
            try_removals()

            if len(best_cw) < len(cw):
                return best_cw
            return cw

        # Multi-restart
        best_overall = None
        best_overall_size = float('inf')

        while time.time() < deadline - 3:
            # Greedy init
            cw = greedy_init()
            if len(cw) >= best_overall_size:
                continue

            # Local search with remaining time
            remaining = deadline - 3 - time.time()
            if remaining < 5:
                break
            time_budget = min(remaining * 0.8, 20.0)
            cw = local_search(cw, time_budget)

            # Validate
            uncovered = np.ones(total, dtype=np.int8)
            for c in cw:
                uncovered[neighbors[c]] = 0
            if uncovered.sum() > 0:
                while uncovered.sum() > 0:
                    c = int(np.random.randint(total))
                    if c not in cw:
                        cw.append(c)
                        uncovered[neighbors[c]] = 0

            if len(cw) < best_overall_size:
                best_overall_size = len(cw)
                best_overall = list(cw)

            # If we hit 73, try to push lower with more time
            if best_overall_size <= 74 and time.time() < deadline - 10:
                extra_budget = min(time.time() + 15, deadline - 3)
                cw2 = local_search(best_overall, extra_budget)
                uncovered2 = np.ones(total, dtype=np.int8)
                for c in cw2:
                    uncovered2[neighbors[c]] = 0
                if uncovered2.sum() > 0:
                    while uncovered2.sum() > 0:
                        c = int(np.random.randint(total))
                        if c not in cw2:
                            cw2.append(c)
                            uncovered2[neighbors[c]] = 0
                if len(cw2) < best_overall_size:
                    best_overall_size = len(cw2)
                    best_overall = list(cw2)

        if best_overall is None:
            best_overall = [0]

        # Final validation
        uncovered = np.ones(total, dtype=np.int8)
        for c in best_overall:
            uncovered[neighbors[c]] = 0
        if uncovered.sum() > 0:
            while uncovered.sum() > 0:
                c = int(np.random.randint(total))
                if c not in best_overall:
                    best_overall.append(c)
                    uncovered[neighbors[c]] = 0

        return [idx_to_list(idx) for idx in best_overall]

    else:
        total = q ** n
        return [[0] * n]
