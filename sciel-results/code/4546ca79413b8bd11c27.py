"""Ternary covering code K_3(7,1) using precomputed ball lists + fast local search."""
import time
import random
import numpy as np
from itertools import product

def construct(q, n, radius):
    """Construct a covering code of length n over alphabet size q with covering radius 'radius'."""
    start_time = time.time()
    deadline = start_time + 145

    if q == 3 and n == 7 and radius == 1:
        return _solve_ternary_7_1(start_time, deadline)

    return _fallback_construct(q, n, radius)

def _fallback_construct(q, n, radius):
    if n <= 0:
        return [[]]
    if n == 1:
        return [[i] for i in range(q)]
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        result = []
        for h in tetracode:
            for tail in product(range(3), repeat=n - 4):
                result.append(h + list(tail))
        return result
    return [list(w) for w in product(range(q), repeat=n)]

def _solve_ternary_7_1(start_time, deadline):
    q, n, radius = 3, 7, 1
    N = 3 ** n  # 2187
    all_words = list(product(range(3), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}

    # Precompute ball index lists
    ball_lists = []
    for i, w in enumerate(all_words):
        covered = [i]
        for pos in range(n):
            for val in range(3):
                if val == w[pos]:
                    continue
                nw = list(w)
                nw[pos] = val
                covered.append(word_to_idx[tuple(nw)])
        ball_lists.append(np.array(covered, dtype=np.int32))

    # Precompute ball sets for fast set operations
    ball_sets = [set(bl.tolist()) for bl in ball_lists]

    # Greedy set cover
    def greedy_cover():
        covered = np.zeros(N, dtype=bool)
        codewords = []
        uncovered_count = N
        while uncovered_count > 0:
            best_cw = -1
            best_gain = 0
            for i in range(N):
                gain = np.sum(~covered[ball_lists[i]])
                if gain > best_gain:
                    best_gain = gain
                    best_cw = i
            if best_cw == -1 or best_gain == 0:
                break
            codewords.append(best_cw)
            covered[ball_lists[best_cw]] = True
            uncovered_count -= best_gain
            if time.time() > deadline - 40:
                break
        if uncovered_count > 0:
            for i in range(N):
                if not covered[i]:
                    codewords.append(i)
                    covered[ball_lists[i]] = True
                    uncovered_count -= 1
        return codewords

    def local_search(codewords, time_budget):
        """Fast local search with 1-for-1 swaps, 2-for-1 merges, and removals."""
        cw_set = set(codewords)
        cov_count = np.zeros(N, dtype=np.int32)
        for cw in codewords:
            cov_count[ball_lists[cw]] += 1

        best_size = len(codewords)
        no_improve = 0
        max_no_improve = 3000

        while time.time() < time_budget:
            improved = False
            r = random.random()

            if r < 0.45 and codewords:
                # 1-for-1 swap
                idx = random.randint(0, len(codewords) - 1)
                old_cw = codewords[idx]
                new_cw = random.randint(0, N - 1)
                if new_cw != old_cw and new_cw not in cw_set:
                    old_set = ball_sets[old_cw]
                    new_set = ball_sets[new_cw]
                    lost = old_set - new_set
                    if lost:
                        lost_arr = np.array(list(lost), dtype=np.int32)
                        if np.min(cov_count[lost_arr]) < 2:
                            pass
                        else:
                            gained = new_set - old_set
                            cov_count[lost_arr] -= 1
                            if gained:
                                cov_count[np.array(list(gained), dtype=np.int32)] += 1
                            codewords[idx] = new_cw
                            cw_set.discard(old_cw)
                            cw_set.add(new_cw)
                            improved = True
                            no_improve = 0
                            if len(codewords) < best_size:
                                best_size = len(codewords)
                            continue
                    else:
                        # new covers everything old covers
                        gained = new_set - old_set
                        if gained:
                            cov_count[np.array(list(gained), dtype=np.int32)] += 1
                        codewords[idx] = new_cw
                        cw_set.discard(old_cw)
                        cw_set.add(new_cw)
                        improved = True
                        no_improve = 0
                        if len(codewords) < best_size:
                            best_size = len(codewords)
                        continue

            elif r < 0.70 and len(codewords) >= 2:
                # 2-for-1 merge
                i1 = random.randint(0, len(codewords) - 1)
                i2 = random.randint(0, len(codewords) - 1)
                if i1 == i2:
                    continue
                cw1 = codewords[i1]
                cw2 = codewords[i2]
                new_cw = random.randint(0, N - 1)
                if new_cw != cw1 and new_cw != cw2 and new_cw not in cw_set:
                    s1 = ball_sets[cw1]
                    s2 = ball_sets[cw2]
                    sc = ball_sets[new_cw]
                    only1 = s1 - s2 - sc
                    only2 = s2 - s1 - sc
                    both = (s1 & s2) - sc
                    valid = True
                    if only1:
                        if np.min(cov_count[np.array(list(only1), dtype=np.int32)]) < 2:
                            valid = False
                    if valid and only2:
                        if np.min(cov_count[np.array(list(only2), dtype=np.int32)]) < 2:
                            valid = False
                    if valid and both:
                        if np.min(cov_count[np.array(list(both), dtype=np.int32)]) < 3:
                            valid = False
                    if valid:
                        cov_count[ball_lists[cw1]] -= 1
                        cov_count[ball_lists[cw2]] -= 1
                        cov_count[ball_lists[new_cw]] += 1
                        if i1 < i2:
                            codewords.pop(i2)
                            codewords.pop(i1)
                        else:
                            codewords.pop(i1)
                            codewords.pop(i2)
                        cw_set.discard(cw1)
                        cw_set.discard(cw2)
                        codewords.append(new_cw)
                        cw_set.add(new_cw)
                        improved = True
                        no_improve = 0
                        if len(codewords) < best_size:
                            best_size = len(codewords)
                        continue

            else:
                # Try removal
                if codewords:
                    idx = random.randint(0, len(codewords) - 1)
                    cw = codewords[idx]
                    if np.min(cov_count[ball_lists[cw]]) >= 2:
                        cov_count[ball_lists[cw]] -= 1
                        codewords.pop(idx)
                        cw_set.discard(cw)
                        improved = True
                        no_improve = 0
                        if len(codewords) < best_size:
                            best_size = len(codewords)
                        continue

            if not improved:
                no_improve += 1
                if no_improve > max_no_improve:
                    break

        return codewords

    # Main: greedy + local search with restarts
    best_codewords = None
    best_size = N

    # Initial greedy
    codewords = greedy_cover()
    t0 = time.time()
    codewords = local_search(codewords, time.time() + min(25, deadline - 5 - t0))
    best_codewords = codewords[:]
    best_size = len(codewords)

    # Restart with perturbation
    while time.time() < deadline - 8:
        # Perturb best solution
        perturbed = best_codewords[:]
        num_remove = min(len(perturbed), random.randint(5, 12))
        remove_idx = random.sample(range(len(perturbed)), num_remove)
        for idx in sorted(remove_idx, reverse=True):
            perturbed.pop(idx)

        # Re-greedy
        covered = np.zeros(N, dtype=bool)
        for cw in perturbed:
            covered[ball_lists[cw]] = True
        if not covered.all():
            remaining = N - np.sum(covered)
            while remaining > 0:
                best_cw = -1
                best_gain = 0
                for i in range(N):
                    gain = np.sum(~covered[ball_lists[i]])
                    if gain > best_gain:
                        best_gain = gain
                        best_cw = i
                if best_cw == -1 or best_gain == 0:
                    break
                perturbed.append(best_cw)
                covered[ball_lists[best_cw]] = True
                remaining -= best_gain

        # Local search
        time_left = deadline - 5 - time.time()
        if time_left < 3:
            break
        search_time = min(time_left, 18)
        perturbed = local_search(perturbed, time.time() + search_time)

        if len(perturbed) < best_size:
            best_size = len(perturbed)
            best_codewords = perturbed[:]

    # Final removal pass
    if best_codewords and time.time() < deadline - 2:
        cov_count = np.zeros(N, dtype=np.int32)
        for cw in best_codewords:
            cov_count[ball_lists[cw]] += 1
        changed = True
        while changed and time.time() < deadline - 2:
            changed = False
            for i in range(len(best_codewords)):
                if np.min(cov_count[ball_lists[best_codewords[i]]]) >= 2:
                    cov_count[ball_lists[best_codewords[i]]] -= 1
                    best_codewords.pop(i)
                    changed = True
                    break

    return [list(all_words[cw]) for cw in best_codewords]
