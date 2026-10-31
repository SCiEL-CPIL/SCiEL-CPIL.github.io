import math
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = start_time + 135.0  # Safety margin before 150s kill

    def remaining():
        return DEADLINE - time.time()

    def compute_score(u):
        """Compute score using NumPy for speed."""
        u_arr = np.array(sorted(set(u)), dtype=np.int64)
        n = len(u_arr)
        if n < 2:
            return 1.0
        max_u = int(u_arr.max())
        m = 2 * max_u + 1
        if m <= 1:
            return 1.0
        sums = u_arr[:, None] + u_arr[None, :]
        sumset_size = len(np.unique(sums))
        diffs = u_arr[:, None] - u_arr[None, :]
        diffset_size = len(np.unique(diffs))
        r = diffset_size / sumset_size
        if r <= 0:
            return -1.0
        return 1.0 + math.log(r) / math.log(m)

    def popcount(x):
        return bin(x).count('1')

    def build_state(u):
        """Build bitset state for a set u."""
        u = sorted(set(u))
        n = len(u)
        if n == 0:
            return None
        max_u = max(u)
        sum_bits = 0
        diff_bits = 0
        for a in u:
            for b in u:
                sum_bits |= (1 << (a + b))
                diff_bits |= (1 << (a - b + max_u))
        count_sum = popcount(sum_bits)
        count_diff = popcount(diff_bits)
        return (u, sum_bits, diff_bits, count_sum, count_diff, max_u)

    def score_from_state(state):
        u, sum_bits, diff_bits, count_sum, count_diff, max_u = state
        if count_sum == 0 or count_diff == 0:
            return 1.0
        m = 2 * max_u + 1
        if m <= 1:
            return 1.0
        r = count_diff / count_sum
        if r <= 0:
            return -1.0
        return 1.0 + math.log(r) / math.log(m)

    def marginal_add(state, new_val):
        u, sum_bits, diff_bits, count_sum, count_diff, max_u = state
        if new_val in u:
            return None
        new_max = max(max_u, new_val)
        new_sum_bits = 0
        for a in u:
            new_sum_bits |= (1 << (new_val + a))
        new_diff_bits = 0
        for a in u:
            new_diff_bits |= (1 << (new_val - a + new_max))
            new_diff_bits |= (1 << (a - new_val + new_max))
        if new_max != max_u:
            diff_bits = diff_bits << (new_max - max_u)
        combined_sum = sum_bits | new_sum_bits
        combined_diff = diff_bits | new_diff_bits
        new_count_sum = popcount(combined_sum)
        new_count_diff = popcount(combined_diff)
        new_u = sorted(u + [new_val])
        new_state = (new_u, combined_sum, combined_diff, new_count_sum, new_count_diff, new_max)
        new_score = score_from_state(new_state)
        return new_state, new_score

    def marginal_remove(state, val):
        u, sum_bits, diff_bits, count_sum, count_diff, max_u = state
        if val not in u or val == 0:
            return None
        new_u = [x for x in u if x != val]
        if not new_u:
            return None
        ns = build_state(new_u)
        if ns is None:
            return None
        new_score = score_from_state(ns)
        return ns, new_score

    def marginal_swap(state, old_val, new_val):
        u, sum_bits, diff_bits, count_sum, count_diff, max_u = state
        if old_val not in u or old_val == 0:
            return None
        if new_val in u:
            return None
        new_u = sorted([new_val if x == old_val else x for x in u])
        ns = build_state(new_u)
        if ns is None:
            return None
        new_score = score_from_state(ns)
        return ns, new_score

    def local_search(initial_u, time_budget=15.0):
        start_ls = time.time()
        state = build_state(list(initial_u))
        if state is None:
            return list(initial_u)
        best_score = score_from_state(state)
        best_u = list(state[0])
        no_improve = 0
        max_no_improve = 8

        while (time.time() - start_ls) < time_budget and remaining() > 8:
            u = state[0]
            max_u = state[5]
            improved = False
            best_cand = None
            best_cand_score = best_score

            # Try adds
            for v in range(1, max_u + len(u) * 4 + 15):
                if (time.time() - start_ls) > time_budget or remaining() < 8:
                    break
                if v in u:
                    continue
                res = marginal_add(state, v)
                if res is None:
                    continue
                ns, sc = res
                if sc > best_cand_score + 1e-12:
                    best_cand_score = sc
                    best_cand = ns

            # Try removes
            for v in u:
                if (time.time() - start_ls) > time_budget or remaining() < 8:
                    break
                if v == 0:
                    continue
                res = marginal_remove(state, v)
                if res is None:
                    continue
                ns, sc = res
                if sc > best_cand_score + 1e-12:
                    best_cand_score = sc
                    best_cand = ns

            # Try swaps (keep size constant)
            for i in range(1, len(u)):
                if (time.time() - start_ls) > time_budget or remaining() < 8:
                    break
                old_val = u[i]
                # Try nearby values
                for delta in range(-5, 6):
                    if delta == 0:
                        continue
                    new_val = old_val + delta
                    if new_val < 0 or new_val in u:
                        continue
                    if new_val == old_val:
                        continue
                    res = marginal_swap(state, old_val, new_val)
                    if res is None:
                        continue
                    ns, sc = res
                    if sc > best_cand_score + 1e-12:
                        best_cand_score = sc
                        best_cand = ns
                # Also try random larger values
                for _ in range(3):
                    new_val = random.randint(1, max_u + 30)
                    if new_val in u or new_val == old_val:
                        continue
                    res = marginal_swap(state, old_val, new_val)
                    if res is None:
                        continue
                    ns, sc = res
                    if sc > best_cand_score + 1e-12:
                        best_cand_score = sc
                        best_cand = ns

            if best_cand is not None:
                state = best_cand
                best_score = best_cand_score
                best_u = list(state[0])
                improved = True
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= max_no_improve:
                    # Random perturbation
                    u = list(state[0])
                    if len(u) > 2:
                        idx = random.randint(1, len(u) - 1)
                        old_val = u[idx]
                        new_val = random.randint(1, max(u) + 25)
                        if new_val not in u:
                            u[idx] = new_val
                            u.sort()
                            ns = build_state(u)
                            if ns is not None:
                                sc = score_from_state(ns)
                                if sc >= best_score - 0.02:
                                    state = ns
                                    best_u = list(state[0])
                                    no_improve = 0
                    else:
                        v = random.randint(1, max(u) + 15)
                        if v not in u:
                            u.append(v)
                            u.sort()
                            ns = build_state(u)
                            if ns is not None:
                                state = ns
                                best_u = list(state[0])
                            no_improve = 0

        return best_u

    # Known Golomb rulers
    known_rulers = {
        2: [0, 1],
        3: [0, 1, 3],
        4: [0, 1, 4, 6],
        5: [0, 1, 4, 9, 11],
        6: [0, 1, 4, 10, 12, 17],
        7: [0, 1, 4, 10, 12, 17, 22],
        8: [0, 1, 4, 10, 12, 17, 22, 29],
        9: [0, 1, 4, 10, 12, 17, 22, 29, 36],
        10: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44],
        11: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52],
        12: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60],
    }

    # Evaluate all known rulers directly
    best_set = [0, 1]
    best_score = 0.0
    for ruler in known_rulers.values():
        s = compute_score(ruler)
        if s > best_score:
            best_score = s
            best_set = list(ruler)

    # Run local search from each ruler
    all_candidates = [list(best_set)]
    for n in sorted(known_rulers.keys(), reverse=True):
        if remaining() < 10:
            break
        ruler = known_rulers[n]
        budget = min(10.0, remaining() * 0.12)
        if budget < 2.0:
            break
        improved = local_search(ruler, time_budget=budget)
        if improved:
            all_candidates.append(improved)
            s = compute_score(improved)
            if s > best_score:
                best_score = s
                best_set = list(improved)

    # Random initial sets
    for _ in range(5):
        if remaining() < 8:
            break
        n = random.randint(4, 12)
        u = [0] + sorted(random.sample(range(1, 80), n - 1))
        improved = local_search(u, time_budget=min(7.0, remaining() * 0.08))
        if improved:
            all_candidates.append(improved)
            s = compute_score(improved)
            if s > best_score:
                best_score = s
                best_set = list(improved)

    # Final refinement on best
    if remaining() > 6 and len(best_set) >= 3:
        refined = local_search(best_set, time_budget=min(12.0, remaining() * 0.5))
        if refined:
            s = compute_score(refined)
            if s > best_score:
                best_set = refined
                best_score = s

    return best_set
