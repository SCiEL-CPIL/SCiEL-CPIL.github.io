import math
import time
import random
import numpy as np

def construct():
    START = time.time()
    DEADLINE = START + 135  # 15s safety margin before 150s kill

    def elapsed():
        return time.time() - START

    def time_ok(margin=5):
        return elapsed() < DEADLINE - margin

    def score_bitset(u):
        """Fast exact scoring using Python integer bitsets."""
        if not u or 0 not in u or len(u) < 2:
            return -1.0
        max_u = max(u)
        m = 2 * max_u + 1
        if m <= 1:
            return 1.0
        # Build bitsets for sumset and diffset
        sumset = 0
        diffset = 0
        offset = max_u
        for a in u:
            for b in u:
                sumset |= (1 << (a + b))
                diffset |= (1 << (a - b + offset))
        ss = sumset.bit_count()
        ds = diffset.bit_count()
        if ss == 0:
            return -1.0
        r = ds / ss
        if r <= 0:
            return -1.0
        return 1.0 + math.log(r) / math.log(m)

    def score_np(u):
        """NumPy-based scoring for moderate sets."""
        u_arr = np.array(u, dtype=np.int64)
        n = len(u_arr)
        if n < 2 or 0 not in u_arr:
            return -1.0
        max_u = int(u_arr.max())
        m = 2 * max_u + 1
        if m <= 1:
            return 1.0
        if n <= 40 and max_u <= 2000:
            return score_bitset(u)
        sums = u_arr[:, None] + u_arr[None, :]
        ss = len(np.unique(sums))
        diffs = u_arr[:, None] - u_arr[None, :]
        ds = len(np.unique(diffs))
        r = ds / ss
        if r <= 0:
            return -1.0
        return 1.0 + math.log(r) / math.log(m)

    def score(u):
        u = sorted(set(u))
        return score_np(u)

    # Known good Golomb rulers and related sets
    known_sets = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 4, 9, 11],
        [0, 1, 4, 10, 12, 17],
        [0, 1, 4, 10, 18, 23, 25],
        [0, 1, 4, 10, 18, 23, 29, 34],
        [0, 1, 4, 10, 18, 23, 29, 37, 45],
        [0, 1, 4, 10, 18, 23, 29, 37, 45, 52],
        [0, 1, 4, 10, 18, 23, 29, 37, 45, 52, 60],
        [0, 1, 4, 10, 18, 23, 29, 37, 45, 52, 60, 72],
        [0, 1, 4, 10, 18, 23, 29, 37, 45, 52, 60, 72, 85],
        # Sidon-like / sparse sumset constructions
        [0, 1, 2, 5, 10, 20, 40, 80],
        [0, 1, 2, 5, 10, 20, 40],
        [0, 1, 2, 5, 10, 20],
        [0, 1, 3, 7, 15, 31],
        [0, 1, 3, 7, 15],
        [0, 1, 3, 7],
    ]

    best_set = [0, 1, 3]
    best_score = score(best_set)

    for u in known_sets:
        s = score(u)
        if s > best_score:
            best_score = s
            best_set = list(u)

    # Simulated annealing focused on sumset compression
    def sa_optimize(start_u, time_budget=20):
        t0 = time.time()
        cur = sorted(set(start_u))
        cur_s = score(cur)
        best = list(cur)
        best_s = cur_s
        T = 0.15
        T_min = 1e-8
        cool = 0.9985

        while time.time() - t0 < time_budget and time_ok(8):
            if T < T_min:
                break
            r = random.random()
            cand = list(cur)

            if r < 0.30 and len(cand) > 2:
                # Remove a random non-zero element
                idx = random.randint(1, len(cand) - 1)
                cand.pop(idx)
            elif r < 0.55:
                # Add element beyond current max (spread differences)
                mx = max(cand)
                new_v = random.randint(mx + 1, mx + len(cand) * len(cand) + 20)
                cand.append(new_v)
            elif r < 0.75:
                # Perturb a non-zero element (small move)
                idx = random.randint(1, len(cand) - 1)
                d = random.randint(-5, 5)
                nv = cand[idx] + d
                if nv >= 1:
                    cand[idx] = nv
                    cand = sorted(set(cand))
                    if len(cand) != len(cur) or 0 not in cand:
                        cand = list(cur)
                        T *= cool
                        continue
            elif r < 0.88:
                # Swap: remove one, add a new larger one
                if len(cand) > 2:
                    idx = random.randint(1, len(cand) - 1)
                    cand.pop(idx)
                    mx = max(cand) if cand else 0
                    new_v = random.randint(mx + 1, mx + len(cand) * len(cand) + 15)
                    cand.append(new_v)
            else:
                # Insert between two existing elements
                if len(cand) >= 2:
                    i = random.randint(1, len(cand) - 1)
                    lo = cand[i - 1]
                    hi = cand[i]
                    if hi - lo > 1:
                        new_v = random.randint(lo + 1, hi - 1)
                        cand.insert(i, new_v)
                        cand = sorted(set(cand))

            cand = sorted(set(cand))
            if len(cand) < 2 or 0 not in cand:
                T *= cool
                continue

            cs = score(cand)
            if cs == -1.0:
                T *= cool
                continue

            delta = cs - cur_s
            if delta > 0 or random.random() < math.exp(delta / T):
                cur = cand
                cur_s = cs
                if cs > best_s:
                    best_s = cs
                    best = list(cand)
            T *= cool

        return best, best_s

    # Run SA from multiple starting points
    random.seed(123)
    starts = [best_set] + known_sets[:8]
    for sp in starts:
        if not time_ok(12):
            break
        sa_u, sa_s = sa_optimize(sp, time_budget=15)
        if sa_s > best_score:
            best_score = sa_s
            best_set = sa_u

    # Intensive local search polish on the best found
    if time_ok(20):
        cur = list(best_set)
        cur_s = best_score
        improved = True
        while improved and time_ok(10):
            improved = False

            # Try adding elements
            mx = max(cur)
            for nv in range(mx + 1, mx + len(cur) * len(cur) + 30):
                cand = sorted(cur + [nv])
                s = score(cand)
                if s > cur_s + 1e-12:
                    cur = cand
                    cur_s = s
                    improved = True
                    break

            # Try removing elements
            if not improved and len(cur) > 2:
                for i in range(1, len(cur)):
                    cand = [x for j, x in enumerate(cur) if j != i]
                    s = score(cand)
                    if s > cur_s + 1e-12:
                        cur = cand
                        cur_s = s
                        improved = True
                        break

            # Try small perturbations
            if not improved:
                for i in range(1, len(cur)):
                    for d in [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]:
                        nv = cur[i] + d
                        if nv < 1:
                            continue
                        cand = list(cur)
                        cand[i] = nv
                        cand = sorted(set(cand))
                        if len(cand) != len(cur) or 0 not in cand:
                            continue
                        s = score(cand)
                        if s > cur_s + 1e-12:
                            cur = cand
                            cur_s = s
                            improved = True
                            break
                    if improved:
                        break

        if cur_s > best_score:
            best_score = cur_s
            best_set = cur

    # Final verification
    best_set = sorted(set(best_set))
    if 0 not in best_set:
        best_set = [0] + best_set
    if len(best_set) < 2:
        best_set = [0, 1]

    return best_set
