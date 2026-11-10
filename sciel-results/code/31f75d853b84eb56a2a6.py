import time
import random
import math

def _max_abs_ap_sum(signs, threshold=None):
    """
    Maximum absolute signed sum over ANY arithmetic progression contained in [0, n).
    For each chain (start s, step d), the max sub-AP sum equals max(prefix)-min(prefix)
    over prefix sums (including the 0 prefix).
    Pure-Python incremental tracking (no per-AP numpy allocation) for speed.
    Returns threshold+1 early if threshold is exceeded.
    """
    n = len(signs)
    max_sum = 0
    for d in range(1, n + 1):
        for s in range(d):
            if s >= n:
                break
            p = 0
            min_p = 0
            max_p = 0
            idx = s
            while idx < n:
                p += signs[idx]
                if p < min_p:
                    min_p = p
                    if max_p - min_p > max_sum:
                        max_sum = max_p - min_p
                        if threshold is not None and max_sum > threshold:
                            return max_sum
                elif p > max_p:
                    max_p = p
                    if max_p - min_p > max_sum:
                        max_sum = max_p - min_p
                        if threshold is not None and max_sum > threshold:
                            return max_sum
                idx += d
    return max_sum

def _get_tight_ap_info(signs, top_k=40):
    """
    Indices that appear in APs whose max sub-AP sum is near the global max.
    More constrained indices are better flip candidates.
    """
    n = len(signs)
    ap_sums = []
    max_sum = 0
    for d in range(1, n + 1):
        for s in range(d):
            if s >= n:
                break
            p = 0
            min_p = 0
            max_p = 0
            idx = s
            while idx < n:
                p += signs[idx]
                if p < min_p:
                    min_p = p
                elif p > max_p:
                    max_p = p
                idx += d
            cur = max_p - min_p
            if cur > 0:
                ap_sums.append((cur, s, d))
            if cur > max_sum:
                max_sum = cur
    if max_sum == 0:
        return list(range(n))
    tight_threshold = max_sum - 1
    idx_count = [0] * n
    for val, s, d in ap_sums:
        if val >= tight_threshold:
            idx = s
            while idx < n:
                idx_count[idx] += 1
                idx += d
    order = sorted(range(n), key=lambda i: -idx_count[i])
    return order[:top_k]

def _thue_morse(n):
    return [1 if bin(i).count('1') % 2 == 0 else -1 for i in range(n)]

def _van_der_corput(n):
    out = []
    for i in range(1, n + 1):
        x = 0.0
        f = 1.0
        m = i
        while m:
            f /= 2.0
            x += f * (m & 1)
            m >>= 1
        out.append(1 if x < 0.5 else -1)
    return out

def _rademacher(n):
    out = []
    for i in range(n):
        r = i
        x = 0.0
        f = 0.5
        while r:
            x += f * (r & 1)
            f *= 0.5
            r >>= 1
        out.append(1 if x < 0.5 else -1)
    return out

def _greedy(n):
    g = [0] * n
    for i in range(n):
        best_s, best_v = 1, 10 ** 9
        for sgn in (1, -1):
            g[i] = sgn
            worst = 0
            for d in range(1, i + 1):
                total = 0
                j = i
                while j >= 0:
                    total += g[j]
                    a = total if total >= 0 else -total
                    if a > worst:
                        worst = a
                    j -= d
            if worst < best_v:
                best_v, best_s = worst, sgn
        g[i] = best_s
    return g

def construct():
    start_time = time.time()
    deadline = start_time + 83.0  # safety margin under 90s wall-clock
    n = 256
    rng = random.Random(20240607)

    # ---- Structured + random initializations ----
    inits = []
    inits.append([1 if i % 2 == 0 else -1 for i in range(n)])
    inits.append([1 if i % 2 == 1 else -1 for i in range(n)])
    inits.append(_thue_morse(n))
    inits.append(_van_der_corput(n))
    inits.append(_rademacher(n))
    inits.append([1 if (i // 2) % 2 == 0 else -1 for i in range(n)])
    inits.append([1 if (i // 3) % 2 == 0 else -1 for i in range(n)])
    inits.append([1 if (i // 4) % 2 == 0 else -1 for i in range(n)])
    inits.append(_greedy(n))
    for _ in range(10):
        inits.append([rng.choice((-1, 1)) for _ in range(n)])

    best_signs = None
    best_score = float('inf')
    for init in inits:
        sc = _max_abs_ap_sum(init)
        if sc < best_score:
            best_score = sc
            best_signs = init[:]

    # ---- Simulated-annealing local search over single flips ----
    current = best_signs[:]
    cur_score = best_score
    T = 3.0
    T_min = 0.05
    no_improve = 0
    max_no_improve = 250
    candidates = _get_tight_ap_info(current, top_k=40)
    cand_refresh = 0

    while time.time() < deadline:
        if no_improve > max_no_improve:
            if rng.random() < 0.5:
                current = best_signs[:]
                for _ in range(rng.randint(4, 16)):
                    current[rng.randint(0, n - 1)] = -current[rng.randint(0, n - 1)]
            else:
                current = [rng.choice((-1, 1)) for _ in range(n)]
            cur_score = _max_abs_ap_sum(current)
            if cur_score < best_score:
                best_score = cur_score
                best_signs = current[:]
            no_improve = 0
            candidates = _get_tight_ap_info(current, top_k=40)
            continue

        cand_refresh += 1
        if cand_refresh >= 25:
            candidates = _get_tight_ap_info(current, top_k=40)
            cand_refresh = 0
            if not candidates:
                candidates = list(range(n))

        if rng.random() < 0.75 and candidates:
            idx = candidates[rng.randint(0, len(candidates) - 1)]
        else:
            idx = rng.randint(0, n - 1)

        old = current[idx]
        current[idx] = -old
        new_score = _max_abs_ap_sum(current, threshold=cur_score)

        if new_score < cur_score:
            cur_score = new_score
            no_improve = 0
            if new_score < best_score:
                best_score = new_score
                best_signs = current[:]
        else:
            # simulated-annealing acceptance to escape local optima
            delta = new_score - cur_score
            if delta <= 0:
                accept = True
            else:
                accept = (delta <= 2) and (rng.random() < math.exp(-delta / max(T, T_min)))
            if accept:
                cur_score = new_score
                no_improve = 0
            else:
                current[idx] = old
                no_improve += 1

        T = max(T_min, T * 0.999)

    return best_signs
