"""Combined program: Golomb-ruler-anchored search + FFT scorer + refined polish."""
import time
import math
import random
import numpy as np

_start_time = time.time()
_DEADLINE = 138  # safety margin under the 150 s wall-clock kill

def _elapsed():
    return time.time() - _start_time

def _next_pow2(n):
    return 1 << (n - 1).bit_length()

# Known near-optimal Golomb rulers (0 included). A Golomb ruler has all
# pairwise differences distinct, so |U-U| is maximal (2n(n-1)+1) for its size.
_KNOWN_GOLOMB = {
    1: [0],
    2: [0, 1],
    3: [0, 1, 3],
    4: [0, 1, 4, 6],
    5: [0, 1, 4, 10, 11],
    6: [0, 1, 4, 10, 12, 17],
    7: [0, 1, 4, 10, 12, 17, 22],
    8: [0, 1, 4, 10, 12, 17, 22, 29],
    9: [0, 1, 4, 10, 12, 17, 22, 29, 36],
    10: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44],
    11: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52],
    12: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60],
    13: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72],
    14: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85],
    15: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99],
    16: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99, 115],
    17: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99, 115, 132],
    18: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99, 115, 132, 151],
    19: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99, 115, 132, 151, 171],
    20: [0, 1, 4, 10, 12, 17, 22, 29, 36, 44, 52, 60, 72, 85, 99, 115, 132, 151, 171, 194],
}

def _score_set(U):
    """score = 1 + log(|U-U|/|U+U|)/log(2*max(U)+1); -1 if invalid."""
    if not U or 0 not in U:
        return -1.0
    n = len(U)
    max_u = max(U)
    if max_u == 0:
        return -1.0

    if n <= 30:
        arr = np.array(U, dtype=np.int64)
        sums = arr[:, None] + arr[None, :]
        diffs = arr[:, None] - arr[None, :]
        num_sums = int(np.unique(sums).size)
        num_diffs = int(np.unique(diffs).size)
    else:
        size = 2 * max_u + 1
        fft_size = _next_pow2(2 * size)
        f = np.zeros(fft_size, dtype=np.complex128)
        for u in U:
            f[u] = 1.0
        F = np.fft.fft(f)
        sum_conv = np.fft.ifft(F * F).real
        num_sums = int(np.sum(sum_conv > 0.5))

        f_rev = np.zeros(fft_size, dtype=np.complex128)
        f_rev[0] = f[0]
        f_rev[1:] = f[1:][::-1]
        F_rev = np.fft.fft(f_rev)
        diff_conv = np.fft.ifft(F * F_rev).real
        num_diffs = int(np.sum(diff_conv[:max_u + 1] > 0.5)) + int(np.sum(diff_conv[fft_size - max_u:] > 0.5))

    if num_sums == 0:
        return -1.0
    ratio = num_diffs / num_sums
    if ratio <= 1:
        return -1.0
    log_denom = math.log(2 * max_u + 1)
    if log_denom <= 0:
        return -1.0
    return 1.0 + math.log(ratio) / log_denom

def _greedy_golomb(start, target_n):
    """Greedy Golomb ruler starting from `start` (must contain 0)."""
    ruler = sorted(set(start))
    diffs = set()
    for i in range(len(ruler)):
        for j in range(i + 1, len(ruler)):
            diffs.add(ruler[j] - ruler[i])
    pos = ruler[-1] + 1
    while len(ruler) < target_n:
        ok = True
        for v in ruler:
            if pos - v in diffs:
                ok = False
                break
        if ok:
            for v in ruler:
                diffs.add(pos - v)
            ruler.append(pos)
        pos += 1
    return ruler

def _local_search(initial, max_steps=4000):
    """Add/remove/swap local search to improve score."""
    best_U = sorted(set(initial))
    best_score = _score_set(best_U)
    current_U = list(best_U)
    current_score = best_score
    stall = 0

    for _ in range(max_steps):
        if _elapsed() > _DEADLINE:
            break
        max_val = max(current_U) if current_U else 0
        cur_set = set(current_U)
        improved = False

        candidates = set()
        for mult in [2, 3, 5, 7, 11]:
            for d in range(1, 25):
                candidates.add(max_val * mult + d)
        for a in current_U:
            for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 37, 47]:
                candidates.add(a + p)
                candidates.add(a + p * 2)
                candidates.add(a + p * 3)
        if max_val > 8:
            step_size = max(1, max_val // 60)
            for v in range(1, max_val + 1, step_size):
                candidates.add(v)
        for _ in range(30):
            candidates.add(random.randint(1, max(2, max_val * 5)))
        candidates = [v for v in candidates if v > 0 and v not in cur_set]
        random.shuffle(candidates)

        for new_val in candidates[:35]:
            if new_val in cur_set:
                continue
            new_U = current_U + [new_val]
            new_U.sort()
            new_score = _score_set(new_U)
            if new_score > current_score + 1e-10:
                current_U = new_U
                current_score = new_score
                cur_set = set(current_U)
                if current_score > best_score:
                    best_score = current_score
                    best_U = list(current_U)
                improved = True
                stall = 0
                break

        if not improved and len(current_U) > 2:
            for idx in random.sample(range(1, len(current_U)), min(15, len(current_U) - 1)):
                new_U = [x for i, x in enumerate(current_U) if i != idx]
                new_score = _score_set(new_U)
                if new_score > current_score + 1e-10:
                    current_U = new_U
                    current_score = new_score
                    cur_set = set(current_U)
                    if current_score > best_score:
                        best_score = current_score
                        best_U = list(current_U)
                    improved = True
                    stall = 0
                    break

        if not improved:
            stall += 1
            if stall > 40 and len(current_U) > 2:
                idx = random.randint(1, len(current_U) - 1)
                new_val = random.randint(1, max(2, max(current_U) * 5))
                if new_val not in cur_set:
                    current_U[idx] = new_val
                    current_U.sort()
                    cur_set = set(current_U)
                    current_score = _score_set(current_U)
                    stall = 0
            elif stall > 120:
                break

    return best_U, best_score

def construct():
    global _start_time
    _start_time = time.time()
    random.seed(12345)

    best_U = [0, 1, 3]
    best_score = _score_set(best_U)

    # 1. Golomb rulers: maximal |U-U| for the size; minimize span via local search.
    for n in range(3, 21):
        if _elapsed() > _DEADLINE - 40:
            break
        start = _KNOWN_GOLOMB.get(n, _greedy_golomb([0, 1], n))
        U = list(start)
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
        if _elapsed() < _DEADLINE - 20:
            U2, s2 = _local_search(U, max_steps=1500)
            if s2 > best_score:
                best_score = s2
                best_U = U2

    # 2. Exhaustive small sets (directly optimizes the ratio for tiny n).
    for a in range(1, 40):
        if _elapsed() > _DEADLINE - 30:
            break
        for b in range(a + 1, 40):
            if _elapsed() > _DEADLINE - 30:
                break
            for c in range(b + 1, 42):
                s = _score_set([0, a, b, c])
                if s > best_score:
                    best_score = s
                    best_U = [0, a, b, c]

    # 3. Geometric / sparse seeds + local search.
    for base in range(2, 20):
        if _elapsed() > _DEADLINE - 15:
            break
        U = [0]
        val = 1
        for _ in range(1, 9):
            val *= base
            U.append(val)
        s = _score_set(U)
        if s > best_score:
            best_score = s
            best_U = U[:]
        if _elapsed() < _DEADLINE - 8:
            U2, s2 = _local_search(U, max_steps=800)
            if s2 > best_score:
                best_score = s2
                best_U = U2

    # 4. Main local search from the best.
    if _elapsed() < _DEADLINE - 5:
        U, s = _local_search(best_U, max_steps=4000)
        if s > best_score:
            best_score = s
            best_U = U

    # 5. Final high-precision polish: fine add/remove sweep.
    if _elapsed() < _DEADLINE - 1:
        max_val = max(best_U)
        cur_set = set(best_U)
        cands = set()
        for a in best_U:
            for d in range(1, 40):
                cands.add(a + d)
                cands.add(a - d)
            for m in [2, 3, 5, 7, 11]:
                cands.add(a * m)
                cands.add(a * m + 1)
                cands.add(a * m - 1)
        cands.update(range(1, min(max_val + 40, 1500)))
        cands.update(range(max_val + 1, max_val + 80))
        cands = sorted(v for v in cands if v > 0 and v not in cur_set)
        for nv in cands:
            if _elapsed() > _DEADLINE - 1:
                break
            new_U = sorted(best_U + [nv])
            ns = _score_set(new_U)
            if ns > best_score + 1e-12:
                best_score = ns
                best_U = new_U
                cur_set = set(best_U)
                max_val = max(max_val, nv)
        if len(best_U) > 2:
            for idx in range(1, len(best_U)):
                if _elapsed() > _DEADLINE - 1:
                    break
                new_U = [x for i, x in enumerate(best_U) if i != idx]
                ns = _score_set(new_U)
                if ns > best_score + 1e-12:
                    best_score = ns
                    best_U = new_U

    if 0 not in best_U:
        best_U.append(0)
    return sorted(best_U)
