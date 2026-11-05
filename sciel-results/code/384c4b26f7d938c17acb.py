import time
import random
import math
import numpy as np
from scipy import signal

N = 128
LO = 128
HI = 256
DEADLINE = 90.0
SAFETY = 6.0

def _pair_counts(Aset):
    f = np.zeros(N + 1, dtype=np.float64)
    for a in Aset:
        f[a] = 1.0
    conv = signal.fftconvolve(f, f, mode='full')
    counts = np.empty(HI - LO + 1, dtype=np.int32)
    for s in range(LO, HI + 1):
        ordered = conv[s]
        diag = f[s // 2] if (s % 2 == 0) else 0.0
        counts[s - LO] = int(round((ordered + diag) / 2.0))
    return counts

def _evaluate(Aset):
    counts = _pair_counts(Aset)
    peak = int(counts.max()) if counts.size else 0
    uncovered = int(np.count_nonzero(counts == 0))
    return peak, uncovered

def _greedy_init(rng):
    A = {64, 128}
    while True:
        counts = _pair_counts(A)
        uncovered = set(s for s in range(LO, HI + 1) if counts[s - LO] == 0)
        if not uncovered:
            break
        best_el, best_gain = None, 0
        for el in range(0, N + 1):
            if el in A:
                continue
            gain = 0
            for b in A:
                if b < el:
                    continue
                s = el + b
                if s in uncovered:
                    gain += 1
            s2 = el + el
            if s2 in uncovered:
                gain += 1
            if gain > best_gain:
                best_gain = gain
                best_el = el
        if best_el is None or best_gain == 0:
            break
        A.add(best_el)
    return A

def construct():
    t0 = time.time()
    rng = random.Random(20240601)

    # Build several candidate initial sets
    candidates = []
    for seed in [0, 1, 2, 3, 4, 5, 6, 7]:
        if time.time() - t0 > DEADLINE - SAFETY:
            break
        r = random.Random(seed)
        A = _greedy_init(r)
        candidates.append(A)

    # Pick the best initial by (uncovered==0, peak)
    best_A = None
    best_peak = 10**9
    for A in candidates:
        p, u = _evaluate(A)
        if u == 0 and p < best_peak:
            best_peak = p
            best_A = set(A)

    if best_A is None:
        # fallback guaranteed cover
        return list(range(64, 129))

    # Local search to reduce peak while preserving coverage
    A = set(best_A)
    cur_peak, cur_unc = _evaluate(A)
    T = 0.8
    T_min = 0.03
    cooldown = 0.9997
    while time.time() - t0 < DEADLINE - SAFETY:
        T = max(T_min, T * cooldown)
        move_type = rng.random()
        A2 = set(A)
        if move_type < 0.5:
            if len(A2) <= 1:
                continue
            el = rng.choice(sorted(A2))
            A2.discard(el)
        elif move_type < 0.85:
            el = rng.randint(0, N)
            A2.add(el)
        else:
            if len(A2) < 1:
                continue
            el_out = rng.choice(sorted(A2))
            el_in = rng.randint(0, N)
            A2.discard(el_out)
            A2.add(el_in)

        new_peak, new_unc = _evaluate(A2)
        if new_unc > 0:
            continue
        delta = new_peak - cur_peak
        if delta <= 0:
            A = A2
            cur_peak = new_peak
            if cur_peak < best_peak:
                best_peak = cur_peak
                best_A = set(A)
        else:
            if rng.random() < math.exp(-delta / max(T, 1e-9)):
                A = A2
                cur_peak = new_peak

    final_peak, final_unc = _evaluate(best_A)
    if final_unc > 0:
        return list(range(64, 129))
    return sorted(best_A)
