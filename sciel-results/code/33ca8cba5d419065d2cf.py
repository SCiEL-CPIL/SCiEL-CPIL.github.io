import numpy as np
import time

N = 100
N2 = N * N
N4 = N // 4


def _cyclic_corr(h):
    """Full cyclic autocorrelation via FFT. C[s] = sum_i h[i]*h[(i-s)%N]."""
    H = np.fft.rfft(h, 2 * N)
    return np.fft.irfft(H * np.conj(H), 2 * N)[:N]


def compute_score(h):
    """Maximum overlap of h with shifted (1-h) over all shifts s=1..N-1.
    overlap(s) = N/2 - C(s).  Minimize this."""
    C = _cyclic_corr(h)
    # s=0 gives overlap 0; consider s=1..N-1
    return N / 2.0 - np.min(C[1:])


def _delta_swap(h, i, j):
    """Change in C(s) for all s when flipping h[i]:1->0 and h[j]:0->1.
    Returns array d of length N where C[s] += d[s]."""
    hi = h[i]
    hj = h[j]
    diff = (1.0 - hj) - hi
    idx = (np.arange(N) - i) % N
    d = np.zeros(N)
    d[idx] += diff * h
    idx = (np.arange(N) - j) % N
    d[idx] += diff * h
    return d


def _local_search(h, deadline):
    """Hill-climb by random 1<->0 swaps (keeps sum constant)."""
    cur = compute_score(h)
    best = cur
    best_h = h.copy()
    h = h.copy()
    while time.time() < deadline:
        ones = np.where(h > 0.5)[0]
        zeros = np.where(h < 0.5)[0]
        if len(ones) == 0 or len(zeros) == 0:
            break
        i = int(np.random.choice(ones))
        j = int(np.random.choice(zeros))
        d = _delta_swap(h, i, j)
        # new overlap(s) = N/2 - (C[s]+d[s]); new max overlap = N/2 - min_s(C[s]+d[s])
        C = _cyclic_corr(h)
        new_score = N / 2.0 - np.min((C[1:] + d[1:]))
        if new_score < cur - 1e-12:
            h[i] = 1.0 - h[i]
            h[j] = 1.0 - h[j]
            cur = new_score
            if cur < best - 1e-12:
                best = cur
                best_h = h.copy()
        # occasional full recompute to guard against drift
        if np.random.rand() < 0.02:
            cur = compute_score(h)
    return best_h, best


def construct():
    deadline = time.time() + 140.0
    np.random.seed(42)

    best_h = np.full(N, 0.5)
    best_score = compute_score(best_h)

    # ---- Structured initial patterns (from inspiration 2) ----
    patterns = []
    # alternating
    alt = np.tile([1.0, 0.0], N // 2)
    patterns.append(alt)
    # block: 1/4, 1/2, 1/4
    block = np.concatenate([np.full(N // 4, 1.0), np.full(N // 2, 0.0), np.full(N // 4, 1.0)])
    patterns.append(block)
    # quadratic residue style (pseudo-random balanced)
    for k in range(4):
        p = np.zeros(N)
        pos = np.random.choice(N, N // 2, replace=False)
        p[pos] = 1.0
        patterns.append(p)

    for p in patterns:
        p = np.clip(p, 0.0, 1.0)
        diff = N / 2.0 - np.sum(p)
        if abs(diff) > 1e-10:
            p = p + diff / N
            p = np.clip(p, 0.0, 1.0)
            diff2 = N / 2.0 - np.sum(p)
            if abs(diff2) > 1e-10:
                p = p + diff2 / N
        s = compute_score(p)
        if s < best_score:
            best_score = s
            best_h = p.copy()

    # ---- Local search from best pattern and from random restarts ----
    for restart in range(40):
        if time.time() > deadline - 5.0:
            break
        if restart == 0:
            start = best_h.copy()
        else:
            start = np.zeros(N)
            pos = np.random.choice(N, N // 2, replace=False)
            start[pos] = 1.0
        cand, sc = _local_search(start, deadline - 3.0)
        if sc < best_score:
            best_score = sc
            best_h = cand.copy()

    # ---- Continuous refinement: small gradient-ish perturbations ----
    # Try nudging values toward extremes (binary) while keeping sum.
    for _ in range(60):
        if time.time() > deadline - 2.0:
            break
        trial = best_h.copy()
        # push toward binary
        for _ in range(N):
            k = int(np.random.choice(N))
            if trial[k] > 0.5:
                trial[k] = min(1.0, trial[k] + 0.1)
            else:
                trial[k] = max(0.0, trial[k] - 0.1)
        # re-normalize sum to N/2
        diff = N / 2.0 - np.sum(trial)
        trial = trial + diff / N
        trial = np.clip(trial, 0.0, 1.0)
        diff2 = N / 2.0 - np.sum(trial)
        if abs(diff2) > 1e-10:
            trial = trial + diff2 / N
        s = compute_score(trial)
        if s < best_score:
            best_score = s
            best_h = trial.copy()

    # Final: ensure exact sum N/2
    diff = N / 2.0 - np.sum(best_h)
    if abs(diff) > 1e-10:
        best_h = best_h + diff / N
        best_h = np.clip(best_h, 0.0, 1.0)
    return best_h
