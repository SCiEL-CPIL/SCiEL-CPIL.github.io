"""Erdos minimum overlap - spectral + local search design."""
import numpy as np
import time

N = 100
DEADLINE = time.time() + 130.0  # safety margin before 150s kill

def _overlap_max(h):
    """Max over shifts s of integral h(x)*(1-h(x-s)) dx, h on [0,2] with N steps.
    Using discrete approximation: h is a sequence of N values in [0,1].
    Overlap at shift k (in step units): sum_i h[i] * (1 - h[(i+k) mod N]) / N
    But h is on [0,2] with period 2, so shifts wrap mod N (circular).
    """
    h = np.asarray(h, dtype=float)
    n = len(h)
    # overlap(k) = sum_i h[i]*(1-h[(i+k)%n]) = sum_i h[i] - sum_i h[i]*h[(i+k)%n]
    # = S - autocorr(k), where S = sum h, autocorr(k)=sum_i h[i]*h[(i+k)%n]
    # Max overlap = S - min_k autocorr(k)  ... wait we want min over? No:
    # We minimize the MAX overlap. Max overlap = max_k (S - autocorr(k)) = S - min_k autocorr(k)
    S = h.sum()
    # autocorr via FFT
    f = np.fft.fft(h)
    ac = np.real(np.fft.ifft(f * np.conj(f)))  # circular autocorr
    min_ac = ac.min()
    return S - min_ac

def _overlap_max_direct(h):
    """Direct (slower, exact) check."""
    h = np.asarray(h, dtype=float)
    n = len(h)
    S = h.sum()
    best = 0.0
    for k in range(n):
        shifted = np.roll(h, -k)
        ov = (h * (1.0 - shifted)).sum()
        if ov > best:
            best = ov
    return best

def construct():
    best_h = np.full(N, 0.5)
    best_score = _overlap_max(best_h)
    t0 = time.time()

    # Phase 1: spectral design - binary sequence via Fourier phase control
    # We want a binary sequence (0/1) with sum N/2 that minimizes max overlap.
    # max overlap = S - min_k autocorr(k) = N/2 - min_k autocorr(k)
    # So minimize max overlap <=> maximize min_k autocorr(k)
    # i.e., make the circular autocorrelation as flat/high as possible.

    # Start from a good periodic pattern: alternating gives autocorr pattern.
    # For N even, h = [1,0,1,0,...] sum = N/2. autocorr(k): for even k, N/2; odd k, 0.
    # min autocorr = 0, max overlap = N/2 - 0 = N/2 = 50. Bad.
    #
    # Better: use a sequence whose autocorrelation is as flat as possible (low sidelobe).
    # A "perfect" autocorrelation sequence would have autocorr(k)=0 for k!=0 and N/2 at k=0.
    # But we need real-valued h in [0,1]. Let's use complex Fourier and threshold.

    rng = np.random.default_rng(42)

    # Build a candidate via random Fourier phases, threshold to binary, fix sum.
    def make_candidate(freqs, phases):
        # h_hat[k] = amp[k]*exp(1j*phase[k]), conjugate symmetric for real
        n = N
        h_hat = np.zeros(n, dtype=complex)
        # DC
        h_hat[0] = n / 2.0  # sum = n/2
        for k in range(1, n // 2 + 1):
            if k in freqs:
                a = 1.0
                h_hat[k] = a * np.exp(1j * phases[k])
                h_hat[n - k] = np.conj(h_hat[k])
        h = np.real(np.fft.ifft(h_hat))
        # threshold to binary
        h_bin = (h >= 0.5).astype(float)
        # fix sum to exactly n/2
        target = n // 2
        cur = int(h_bin.sum())
        idx = np.argsort(h)
        if cur > target:
            # flip some 1s to 0s, those with smallest h
            flip = idx[:cur - target]
            h_bin[flip] = 0.0
        elif cur < target:
            flip = idx[-(target - cur):]
            h_bin[flip] = 1.0
        return h_bin

    # Try several spectral candidates
    half = N // 2
    all_freqs = list(range(1, half + 1))
    for trial in range(200):
        if time.time() > DEADLINE:
            break
        # random subset of frequencies
        m = rng.integers(3, half)
        freqs = set(rng.choice(all_freqs, size=m, replace=False).tolist())
        phases = {k: rng.uniform(0, 2 * np.pi) for k in freqs}
        h_c = make_candidate(freqs, phases)
        s = _overlap_max(h_c)
        if s < best_score:
            best_score = s
            best_h = h_c.copy()

    # Phase 2: local search (bit flips) to polish
    def try_flips(h, iters):
        h = h.copy()
        cur = _overlap_max(h)
        n = len(h)
        for it in range(iters):
            if time.time() > DEADLINE:
                break
            improved = False
            # try flipping each bit
            order = rng.permutation(n)
            for i in order:
                if time.time() > DEADLINE:
                    break
                old = h[i]
                new = 1.0 - old
                h[i] = new
                s = _overlap_max(h)
                if s < cur - 1e-12:
                    cur = s
                    improved = True
                else:
                    h[i] = old
            if not improved:
                # try pairs of flips
                pair_improved = False
                for _ in range(200):
                    if time.time() > DEADLINE:
                        break
                    i, j = rng.integers(0, n, 2)
                    if i == j:
                        continue
                    oi, oj = h[i], h[j]
                    h[i] = 1.0 - oi
                    h[j] = 1.0 - oj
                    s = _overlap_max(h)
                    if s < cur - 1e-12:
                        cur = s
                        pair_improved = True
                    else:
                        h[i] = oi
                        h[j] = oj
                if not pair_improved:
                    break
        return h, cur

    best_h, best_score = try_flips(best_h, 30)

    # Phase 3: restart from different spectral seeds if time remains
    for trial in range(40):
        if time.time() > DEADLINE - 5:
            break
        m = rng.integers(3, half)
        freqs = set(rng.choice(all_freqs, size=m, replace=False).tolist())
        phases = {k: rng.uniform(0, 2 * np.pi) for k in freqs}
        h_c = make_candidate(freqs, phases)
        h_c, s = try_flips(h_c, 8)
        if s < best_score:
            best_score = s
            best_h = h_c.copy()

    # Final: ensure sum is exactly N/2
    target = N // 2
    cur = int(round(best_h.sum()))
    if cur != target:
        idx = np.argsort(best_h)
        if cur > target:
            flip = idx[:cur - target]
            best_h[flip] = 0.0
        else:
            flip = idx[-(target - cur):]
            best_h[flip] = 1.0

    # clamp to [0,1]
    best_h = np.clip(best_h, 0.0, 1.0)
    return best_h
