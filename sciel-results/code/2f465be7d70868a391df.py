"""Rewritten program: optimize signed step heights to minimize autoconvolution upper bound."""
import time
import numpy as np

N = 400
DEADLINE = 135.0  # safety margin before 150s kill

def _autocorr(s):
    """Circular autocorrelation of sign sequence s via FFT."""
    F = np.fft.fft(s)
    ac = np.fft.ifft(F * np.conj(F)).real
    return ac

def _metric(ac):
    """Upper bound on C3: max absolute off-center autocorrelation."""
    # Exclude lag 0 (the peak). Use lags 1..N-1.
    return float(np.max(np.abs(ac[1:])))

def _score(s):
    ac = _autocorr(s)
    return _metric(ac)

def construct():
    start = time.time()
    rng = np.random.default_rng(12345)

    best_s = None
    best_val = float('inf')

    def update(s, val):
        nonlocal best_s, best_val
        if val < best_val:
            best_val = val
            best_s = s.copy()

    # ---- Phase 1: multiple random restarts with greedy single-flip descent ----
    # Budget: try as many restarts as time allows; each restart does greedy descent.
    max_restarts = 200
    for r in range(max_restarts):
        if time.time() - start > DEADLINE - 5.0:
            break
        s = rng.choice([-1.0, 1.0], size=N)
        val = _score(s)
        update(s, val)

        # Greedy single-flip descent
        improved = True
        while improved and (time.time() - start) < DEADLINE - 5.0:
            improved = False
            ac = _autocorr(s)
            cur = _metric(ac)
            # Try each flip; pick best improving flip
            best_gain = 1e-9
            best_i = -1
            for i in range(N):
                s[i] = -s[i]
                ac2 = _autocorr(s)
                v2 = _metric(ac2)
                s[i] = -s[i]
                gain = cur - v2
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
            if best_i >= 0:
                s[best_i] = -s[best_i]
                val = _score(s)
                update(s, val)
                improved = True

    # ---- Phase 2: block flips to escape local minima from best_s ----
    if best_s is not None:
        s = best_s.copy()
        val = _score(s)
        update(s, val)
        block_sizes = [4, 8, 16, 32]
        while (time.time() - start) < DEADLINE - 2.0:
            made = False
            for bs in block_sizes:
                if (time.time() - start) > DEADLINE - 2.0:
                    break
                # Flip a random contiguous block of size bs
                start_i = rng.integers(0, N)
                idx = np.arange(start_i, start_i + bs) % N
                s[idx] = -s[idx]
                v2 = _score(s)
                if v2 < val:
                    val = v2
                    update(s, val)
                    made = True
                else:
                    s[idx] = -s[idx]  # revert
            if not made and val < 0.001:
                # converged tightly; do a few more random perturbations
                for _ in range(50):
                    if (time.time() - start) > DEADLINE - 2.0:
                        break
                    i = rng.integers(0, N)
                    s[i] = -s[i]
                    v2 = _score(s)
                    if v2 < val:
                        val = v2
                        update(s, val)
                    else:
                        s[i] = -s[i]
                break

    # ---- Phase 3: final greedy polish on best_s ----
    if best_s is not None:
        s = best_s.copy()
        improved = True
        while improved and (time.time() - start) < DEADLINE - 0.5:
            improved = False
            ac = _autocorr(s)
            cur = _metric(ac)
            best_gain = 1e-9
            best_i = -1
            for i in range(N):
                s[i] = -s[i]
                ac2 = _autocorr(s)
                v2 = _metric(ac2)
                s[i] = -s[i]
                gain = cur - v2
                if gain > best_gain:
                    best_gain = gain
                    best_i = i
            if best_i >= 0:
                s[best_i] = -s[best_i]
                update(s, _score(s))
                improved = True

    if best_s is None:
        best_s = np.ones(N)

    # Deadline guard: ensure we return well before kill
    if time.time() - start > DEADLINE:
        return best_s

    return best_s
