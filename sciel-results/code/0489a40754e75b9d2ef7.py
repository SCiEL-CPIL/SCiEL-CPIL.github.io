"""Design: sign of sum of linear + quadratic phase chirps, polished by sign-flip local search.
Minimizes max_k |c[k]| (autoconvolution peak) for a 400-interval signed step function."""
import numpy as np
import time

N = 400
_DEADLINE = 140.0  # safety margin under 150s wall-clock kill
_T0 = time.time()


def _elapsed():
    return time.time() - _T0


def _autocorr_peak(s):
    """Return max |c[k]| for k != 0, where c = s autoconvolved with itself."""
    fs = np.fft.fft(s, n=2 * N)
    c = np.fft.ifft(fs * np.conj(fs)).real
    c = c[:N]  # lags 0..N-1
    # c[0] = sum s^2 = N; exclude lag 0
    return float(np.max(np.abs(c[1:])))


def _chirp_base(f1, f2, a, b):
    """sign of a*exp(i*2pi*f1*j/N) + b*exp(i*2pi*f2*j^2/N^2)."""
    j = np.arange(N)
    lin = np.exp(2j * np.pi * f1 * j / N)
    quad = np.exp(2j * np.pi * f2 * j * j / (N * N))
    z = a * lin + b * quad
    s = np.sign(z.real)
    s[s == 0] = 1
    return s.astype(np.float64)


def _thue_morse():
    s = np.zeros(N)
    for j in range(N):
        s[j] = 1.0 if (bin(j).count("1") % 2 == 0) else -1.0
    return s


def _local_search(s, max_passes=200, time_budget=30.0):
    """Greedy sign flips reducing max |c[k]| (k!=0). Returns best s and best value."""
    s = s.copy()
    best_val = _autocorr_peak(s)
    best_s = s.copy()
    n = len(s)
    # Precompute autocorr incrementally: flipping s[i] changes c[k] by
    #   -2*s[i]*(s[i+k] + s[i-k]) for the symmetric part. Use full recompute for safety
    # but N=400 is cheap. We recompute peak each candidate flip via FFT (fast).
    order = np.arange(n)
    for p in range(max_passes):
        if _elapsed() > _DEADLINE - 5:
            break
        improved = False
        np.random.shuffle(order)
        for i in order:
            if _elapsed() > _DEADLINE - 5:
                break
            old = s[i]
            s[i] = -old
            val = _autocorr_peak(s)
            if val < best_val - 1e-12:
                best_val = val
                best_s = s.copy()
                improved = True
            else:
                s[i] = old
        if not improved:
            break
    return best_s, best_val


def construct():
    """Return sequence of N signed step heights (in {-1, +1})."""
    candidates = []

    # Chirp bases with varied parameters (linear + quadratic phase, Zadoff-Chu-like)
    chirp_params = [
        (1, 1, 1.0, 1.0),
        (1, 2, 1.0, 1.0),
        (1, 3, 1.0, 1.0),
        (2, 1, 1.0, 1.0),
        (2, 3, 1.0, 1.0),
        (1, 1, 1.0, -1.0),
        (1, 2, 1.0, -1.0),
        (3, 2, 1.0, 1.0),
        (1, 4, 1.0, 1.0),
        (2, 2, 1.0, 1.0),
    ]
    for (f1, f2, a, b) in chirp_params:
        if _elapsed() > _DEADLINE - 10:
            break
        candidates.append(_chirp_base(f1, f2, a, b))

    # Thue-Morse (known low autocorrelation)
    candidates.append(_thue_morse())

    # A couple of random seeds for diversity
    rng = np.random.default_rng(12345)
    for _ in range(3):
        if _elapsed() > _DEADLINE - 10:
            break
        candidates.append(rng.choice([-1.0, 1.0], size=N))

    # Evaluate each candidate, then local-search the best few.
    scored = []
    for s in candidates:
        if _elapsed() > _DEADLINE - 10:
            break
        val = _autocorr_peak(s)
        scored.append((val, s))
    scored.sort(key=lambda x: x[0])

    best_s = scored[0][1]
    best_val = scored[0][0]

    # Local-search the top candidates within remaining time.
    top_k = min(5, len(scored))
    for val0, s0 in scored[:top_k]:
        if _elapsed() > _DEADLINE - 5:
            break
        s_best, v_best = _local_search(s0, max_passes=1000)
        if v_best < best_val:
            best_val = v_best
            best_s = s_best

    return best_s


# Self-test (not used by evaluator, but keeps module importable cleanly)
if __name__ == "__main__":
    s = construct()
    print("N =", len(s))
    print("peak =", _autocorr_peak(s))
