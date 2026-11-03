"""Signed step function (400 intervals) minimizing the autoconvolution peak (C3 upper bound).
Initialization: Legendre symbol (quadratic residues mod prime) -> low additive energy.
Refinement: fast O(N) delta-update local search within a wall-clock deadline."""
import numpy as np
import time

N = 400
DEADLINE_S = 135.0  # safety margin vs 150s wall-clock kill

# ---------- fast autoconvolution (circular) peak ----------
def _acf_peak(x):
    """Max |circular autocorrelation| over all lags, via FFT (for full scoring)."""
    n = x.size
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f), n=2 * n)[:n]
    return float(np.max(np.abs(acf)))

# ---------- O(N) delta of the peak when flipping one bit ----------
def _delta_peak(x, acf, i, n):
    """Return new peak |acf'| after flipping x[i] (x in {-1,1}), given current acf.
    acf is the linear autocorrelation array (length 2n-1). O(n) update."""
    d = -2.0 * x[i]
    # acf[k] = sum_j x[j]*x[j-k]; flipping x[i] changes terms involving i:
    # new_acf[k] = acf[k] + d*(x[i-k] + x[i+k])  (with zero-padding out of range)
    # We only need the max |.|, but to be safe compute affected lags.
    # Affected lags k where i-k in [0,n-1] or i+k in [0,n-1] -> k in [-i, i] and [i-n+1, i]
    lo = max(-(n - 1), -i)
    hi = min(n - 1, i)
    # also the symmetric side
    lo2 = max(-(n - 1), i - (n - 1))
    hi2 = min(n - 1, i + (n - 1))
    kmin = min(lo, lo2)
    kmax = max(hi, hi2)
    ks = np.arange(kmin, kmax + 1)
    # x at i-k and i+k with zero padding
    a = np.where((ks + i) >= 0, ks + i, -1)
    b = np.where((i - ks) >= 0, i - ks, -1)
    xa = np.where(a >= 0, x[np.clip(a, 0, n - 1)], 0.0)
    xb = np.where(b >= 0, x[np.clip(b, 0, n - 1)], 0.0)
    new = acf[ks + (n - 1)] + d * (xa + xb)
    local_max = float(np.max(np.abs(new)))
    # unaffected lags keep their old values; global peak = max(local_max, rest)
    # rest max = max |acf| over lags not in [kmin,kmax]
    full_idx = np.arange(0, 2 * n - 1)
    mask = (full_idx < kmin + (n - 1)) | (full_idx > kmax + (n - 1))
    rest = float(np.max(np.abs(acf[mask]))) if np.any(mask) else 0.0
    return max(local_max, rest)


def _linear_acf(x):
    n = x.size
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f), n=2 * n)[:2 * n - 1]
    return acf


def _legendre_init(n):
    """Legendre symbol sequence (quadratic residues mod prime) mapped to +-1, length n."""
    # smallest prime >= n
    p = n
    while True:
        prime = True
        for d in range(2, int(p ** 0.5) + 1):
            if p % d == 0:
                prime = False
                break
        if prime:
            break
        p += 1
    x = np.zeros(n, dtype=np.float64)
    for i in range(n):
        r = pow(i, (p - 1) // 2, p)  # Euler criterion: 1 if residue, p-1 if non-residue, 0 if 0
        if r == 1:
            x[i] = 1.0
        elif r == p - 1:
            x[i] = -1.0
        else:
            x[i] = 1.0
    return x


def construct():
    t0 = time.time()
    best = None
    best_val = np.inf

    def _consider(x, val):
        nonlocal best, best_val
        if val < best_val:
            best_val = val
            best = x.copy()

    # ---- initialization: Legendre ----
    x = _legendre_init(N)
    acf = _linear_acf(x)
    _consider(x, _acf_peak(x))

    # ---- local search: greedy single-bit flips with O(N) delta ----
    n = N
    idx = np.arange(n)
    while time.time() - t0 < DEADLINE_S:
        improved = False
        for i in idx:
            if time.time() - t0 > DEADLINE_S - 2.0:
                break
            new_peak = _delta_peak(x, acf, i, n)
            if new_peak < best_val - 1e-12:
                x[i] = -x[i]
                acf = _linear_acf(x)  # refresh (exact) after a beneficial flip
                _consider(x, new_peak)
                improved = True
        if not improved:
            break

    # ---- final polish: a few more exact passes ----
    if time.time() - t0 < DEADLINE_S - 5.0:
        for _ in range(3):
            acf = _linear_acf(x)
            any_imp = False
            for i in idx:
                if time.time() - t0 > DEADLINE_S - 3.0:
                    break
                new_peak = _delta_peak(x, acf, i, n)
                if new_peak < best_val - 1e-12:
                    x[i] = -x[i]
                    acf = _linear_acf(x)
                    _consider(x, new_peak)
                    any_imp = True
            if not any_imp:
                break

    if best is None:
        best = x
    return best
