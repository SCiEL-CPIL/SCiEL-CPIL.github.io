"""Optimize signed step function (±1, N=400) to minimize peak sidelobe of autocorrelation."""
import time
import numpy as np

N = 400
MAX_TIME = 140.0  # safety margin under 150s kill


def _acf(x):
    """Full linear autocorrelation via FFT, returned as array of length 2N-1 (lags -(N-1)..N-1)."""
    n = len(x)
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f), n=2 * n)
    return acf[:2 * n - 1]


def _peak_sidelobe(x):
    """Max |acf| over non-zero lags."""
    a = _acf(x)
    # a is centered: index N-1 is lag 0
    c = N - 1
    return max(np.max(np.abs(a[:c])), np.max(np.abs(a[c + 1:])))


def _continuous_seed():
    """Flat-spectrum (low L2 autocorrelation) continuous optimization -> binary seed."""
    rng = np.random.default_rng(0)
    best = None
    best_val = np.inf
    t0 = time.time()
    for _ in range(8):
        # random complex init with some structure
        x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
        x /= np.sqrt(np.sum(np.abs(x) ** 2))
        lr = 0.05
        for it in range(600):
            f = np.fft.rfft(x, n=2 * N)
            power = f * np.conj(f)
            # gradient of sum |power|^2 w.r.t. x
            g = np.fft.irfft(2 * power * np.conj(f), n=2 * N)[:N]
            x = x - lr * g
            x /= np.sqrt(np.sum(np.abs(x) ** 2))
            if it % 100 == 0 and time.time() - t0 > MAX_TIME * 0.35:
                break
        # binarize: sign of real part
        b = np.sign(np.real(x))
        b[b == 0] = 1
        val = _peak_sidelobe(b)
        if val < best_val:
            best_val = val
            best = b.copy()
    return best, best_val


def _flip_delta(x, idx):
    """Return the change in peak sidelobe if we flip x[idx]."""
    # cheap: recompute acf
    x2 = x.copy()
    x2[idx] = -x2[idx]
    return _peak_sidelobe(x2)


def _anneal(x, start_temp, end_temp, iters, deadline):
    """Simulated annealing with single-flip moves."""
    current = x.copy()
    current_val = _peak_sidelobe(current)
    best = current.copy()
    best_val = current_val
    rng = np.random.default_rng(1)
    n = len(current)
    for i in range(iters):
        if time.time() > deadline:
            break
        temp = start_temp * (end_temp / start_temp) ** (i / max(1, iters - 1))
        idx = rng.integers(0, n)
        current[idx] = -current[idx]
        new_val = _peak_sidelobe(current)
        delta = new_val - current_val
        if delta < 0 or rng.random() < np.exp(-delta / max(temp, 1e-12)):
            current_val = new_val
            if new_val < best_val:
                best_val = new_val
                best = current.copy()
        else:
            current[idx] = -current[idx]
    return best, best_val


def _greedy_polish(x, deadline):
    """First-improvement single-flip polish."""
    current = x.copy()
    current_val = _peak_sidelobe(current)
    n = len(current)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for idx in range(n):
            if time.time() > deadline:
                break
            current[idx] = -current[idx]
            new_val = _peak_sidelobe(current)
            if new_val < current_val:
                current_val = new_val
                improved = True
            else:
                current[idx] = -current[idx]
    return current, current_val


def construct():
    t0 = time.time()
    deadline = t0 + MAX_TIME

    # Phase 1: continuous seed
    seed, seed_val = _continuous_seed()
    best = seed
    best_val = seed_val

    # Phase 2: greedy polish the seed
    if time.time() < deadline - 5:
        best, best_val = _greedy_polish(best, deadline - 5)

    # Phase 3: annealing restarts
    temp_start = 0.05
    temp_end = 0.005
    while time.time() < deadline - 3:
        iters = 20000
        cand, cand_val = _anneal(best, temp_start, temp_end, iters, deadline - 3)
        if cand_val < best_val:
            best = cand
            best_val = cand_val
            # polish
            best, best_val = _greedy_polish(best, deadline - 3)

    return best


if __name__ == "__main__":
    res = construct()
    print("peak sidelobe:", _peak_sidelobe(res))
