import numpy as np
import time
from scipy.optimize import minimize

N = 400

def _autoconv(x):
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    X = np.fft.rfft(x, size)
    ac = np.fft.irfft(X * np.conj(X), size)[:2 * n - 1]
    return ac

def _score(x):
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    ac = _autoconv(x)
    max_ac = np.max(np.abs(ac))
    return max_ac / (s * s)

def _is_prime(m):
    if m < 2:
        return False
    if m == 2:
        return True
    if m % 2 == 0:
        return False
    for i in range(3, int(np.sqrt(m)) + 1, 2):
        if m % i == 0:
            return False
    return True

def _legendre_seq(n, offset=0):
    """Legendre symbol sequence: (x/p) for smallest prime p >= n."""
    p = n + offset
    while not _is_prime(p):
        p += 1
    half = (p - 1) // 2
    h = np.zeros(n)
    for x in range(n):
        val = (x + offset) % p
        if val == 0:
            h[x] = 1.0
        else:
            r = pow(val, half, p)
            h[x] = 1.0 if r == 1 else -1.0
    return h

def _blackman_harris_window(n):
    """4-term Blackman-Harris window."""
    x = np.arange(n)
    c1, c2, c3 = 0.4112348, 0.2774046, 0.0865420
    w = (c1 - c2 * np.cos(2 * np.pi * x / (n - 1)) +
         c3 * np.cos(4 * np.pi * x / (n - 1)) -
         0.0088590 * np.cos(6 * np.pi * x / (n - 1)))
    return w

def _spectral_window_seed(n, window='bh'):
    """Create a signed sequence from a windowed-sinc FIR design.
    The idea: design a lowpass filter with sharp cutoff, take its real
    impulse response, and sign-quantize. The window shapes the spectral
    sidelobes, structurally suppressing peak autoconvolution."""
    if window == 'bh':
        w = _blackman_harris_window(n)
    elif window == 'hamming':
        w = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n) / (n - 1))
    else:
        w = np.ones(n)
    
    # Create a bandpass-like pattern: high in middle, low at edges
    # Then sign-quantize
    center = n / 2.0
    sigma = n / 4.0
    base = np.exp(-(np.arange(n) - center) ** 2 / (2 * sigma ** 2))
    h = base * w
    # Sign quantize
    h_sign = np.where(h >= 0, 1.0, -1.0)
    # Also try the negative
    return h_sign

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_x = np.ones(N, dtype=np.float64)
    best_score = _score(best_x)

    # Phase 1: Generate diverse seeds from multiple families
    seed_list = []

    # Family 1: Legendre sequences (number-theoretic, bounded autocorrelation)
    for offset in range(0, 20):
        if time.time() > deadline - 100:
            break
        h = _legendre_seq(N, offset)
        sc = _score(h)
        seed_list.append((sc, h.copy()))
        if sc < best_score:
            best_score = sc
            best_x = h.copy()
        # Also try negation
        h_neg = -h
        sc_neg = _score(h_neg)
        seed_list.append((sc_neg, h_neg.copy()))
        if sc_neg < best_score:
            best_score = sc_neg
            best_x = h_neg.copy()

    # Family 2: Spectral window seeds
    for window in ['bh', 'hamming']:
        if time.time() > deadline - 100:
            break
        h = _spectral_window_seed(N, window)
        sc = _score(h)
        seed_list.append((sc, h.copy()))
        if sc < best_score:
            best_score = sc
            best_x = h.copy()
        h_neg = -h
        sc_neg = _score(h_neg)
        seed_list.append((sc_neg, h_neg.copy()))
        if sc_neg < best_score:
            best_score = sc_neg
            best_x = h_neg.copy()

    # Family 3: Structured periodic patterns
    patterns = [
        np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]),
        np.array([1.0 if i % 4 < 2 else -1.0 for i in range(N)]),
        np.array([1.0 if i % 8 < 4 else -1.0 for i in range(N)]),
        np.ones(N//2).tolist() + [-1.0]*(N - N//2),
    ]
    for p in patterns:
        p = np.asarray(p, dtype=np.float64)
        sc = _score(p)
        seed_list.append((sc, p.copy()))
        if sc < best_score:
            best_score = sc
            best_x = p.copy()

    # Family 4: Random binary seeds
    rng = np.random.default_rng(42)
    for seed_idx in range(150):
        if time.time() > deadline - 90:
            break
        x = rng.choice([-1.0, 1.0], size=N)
        sc = _score(x)
        seed_list.append((sc, x.copy()))
        if sc < best_score:
            best_score = sc
            best_x = x.copy()

    # Sort seeds by score
    seed_list.sort(key=lambda t: t[0])

    # Phase 2: Fast binary local search using incremental FFT updates
    # For each of the top seeds, run local search
    fft_size = 1
    while fft_size < 2 * N - 1:
        fft_size *= 2

    k = np.arange(fft_size // 2 + 1)
    exp_factors = np.exp(-2j * np.pi * np.outer(np.arange(N), k) / fft_size)

    top_seeds = seed_list[:15]
    
    for seed_score, seed_x in top_seeds:
        if time.time() > deadline - 15:
            break

        x = seed_x.copy()
        cur_score = _score(x)
        X = np.fft.rfft(x, fft_size)

        # Random single-flip local search
        rng2 = np.random.default_rng(hash(tuple(x[:10])) % 100000)
        no_improve = 0
        max_no_improve = 500

        while time.time() < deadline - 10 and no_improve < max_no_improve:
            i = rng2.integers(0, N)
            old_val = x[i]
            x[i] = -old_val
            X_new = X - 2.0 * old_val * exp_factors[i]
            ac = np.fft.irfft(X_new * np.conj(X_new), fft_size)[:2*N-1]
            max_ac = np.max(np.abs(ac))
            s = np.sum(x)
            if abs(s) < 1e-12:
                new_score = float('inf')
            else:
                new_score = max_ac / (s * s)

            if new_score < cur_score:
                X = X_new
                cur_score = new_score
                no_improve = 0
                if cur_score < best_score:
                    best_score = cur_score
                    best_x = x.copy()
            else:
                x[i] = old_val
                no_improve += 1

        # Systematic sweep - try flipping each index in order
        improved = True
        while improved and time.time() < deadline - 8:
            improved = False
            for i in range(N):
                if time.time() > deadline - 8:
                    break
                old_val = x[i]
                x[i] = -old_val
                X_new = X - 2.0 * old_val * exp_factors[i]
                ac = np.fft.irfft(X_new * np.conj(X_new), fft_size)[:2*N-1]
                max_ac = np.max(np.abs(ac))
                s = np.sum(x)
                if abs(s) < 1e-12:
                    new_score = float('inf')
                else:
                    new_score = max_ac / (s * s)

                if new_score < cur_score:
                    X = X_new
                    cur_score = new_score
                    improved = True
                    if cur_score < best_score:
                        best_score = cur_score
                        best_x = x.copy()
                else:
                    x[i] = old_val

        # Multi-flip search (2-flip)
        if time.time() < deadline - 5:
            x = best_x.copy()
            cur_score = best_score
            X = np.fft.rfft(x, fft_size)
            rng3 = np.random.default_rng(1234)
            no_improve = 0
            max_no_improve = 200
            while time.time() < deadline - 3 and no_improve < max_no_improve:
                i1 = rng3.integers(0, N)
                i2 = rng3.integers(0, N)
                if i1 == i2:
                    continue
                v1, v2 = x[i1], x[i2]
                x[i1] = -v1
                x[i2] = -v2
                X_new = X - 2.0 * v1 * exp_factors[i1] - 2.0 * v2 * exp_factors[i2]
                ac = np.fft.irfft(X_new * np.conj(X_new), fft_size)[:2*N-1]
                max_ac = np.max(np.abs(ac))
                s = np.sum(x)
                if abs(s) < 1e-12:
                    new_score = float('inf')
                else:
                    new_score = max_ac / (s * s)
                if new_score < cur_score:
                    X = X_new
                    cur_score = new_score
                    no_improve = 0
                    if cur_score < best_score:
                        best_score = cur_score
                        best_x = x.copy()
                else:
                    x[i1] = v1
                    x[i2] = v2
                    no_improve += 1

    # Final verification
    final_score = _score(best_x)
    if final_score < best_score:
        best_score = final_score

    return best_x
