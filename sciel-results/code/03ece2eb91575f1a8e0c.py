"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _autoconv(x):
    """Compute full linear autoconvolution of x with itself."""
    n = len(x)
    size = 2 * n
    X = np.fft.rfft(x, n=size)
    ac = np.fft.irfft(X * np.conj(X), n=size)[:2 * n - 1]
    return ac

def _score(x):
    """Compute max|autoconv| / (sum(x))^2. Returns inf if sum(x) is 0."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    ac = _autoconv(x)
    return np.max(np.abs(ac)) / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    # Precompute FFT size
    fft_size = 1
    while fft_size < 2 * N - 1:
        fft_size *= 2

    # Precompute phase factors for incremental FFT updates
    k = np.arange(fft_size // 2 + 1)
    exp_factors = np.exp(-2j * np.pi * np.outer(np.arange(N), k) / fft_size)

    def _score_from_fft(X, s):
        """Compute score from FFT and sum."""
        if abs(s) < 1e-12:
            return float('inf')
        ac = np.fft.irfft(X * np.conj(X), fft_size)[:2 * N - 1]
        return np.max(np.abs(ac)) / (s * s)

    # Phase 1: Try structured and random candidates
    rng = np.random.default_rng(42)
    best_x = np.ones(N, dtype=np.float64)
    best_score = _score(best_x)

    # Structured candidates
    candidates = [
        np.ones(N),
        np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]),
        np.array([1.0 if i % 4 < 2 else -1.0 for i in range(N)]),
        np.array([1.0 if i % 8 < 4 else -1.0 for i in range(N)]),
        np.ones(N//2) + np.ones(N - N//2) * (-1.0),
    ]
    for c in candidates:
        c = np.asarray(c, dtype=np.float64)
        sc = _score(c)
        if sc < best_score:
            best_score = sc
            best_x = c.copy()

    # Random ±1 candidates
    for seed in range(300):
        if time.time() > deadline - 60:
            break
        x = rng.choice([-1.0, 1.0], size=N)
        sc = _score(x)
        if sc < best_score:
            best_score = sc
            best_x = x.copy()

    # Phase 2: Fast local search with incremental FFT
    x = best_x.copy()
    cur_score = best_score
    X = np.fft.rfft(x, fft_size)
    cur_sum = np.sum(x)

    rng2 = np.random.default_rng(999)
    no_improve = 0
    max_no_improve = 300

    while time.time() < deadline - 15 and no_improve < max_no_improve:
        idx = rng2.integers(0, N)
        old_val = x[idx]
        new_val = -old_val
        delta = new_val - old_val  # -2 * old_val

        # Update FFT incrementally
        X_new = X + delta * exp_factors[idx]
        new_sum = cur_sum + delta

        new_score = _score_from_fft(X_new, new_sum)

        if new_score < cur_score:
            x[idx] = new_val
            X = X_new
            cur_sum = new_sum
            cur_score = new_score
            no_improve = 0
            if cur_score < best_score:
                best_score = cur_score
                best_x = x.copy()
        else:
            no_improve += 1

    # Phase 3: Systematic sweep for final refinement
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        for idx in range(N):
            if time.time() > deadline - 3:
                break
            old_val = x[idx]
            new_val = -old_val
            delta = new_val - old_val

            X_new = X + delta * exp_factors[idx]
            new_sum = cur_sum + delta
            new_score = _score_from_fft(X_new, new_sum)

            if new_score < cur_score:
                x[idx] = new_val
                X = X_new
                cur_sum = new_sum
                cur_score = new_score
                improved = True
                if cur_score < best_score:
                    best_score = cur_score
                    best_x = x.copy()

    # Phase 4: Try small continuous perturbations
    if time.time() < deadline - 5:
        x = best_x.copy()
        cur_score = best_score
        X = np.fft.rfft(x, fft_size)
        cur_sum = np.sum(x)
        rng3 = np.random.default_rng(1234)

        for _ in range(200):
            if time.time() > deadline - 5:
                break
            idx = rng3.integers(0, N)
            perturbation = rng3.standard_normal() * 0.3
            old_val = x[idx]
            new_val = old_val + perturbation
            delta = new_val - old_val

            X_new = X + delta * exp_factors[idx]
            new_sum = cur_sum + delta
            new_score = _score_from_fft(X_new, new_sum)

            if new_score < cur_score:
                x[idx] = new_val
                X = X_new
                cur_sum = new_sum
                cur_score = new_score
                if cur_score < best_score:
                    best_score = cur_score
                    best_x = x.copy()

    return best_x
# EVOLVE-BLOCK-END
