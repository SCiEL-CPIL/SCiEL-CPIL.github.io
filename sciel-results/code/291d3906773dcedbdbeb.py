import numpy as np
import time

N = 400

def _autoconv(x):
    """Compute full autoconvolution of x via FFT."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    X = np.fft.rfft(x, size)
    ac = np.fft.irfft(X * np.conj(X), size)[:2 * n - 1]
    return ac

def _score(x):
    """Compute max|autoconv| / (sum(x))^2. Lower is better."""
    s = np.sum(x)
    if abs(s) < 1e-12:
        return float('inf')
    ac = _autoconv(x)
    max_ac = np.max(np.abs(ac))
    return max_ac / (s * s)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Precompute FFT size for incremental scoring
    fft_size = 1
    while fft_size < 2 * N - 1:
        fft_size *= 2

    # Phase 1: Try many random ±1 initializations and structured candidates
    rng = np.random.default_rng(42)
    best_x = np.ones(N, dtype=np.float64)
    best_score = _score(best_x)

    # Try structured candidates
    candidates = [
        np.ones(N),
        np.ones(N//2).tolist() + [-1.0]*(N - N//2),
        np.array([1.0 if i % 2 == 0 else -1.0 for i in range(N)]),
        np.array([1.0 if i % 4 < 2 else -1.0 for i in range(N)]),
        np.array([1.0 if i % 8 < 4 else -1.0 for i in range(N)]),
    ]
    for c in candidates:
        c = np.asarray(c, dtype=np.float64)
        sc = _score(c)
        if sc < best_score:
            best_score = sc
            best_x = c.copy()

    # Try many random ±1 candidates
    for seed in range(500):
        if time.time() > deadline - 60:
            break
        x = rng.choice([-1.0, 1.0], size=N)
        sc = _score(x)
        if sc < best_score:
            best_score = sc
            best_x = x.copy()

    # Phase 2: Fast local search using incremental autoconvolution update
    # When we flip x[i] -> -x[i], the change in autoconv is -2*x[i] * (x convolved with delta_i)
    # x convolved with delta_i = x shifted by i (for i < N) and x reversed-shifted (for i >= N)
    # We can compute this via FFT: ifft(fft(x) * fft(delta_i))
    # But more efficiently, we precompute FFT of x and update it incrementally

    x = best_x.copy()
    cur_score = best_score

    # Precompute FFT of x for incremental updates
    X = np.fft.rfft(x, fft_size)

    # Helper to compute autoconv from FFT
    def _ac_from_fft(X):
        ac = np.fft.irfft(X * np.conj(X), fft_size)[:2*N-1]
        return ac

    # Helper to get sum
    def _sum_from_x(x):
        return np.sum(x)

    # For flipping index i: new_X = X - 2*x[i]*exp(-2πi*k*i/fft_size)
    # We can precompute the phase factors
    k = np.arange(fft_size // 2 + 1)
    # phase[i] = exp(-2πi * i / fft_size) for the FFT bin
    # Actually, fft(delta_i) has value 1 at all bins (since delta_i is 1 at position i)
    # Wait: fft of a delta at position i: X[k] = sum_j x[j] exp(-2πi jk/fft_size)
    # For delta_i: x[j] = 1 if j==i else 0, so X[k] = exp(-2πi i k/fft_size)
    # So fft(delta_i) = exp(-2πi * i * k / fft_size) for k = 0..fft_size//2

    # Precompute phase factors for all i
    # phases[i, k] = exp(-2πi * i * k / fft_size)
    # This is a DFT matrix transpose, but we can compute on the fly

    # Actually, for incremental update: when x[i] changes by delta, X changes by delta * exp(-2πi*i*k/fft_size)
    # Flipping x[i] means delta = -2*x[i]
    # So new_X = X - 2*x[i] * exp(-2πi*i*k/fft_size)

    # Let's precompute the exponential factors for efficiency
    # exp_factors[i] = exp(-2πi * i * k / fft_size) for k = 0..fft_size//2
    # This is a (N, fft_size//2+1) matrix - might be large but N=400, fft_size=1024, so 400*513 ~ 200k, fine

    exp_factors = np.exp(-2j * np.pi * np.outer(np.arange(N), k) / fft_size)

    # Fast flip function
    def _flip_and_score(x, i, X):
        """Flip x[i], return new score and updated X."""
        old_val = x[i]
        x[i] = -old_val
        # Update X
        X_new = X - 2.0 * old_val * exp_factors[i]
        # Compute new autoconv
        ac = np.fft.irfft(X_new * np.conj(X_new), fft_size)[:2*N-1]
        max_ac = np.max(np.abs(ac))
        s = np.sum(x)
        if abs(s) < 1e-12:
            score = float('inf')
        else:
            score = max_ac / (s * s)
        # Undo if we don't want to keep
        return score, X_new

    # Local search with incremental scoring
    no_improve = 0
    max_no_improve = 500
    rng2 = np.random.default_rng(999)

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
            x[i] = -x[i]  # keep the flip
            X = X_new
            cur_score = new_score
            no_improve = 0
            if cur_score < best_score:
                best_score = cur_score
                best_x = x.copy()
        else:
            x[i] = old_val  # undo
            no_improve += 1

    # Phase 3: Systematic sweep - try flipping each index in order
    improved = True
    while improved and time.time() < deadline - 3:
        improved = False
        for i in range(N):
            if time.time() > deadline - 3:
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
                x[i] = -x[i]  # keep
                X = X_new
                cur_score = new_score
                improved = True
                if cur_score < best_score:
                    best_score = cur_score
                    best_x = x.copy()
            else:
                x[i] = old_val  # undo

    # Phase 4: Multi-flip search - try flipping pairs of indices
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

    # Final verification with exact score
    final_score = _score(best_x)
    if final_score < best_score:
        best_score = final_score

    return best_x
