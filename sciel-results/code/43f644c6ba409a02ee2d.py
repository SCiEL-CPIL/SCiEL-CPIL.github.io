"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time


def _autoconv_max_abs(h):
    """Compute max absolute value of linear autoconvolution using FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    H = np.fft.rfft(h, fsize)
    ac = np.fft.irfft(H * H, fsize)
    return np.max(np.abs(ac))


def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10
    return _autoconv_max_abs(h) / (s * s)


def _local_sign_flip(h, max_passes=3, deadline=None):
    """Local optimization: try flipping each element's sign, keep if improves."""
    h = h.copy()
    best_val = _score(h)
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            h[i] = -h[i]
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = -h[i]  # revert
        if not improved:
            break
    return h, best_val


def _barker_like(n):
    """Generate a Barker-like sequence of length n using known Barker codes or random ±1."""
    # Barker codes exist for lengths 2,3,4,5,7,11,13
    barkers = {
        2: [1, 1],
        3: [1, 1, -1],
        4: [1, 1, -1, 1],
        5: [1, 1, -1, -1, 1],
        7: [1, -1, -1, 1, -1, 1, 1],
        11: [1, 1, 1, -1, 1, -1, -1, -1, 1, -1, 1],
        13: [1, 1, 1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1],
    }
    if n in barkers:
        return np.array(barkers[n], dtype=float)
    # For other lengths, use random ±1
    return np.random.choice([-1.0, 1.0], size=n)


def _chirp_sequence(n):
    """Generate a chirp (Zadoff-Chu-like) sequence."""
    x = np.arange(n)
    # Use a quadratic phase
    h = np.sign(np.cos(2 * np.pi * x * (x + 1) / (2 * n)))
    return h


def _m_sequence_like(n, seed=42):
    """Generate an m-sequence-like pattern using a simple LFSR or just structured random."""
    rng = np.random.RandomState(seed)
    # Use a pattern based on powers of 2
    h = np.ones(n)
    for i in range(n):
        # Use bit-reversal based sign
        if (i & 0x5555) != 0:
            h[i] = -1.0
    return h


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill

    best_h = np.ones(N)
    best_val = _score(best_h)

    np.random.seed(42)
    n_restarts = 200

    for r in range(n_restarts):
        if time.time() > deadline:
            break

        # Different initialization strategies
        if r % 7 == 0:
            # Random ±1
            h = np.random.choice([-1.0, 1.0], size=N)
        elif r % 7 == 1:
            # Mostly +1 with some random -1
            h = np.ones(N)
            n_neg = np.random.randint(N // 4, N // 2)
            neg_idx = np.random.choice(N, n_neg, replace=False)
            h[neg_idx] = -1.0
        elif r % 7 == 2:
            # Chirp-like pattern
            h = _chirp_sequence(N)
        elif r % 7 == 3:
            # Random with continuous values
            h = np.random.uniform(-1, 1, size=N)
        elif r % 7 == 4:
            # Block pattern with random block sizes
            h = np.ones(N)
            pos = 0
            while pos < N:
                block_len = np.random.randint(1, N // 10 + 1)
                sign = np.random.choice([-1.0, 1.0])
                h[pos:pos + block_len] = sign
                pos += block_len
        elif r % 7 == 5:
            # M-sequence-like pattern
            h = _m_sequence_like(N, seed=r * 7 + 1)
        else:
            # Barker-like tiling
            h = np.ones(N)
            pos = 0
            while pos < N:
                # Use a small Barker code length
                blen = np.random.choice([3, 4, 5, 7, 11, 13])
                if pos + blen > N:
                    blen = N - pos
                if blen in [3, 4, 5, 7, 11, 13]:
                    code = _barker_like(blen)
                    if np.random.rand() < 0.5:
                        code = -code
                    h[pos:pos + blen] = code
                else:
                    h[pos:pos + blen] = np.random.choice([-1.0, 1.0], size=blen)
                pos += blen

        # Ensure sum is not too small
        s = np.sum(h)
        if abs(s) < N * 0.1:
            if s < 0:
                h = -h
            h = h + 0.3
            h = np.sign(h)

        # Local sign-flip optimization
        h_opt, val_opt = _local_sign_flip(h, max_passes=3, deadline=deadline)

        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt

    return best_h
# EVOLVE-BLOCK-END
