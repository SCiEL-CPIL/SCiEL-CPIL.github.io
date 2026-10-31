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


def _local_sign_flip(h, max_passes=20, deadline=None):
    """Local optimization: try flipping each element's sign, keep if improves."""
    h = h.copy()
    best_val = _score(h)
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        # Shuffle indices to avoid bias
        indices = np.random.permutation(N)
        for i in indices:
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


def _legendre_symbol(n, p):
    """Compute Legendre symbol (n/p) for prime p."""
    if n % p == 0:
        return 0
    return pow(n, (p - 1) // 2, p) - 1 if pow(n, (p - 1) // 2, p) == 1 else -1


def _legendre_sequence(n, p=None):
    """Generate a Legendre sequence of length n using prime p."""
    if p is None:
        # Find a prime >= n
        p = n
        while True:
            is_prime = True
            for i in range(2, int(np.sqrt(p)) + 1):
                if p % i == 0:
                    is_prime = False
                    break
            if is_prime:
                break
            p += 1
    h = np.zeros(n)
    for i in range(n):
        sym = _legendre_symbol(i + 1, p)
        if sym == 0:
            h[i] = 1.0  # Convention for 0
        else:
            h[i] = float(sym)
    return h


def _chirp_sequence(n):
    """Generate a chirp (Zadoff-Chu-like) sequence."""
    x = np.arange(n)
    h = np.sign(np.cos(2 * np.pi * x * (x + 1) / (2 * n)))
    return h


def _barker_like(n):
    """Generate a Barker-like sequence of length n using known Barker codes or random ±1."""
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
    return np.random.choice([-1.0, 1.0], size=n)


def _m_sequence_like(n, seed=42):
    """Generate an m-sequence-like pattern using a simple LFSR or just structured random."""
    rng = np.random.RandomState(seed)
    h = np.ones(n)
    for i in range(n):
        if (i & 0x5555) != 0:
            h[i] = -1.0
    return h


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 150s kill

    best_h = np.ones(N)
    best_val = _score(best_h)

    np.random.seed(42)
    n_restarts = 150 # Reduced restarts to allow more passes per restart

    for r in range(n_restarts):
        if time.time() > deadline:
            break

        # Different initialization strategies
        if r % 5 == 0:
            # Legendre sequence
            h = _legendre_sequence(N)
            shift = np.random.randint(0, N)
            h = np.roll(h, shift)
        elif r % 5 == 1:
            # Random ±1
            h = np.random.choice([-1.0, 1.0], size=N)
        elif r % 5 == 2:
            # Chirp-like
            h = _chirp_sequence(N)
        elif r % 5 == 3:
            # Block pattern
            h = np.ones(N)
            pos = 0
            while pos < N:
                block_len = np.random.randint(2, N // 8 + 1)
                sign = np.random.choice([-1.0, 1.0])
                h[pos:pos + block_len] = sign
                pos += block_len
        else:
            # Random Gaussian signs
            h = np.sign(np.random.randn(N))
            # Ensure non-zero
            h[h == 0] = 1.0

        # Ensure sum is not too small
        s = np.sum(h)
        if abs(s) < N * 0.05:
            if s < 0:
                h = -h
            # Add a small bias to increase sum
            h = h + 0.2
            h = np.sign(h)

        # Local sign-flip optimization with more passes
        h_opt, val_opt = _local_sign_flip(h, max_passes=20, deadline=deadline)

        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt

    return best_h
# EVOLVE-BLOCK-END
