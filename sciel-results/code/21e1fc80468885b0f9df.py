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


def _local_sign_flip(h, max_passes=10, deadline=None):
    """Local optimization: try flipping each element's sign, keep if improves.
    Uses shuffled indices and multi-flip moves for better convergence."""
    h = h.copy()
    best_val = _score(h)
    rng = np.random.RandomState(12345)
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        # Single-flip pass with shuffled order
        indices = rng.permutation(N)
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
        # Multi-flip pass: try flipping pairs of nearby elements
        for _ in range(3):
            if deadline is not None and time.time() > deadline:
                break
            i = rng.randint(0, N - 1)
            j = (i + 1) % N
            h[i] = -h[i]
            h[j] = -h[j]
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = -h[i]
                h[j] = -h[j]  # revert both
        if not improved:
            break
    return h, best_val


def _polish_local_search(h, deadline=None, max_passes=30):
    """High-precision polish: more passes with shuffled order and triple flips."""
    h = h.copy()
    best_val = _score(h)
    rng = np.random.RandomState(99999)
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        indices = rng.permutation(N)
        for i in indices:
            if deadline is not None and time.time() > deadline:
                break
            h[i] = -h[i]
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = -h[i]
        # Triple-flip moves
        for _ in range(5):
            if deadline is not None and time.time() > deadline:
                break
            i = rng.randint(0, N - 2)
            j = (i + 1) % N
            k = (i + 2) % N
            h[i] = -h[i]
            h[j] = -h[j]
            h[k] = -h[k]
            new_val = _score(h)
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                h[i] = -h[i]
                h[j] = -h[j]
                h[k] = -h[k]
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
    n_restarts = 200  # Reduced to allow more passes per restart

    for r in range(n_restarts):
        if time.time() > deadline:
            break

        # Different initialization strategies - emphasize Legendre-based
        if r % 10 in [0, 1, 2, 3]:
            # Legendre sequence variants (40% of restarts)
            if r % 10 == 0:
                h = _legendre_sequence(N)
            elif r % 10 == 1:
                h = _legendre_sequence(N)
                shift = np.random.randint(0, N)
                h = np.roll(h, shift)
            elif r % 10 == 2:
                h = _legendre_sequence(N)
                n_flips = np.random.randint(1, 8)
                flip_idx = np.random.choice(N, n_flips, replace=False)
                h[flip_idx] = -h[flip_idx]
            else:
                h = _legendre_sequence(N)
                # Random block sign flips
                n_blocks = np.random.randint(2, 6)
                for _ in range(n_blocks):
                    bstart = np.random.randint(0, N)
                    blen = np.random.randint(5, 30)
                    bend = min(bstart + blen, N)
                    h[bstart:bend] = -h[bstart:bend]
        elif r % 10 in [4, 5, 6]:
            # Other structured patterns (30%)
            if r % 10 == 4:
                h = np.random.choice([-1.0, 1.0], size=N)
            elif r % 10 == 5:
                h = _chirp_sequence(N)
            else:
                h = np.ones(N)
                pos = 0
                while pos < N:
                    block_len = np.random.randint(1, N // 10 + 1)
                    sign = np.random.choice([-1.0, 1.0])
                    h[pos:pos + block_len] = sign
                    pos += block_len
        else:
            # Mixed patterns (30%)
            if r % 10 == 7:
                h = _m_sequence_like(N, seed=r * 7 + 1)
            elif r % 10 == 8:
                h = np.ones(N)
                pos = 0
                while pos < N:
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
            else:
                h = np.ones(N)
                n_neg = np.random.randint(N // 4, N // 2)
                neg_idx = np.random.choice(N, n_neg, replace=False)
                h[neg_idx] = -1.0

        # Ensure sum is not too small
        s = np.sum(h)
        if abs(s) < N * 0.1:
            if s < 0:
                h = -h
            h = h + 0.3
            h = np.sign(h)

        # Local sign-flip optimization with improved method
        h_opt, val_opt = _local_sign_flip(h, max_passes=10, deadline=deadline)

        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt

    # Final high-precision polish on the best solution found
    if time.time() < deadline - 5.0:
        h_polished, val_polished = _polish_local_search(best_h, deadline=deadline, max_passes=30)
        if val_polished < best_val:
            best_val = val_polished
            best_h = h_polished

    return best_h
# EVOLVE-BLOCK-END
