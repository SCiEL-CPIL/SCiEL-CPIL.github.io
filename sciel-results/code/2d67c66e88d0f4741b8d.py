"""Search for a low-autocorrelation signed step function (400 equal intervals).

Uses Legendre symbol sequences as principled initializations combined with
fast incremental O(N) flip updates, simulated annealing, and local-search polish.
"""
import numpy as np
import time

N = 400
_FFT_SIZE = 1
while _FFT_SIZE < 2 * N:
    _FFT_SIZE *= 2


def _peak_sidelobe(h):
    """Peak aperiodic autocorrelation sidelobe: max_{k>=1} |sum_i h[i] h[i+k]|."""
    H = np.fft.fft(h, _FFT_SIZE)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return float(np.max(np.abs(ac[1:])))


def _compute_acf(h):
    """Compute aperiodic autocorrelation via FFT, return ac[0..N-1]."""
    H = np.fft.fft(h, _FFT_SIZE)
    ac = np.fft.ifft(H * np.conj(H)).real[:N]
    return ac


def _legendre_symbol(n, p):
    """Compute Legendre symbol (n/p) for prime p using pow."""
    if n % p == 0:
        return 0
    return 1 if pow(n, (p - 1) // 2, p) == 1 else -1


def _legendre_sequence(p, length=N):
    """Generate Legendre symbol sequence of given length for prime p.
    For p-1 >= length: use chi(1..length).
    For p-1 < length: use chi(1..p-1) then pad with chi(p..)."""
    seq = np.zeros(length)
    for i in range(length):
        n = i + 1
        ls = _legendre_symbol(n, p)
        seq[i] = 1.0 if ls >= 0 else -1.0
    return seq


def _flip_single(h, ac, i):
    """Flip h[i] and update ac in place. O(N)."""
    val = h[i]
    if i + 1 < N:
        ac[1:N - i] -= 2.0 * val * h[i + 1:N]
    if i > 0:
        ac[1:i + 1] -= 2.0 * val * h[i - 1:-1:-1]
    h[i] = -val


def _flip_block(h, ac, i, L):
    """Flip h[i:i+L] in place and recompute ac via FFT."""
    h[i:i + L] *= -1.0
    ac[:] = _compute_acf(h)


def _anneal(seed, time_budget, init_h=None):
    """Simulated annealing with single-flip and block-flip moves."""
    rng = np.random.default_rng(seed)
    if init_h is not None:
        h = init_h.copy()
    else:
        h = rng.choice([-1.0, 1.0], size=N)

    ac = _compute_acf(h)
    cur = float(np.max(np.abs(ac[1:])))
    best_h = h.copy()
    best = cur
    T = 2.0
    T_min = 1e-8
    start = time.time()

    while time.time() - start < time_budget:
        r = rng.random()
        if r < 0.85:
            i = int(rng.integers(0, N))
            _flip_single(h, ac, i)
            new_peak = float(np.max(np.abs(ac[1:])))
            d = new_peak - cur
            if d <= 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
                cur = new_peak
                if cur < best:
                    best = cur
                    best_h = h.copy()
            else:
                _flip_single(h, ac, i)
        else:
            L = int(rng.integers(2, 9))
            i = int(rng.integers(0, N - L + 1))
            _flip_block(h, ac, i, L)
            new_peak = float(np.max(np.abs(ac[1:])))
            d = new_peak - cur
            if d <= 0 or rng.random() < np.exp(-d / max(T, 1e-9)):
                cur = new_peak
                if cur < best:
                    best = cur
                    best_h = h.copy()
            else:
                _flip_block(h, ac, i, L)
        T = max(T_min, T * 0.99995)

    return best_h, best


def _local_search(h, time_budget, start_time):
    """Iterated local search: repeatedly find best single flip, apply it, repeat."""
    h = h.copy()
    ac = _compute_acf(h)
    cur = float(np.max(np.abs(ac[1:])))
    best = cur
    best_h = h.copy()

    while time.time() - start_time < time_budget:
        best_i = -1
        best_c = cur

        for i in range(N):
            _flip_single(h, ac, i)
            c = float(np.max(np.abs(ac[1:])))
            if c < best_c:
                best_c = c
                best_i = i
            _flip_single(h, ac, i)

        if best_i < 0:
            break

        _flip_single(h, ac, best_i)
        cur = best_c
        if cur < best:
            best = cur
            best_h = h.copy()

    return best_h, best


def _perturbed_local_search(h, time_budget, start_time, rng):
    """Local search with random perturbations to escape local minima."""
    h = h.copy()
    ac = _compute_acf(h)
    cur = float(np.max(np.abs(ac[1:])))
    best = cur
    best_h = h.copy()

    while time.time() - start_time < time_budget:
        # Apply k random flips
        k = int(rng.integers(1, 6))
        for _ in range(k):
            i = int(rng.integers(0, N))
            _flip_single(h, ac, i)

        # Run local search
        h_ls, c_ls = _local_search(h, min(2.0, time_budget - (time.time() - start_time)), time.time())
        if c_ls < best:
            best = c_ls
            best_h = h_ls
            h = h_ls.copy()
            ac = _compute_acf(h)
            cur = best
        else:
            # Revert to best
            h = best_h.copy()
            ac = _compute_acf(h)
            cur = best

    return best_h, best


def construct():
    try:
        hard_deadline = 140.0
        start = time.time()

        # Start with ones as fallback
        global_best_h = np.ones(N)
        global_best = _peak_sidelobe(global_best_h)

        # Build list of initializations
        inits = []

        # Legendre sequences for primes near 400
        for p in [401, 397, 409, 389, 419]:
            seq = _legendre_sequence(p)
            inits.append((f'legendre_{p}', seq))

        # Thue-Morse
        tm = np.zeros(N)
        tm[0] = 1.0
        for i in range(1, N):
            tm[i] = tm[i >> 1] * (1.0 if (i & 1) == 0 else -1.0)
        inits.append(('thue_morse', tm))

        # Random inits
        for s in [42, 12345, 99999]:
            rng = np.random.default_rng(s)
            inits.append((f'rand_{s}', rng.choice([-1.0, 1.0], size=N)))

        # Alternating
        alt = np.ones(N)
        alt[1::2] = -1.0
        inits.append(('alt', alt))

        # Phase 1: Quick local search from each init (up to 3s each)
        for name, init_h in inits:
            elapsed = time.time() - start
            remaining = hard_deadline - elapsed - 15.0
            if remaining < 1.0:
                break
            sub = min(3.0, remaining)
            h, c = _local_search(init_h, sub, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h

        # Phase 2: Annealing from best (up to 30s)
        elapsed = time.time() - start
        remaining = hard_deadline - elapsed - 10.0
        if remaining > 3.0:
            h, c = _anneal(777777, min(30.0, remaining), init_h=global_best_h)
            if c < global_best:
                global_best = c
                global_best_h = h

        # Phase 3: Perturbed local search (up to 40s)
        elapsed = time.time() - start
        remaining = hard_deadline - elapsed - 5.0
        if remaining > 3.0:
            rng = np.random.default_rng(123456)
            h, c = _perturbed_local_search(global_best_h, min(40.0, remaining), time.time(), rng)
            if c < global_best:
                global_best = c
                global_best_h = h

        # Phase 4: Final local search polish
        elapsed = time.time() - start
        remaining = hard_deadline - elapsed
        if remaining > 2.0:
            h, c = _local_search(global_best_h, remaining - 0.5, time.time())
            if c < global_best:
                global_best = c
                global_best_h = h

        return global_best_h
    except Exception:
        return np.ones(N)
