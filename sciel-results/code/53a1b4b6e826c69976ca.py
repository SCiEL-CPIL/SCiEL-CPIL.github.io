import numpy as np
import time

N = 400
_NFFT = 1 << (2 * N - 1).bit_length()  # next power of 2 >= 2N-1 for fast FFT


def _autocorr_fast(h):
    # linear one-sided autocorrelation ac[k] = sum_i h[i] h[i+k], k = 0..N-1
    # Use real FFT trick: autocorr of real signal = IFFT(|FFT|^2)
    F = np.fft.rfft(h, n=_NFFT)
    ac = np.fft.irfft(F * np.conj(F), n=_NFFT)[:N]
    return ac


def _autocorr(h):
    return _autocorr_fast(h)


def _hard_score(h):
    # the actual metric: peak autoconvolution concentration (lower is better)
    ac = _autocorr(h)
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    return float(np.max(np.abs(ac)) / dc)


def _smooth_score_from_ac(ac, beta):
    """Smooth surrogate computed from already-computed autocorrelation."""
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    y = (ac[1:] / dc) ** 2
    m = float(np.max(y))
    z = beta * (y - m)
    return m + float(np.log(np.sum(np.exp(z)))) / beta


def _hard_score_from_ac(ac):
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    return float(np.max(np.abs(ac)) / dc)


def _ising_init(rng):
    """
    Generate initial sign sequence via 1D Ising model with long-range power-law
    interactions derived from the inverse Fourier transform of a target
    flat-spectrum autocorrelation profile.
    
    The idea: we want ac[k]/ac[0] ≈ 0 for k >= 1. The autocorrelation in
    frequency domain is |H(ω)|². For a flat target (white noise), we want
    |H(ω)|² ≈ constant, which means H(ω) should be flat. This is achieved
    by a sequence with no low-frequency bias.
    
    We use an Ising model with couplings J(i,j) ∝ (-1)^(|i-j|) / |i-j|^α
    which encourages alternating patterns with long-range coherence.
    """
    alpha = 0.7  # power-law exponent
    n_sites = N
    # Build effective field from power-law couplings
    # For efficiency, use a truncated interaction range
    max_range = min(200, n_sites // 2)
    eff_field = np.zeros(n_sites)
    
    # Precompute coupling strengths
    distances = np.arange(1, max_range + 1)
    J = (-1.0) ** distances / (distances ** alpha)
    
    # Compute effective field via convolution-like operation
    # For each site i, sum over j != i: J(|i-j|) * s_j
    # Initialize with random signs
    s = rng.choice(np.array([-1.0, 1.0]), size=n_sites)
    
    # Several sweeps of mean-field update
    for sweep in range(50):
        new_s = np.zeros(n_sites)
        for i in range(n_sites):
            # Compute local field from all other sites within max_range
            lo = max(0, i - max_range)
            hi = min(n_sites, i + max_range + 1)
            dists = np.abs(np.arange(lo, hi) - i)
            mask = dists > 0
            if np.any(mask):
                d = dists[mask]
                js = (-1.0) ** d / (d ** alpha)
                local_field = np.sum(js * s[lo:hi][mask])
            else:
                local_field = 0.0
            # Mean-field update with some noise
            if sweep < 10:  # first 10 sweeps: deterministic
                new_s[i] = 1.0 if local_field >= 0 else -1.0
            else:  # later sweeps: add noise for exploration
                prob = 1.0 / (1.0 + np.exp(-2.0 * local_field))
                new_s[i] = 1.0 if rng.random() < prob else -1.0
        s = new_s
    
    return s


def _quadratic_phase_init(freqs):
    """Generate sign sequence from quadratic phase chirp."""
    idx = np.arange(N)
    phases = 2 * np.pi * freqs * idx ** 2 / (2 * N)
    return np.sign(np.cos(phases))


def construct():
    try:
        t0 = time.time()
        deadline = t0 + 140.0  # safety margin inside the 150s limit
        rng = np.random.default_rng(20240517)

        # ---------- diverse initializations ----------
        idx = np.arange(N)
        inits = [
            np.ones(N),
            rng.choice(np.array([-1.0, 1.0]), size=N),
            np.cos(2 * np.pi * idx ** 2 / (2 * N)),
            np.cos(2 * np.pi * idx ** 2 / (N + 1)),
            np.cos(2 * np.pi * idx ** 2 / (3 * N)),
            rng.choice(np.array([-1.0, 1.0]), size=N),
            np.cos(2 * np.pi * idx / N),
            np.sin(2 * np.pi * idx / N),
            np.cos(2 * np.pi * 2 * idx / N),
            rng.choice(np.array([-1.0, 1.0]), size=N),
            # New: Ising model with power-law interactions
            _ising_init(rng),
            # New: quadratic phase chirps with various frequencies
            _quadratic_phase_init(1.0),
            _quadratic_phase_init(2.0),
            _quadratic_phase_init(3.0),
            # New: alternating blocks of varying sizes
            np.sign(np.cos(2 * np.pi * idx / 8)),
            np.sign(np.cos(2 * np.pi * idx / 16)),
            np.sign(np.cos(2 * np.pi * idx / 32)),
            # New: random block flips
            _random_blocks(rng, block_sizes=[4, 8, 16, 32]),
            _random_blocks(rng, block_sizes=[2, 4, 8]),
            _random_blocks(rng, block_sizes=[1, 2, 4, 8, 16]),
        ]
        best_h = None
        best_hard = 1e9
        for h in inits:
            s = _hard_score(h)
            if s < best_hard:
                best_hard = s
                best_h = h.copy()

        h = best_h.copy()
        cur_ac = _autocorr_fast(h)
        cur_hard = _hard_score_from_ac(cur_ac)
        beta = 30.0
        cur_smooth = _smooth_score_from_ac(cur_ac, beta)

        # SA with single autocorr computation per step
        T = 0.02
        T_min = 1e-4
        decay = 0.9997
        sizes = np.array([1, 2, 3, 4, 8, 12, 16, 20, 24, 32, 48, 64])

        # Adaptive temperature and move sizes
        no_improve_count = 0
        max_no_improve = 500

        while time.time() < deadline - 20.0:
            if rng.random() < 0.60:
                size = int(rng.choice(sizes))
                start = int(rng.integers(0, N - size + 1))
                sl = slice(start, start + size)
                old = h[sl].copy()
                h[sl] *= -1.0
                new_ac = _autocorr_fast(h)
                new_hard = _hard_score_from_ac(new_ac)
                new_smooth = _smooth_score_from_ac(new_ac, beta)
                dS = new_smooth - cur_smooth
                if new_hard < cur_hard or (dS <= 0 or rng.random() < np.exp(-dS / T)):
                    cur_hard = new_hard
                    cur_smooth = new_smooth
                    cur_ac = new_ac
                    if new_hard < best_hard:
                        best_hard = new_hard
                        best_h = h.copy()
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    h[sl] = old
                    no_improve_count += 1
            elif rng.random() < 0.30:
                # Two-bit flip
                i1 = int(rng.integers(0, N))
                i2 = int(rng.integers(0, N))
                if i1 == i2:
                    i2 = (i2 + 1) % N
                old1, old2 = h[i1], h[i2]
                h[i1] *= -1.0
                h[i2] *= -1.0
                new_ac = _autocorr_fast(h)
                new_hard = _hard_score_from_ac(new_ac)
                new_smooth = _smooth_score_from_ac(new_ac, beta)
                dS = new_smooth - cur_smooth
                if new_hard < cur_hard or (dS <= 0 or rng.random() < np.exp(-dS / T)):
                    cur_hard = new_hard
                    cur_smooth = new_smooth
                    cur_ac = new_ac
                    if new_hard < best_hard:
                        best_hard = new_hard
                        best_h = h.copy()
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    h[i1] = old1
                    h[i2] = old2
                    no_improve_count += 1
            else:
                i = int(rng.integers(0, N))
                old = h[i]
                h[i] *= -1.0
                new_ac = _autocorr_fast(h)
                new_hard = _hard_score_from_ac(new_ac)
                new_smooth = _smooth_score_from_ac(new_ac, beta)
                dS = new_smooth - cur_smooth
                if new_hard < cur_hard or (dS <= 0 or rng.random() < np.exp(-dS / T)):
                    cur_hard = new_hard
                    cur_smooth = new_smooth
                    cur_ac = new_ac
                    if new_hard < best_hard:
                        best_hard = new_hard
                        best_h = h.copy()
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    h[i] = old
                    no_improve_count += 1
            
            # Adaptive cooling and restart
            T = max(T_min, T * decay)
            
            # If stuck for too long, do a partial restart
            if no_improve_count > max_no_improve:
                # Keep best_h but perturb current h
                h = best_h.copy()
                # Random perturbation of a random block
                pert_size = int(rng.choice([8, 16, 32, 64]))
                pert_start = int(rng.integers(0, N - pert_size + 1))
                h[pert_start:pert_start + pert_size] *= -1.0
                cur_ac = _autocorr_fast(h)
                cur_hard = _hard_score_from_ac(cur_ac)
                cur_smooth = _smooth_score_from_ac(cur_ac, beta)
                T = 0.02  # reset temperature
                no_improve_count = 0

        # Greedy multi-scale polish
        h = best_h.copy()
        sizes_polish = [1, 2, 4, 8, 16, 32, 64]
        for _pass in range(40):
            if time.time() > deadline - 5.0:
                break
            any_improve = False
            for size in sizes_polish:
                for start in range(0, N - size + 1):
                    if time.time() > deadline - 5.0:
                        return best_h
                    sl = slice(start, start + size)
                    old = h[sl].copy()
                    h[sl] *= -1.0
                    new_ac = _autocorr_fast(h)
                    new_hard = _hard_score_from_ac(new_ac)
                    if new_hard < best_hard - 1e-12:
                        best_hard = new_hard
                        best_h = h.copy()
                        any_improve = True
                    else:
                        h[sl] = old
            if not any_improve:
                break

        # Final single-bit polish
        h = best_h.copy()
        for _pass in range(100):
            if time.time() > deadline - 2.0:
                break
            any_improve = False
            for i in range(N):
                if time.time() > deadline - 2.0:
                    break
                old = h[i]
                h[i] *= -1.0
                new_ac = _autocorr_fast(h)
                new_hard = _hard_score_from_ac(new_ac)
                if new_hard < best_hard - 1e-12:
                    best_hard = new_hard
                    best_h = h.copy()
                    any_improve = True
                else:
                    h[i] = old
            if not any_improve:
                break

        return best_h
    except Exception:
        return np.ones(N)


def _random_blocks(rng, block_sizes):
    """Generate sign sequence by flipping random blocks of various sizes."""
    h = np.ones(N)
    for _ in range(50):
        size = int(rng.choice(block_sizes))
        start = int(rng.integers(0, N - size + 1))
        h[start:start + size] *= -1.0
    return h
