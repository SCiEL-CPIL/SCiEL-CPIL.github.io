"""Autoconvolution minimization via chaotic-map initialization + simulated annealing.

Structural change: instead of pure random initializations, we generate candidate
sign sequences from a logistic map at r=4 sampled at an irrational rotation.
This produces pseudo-random, low-discrepancy sequences that naturally avoid
periodic structures. We then refine with simulated annealing using block flips.

The key insight: chaotic sequences have complex correlation structure that may
lead to lower autoconvolution peaks than purely random signs, and they provide
diverse starting points for the annealing search.
"""
import time
import numpy as np

N = 400
L = 1024  # FFT length >= 2*N-1 = 799
_SEED = 12345

# Precompute the zero-padding indices once.
_PAD = L - N


def _autoconv_max_abs(x):
    """Return max(|linear autoconvolution|) for a real array x of len N."""
    xf = np.zeros(L, dtype=np.float64)
    xf[:N] = x
    spec = np.fft.rfft(xf)
    conv = np.fft.irfft(spec * spec.conj(), n=L)
    return float(np.max(np.abs(conv[:2 * N - 1])))


def _chaotic_sequence(seed_val, n):
    """Generate a sign sequence from logistic map at r=4 sampled at irrational rotation.
    
    Uses x_{k+1} = 4*x_k*(1-x_k) and maps to signs via thresholding at 0.5.
    The initial condition is derived from the seed to ensure reproducibility.
    """
    # Use the seed to create a unique initial condition in (0, 1)
    # Mix in some structure to avoid trivial orbits
    rng = np.random.default_rng(seed_val)
    x = rng.uniform(0.01, 0.99)
    
    signs = np.empty(n, dtype=np.float64)
    for i in range(n):
        # Logistic map iteration
        x = 4.0 * x * (1.0 - x)
        # Map to sign: +1 if x > 0.5, else -1
        signs[i] = 1.0 if x > 0.5 else -1.0
    
    return signs


def _block_flip(x, start, length):
    """Flip a contiguous block of signs in-place."""
    idx = np.arange(start, start + length)
    x[idx] = -x[idx]


def _sa_run(x, budget, rng, t0, deadline):
    """Simulated annealing with block-flip moves. Returns best (val, x)."""
    best_val = _autoconv_max_abs(x)
    best_x = x.copy()
    cur_val = best_val
    cur_x = x.copy()

    # Temperature schedule
    t_start = 0.05
    t_end = 1e-5
    n_iter = int(budget)

    for it in range(n_iter):
        if time.time() - t0 > deadline:
            break

        # Pick a block size
        if rng.random() < 0.75:
            blen = int(rng.integers(4, 17))
        else:
            blen = int(rng.integers(17, 33))
        start = int(rng.integers(0, N - blen + 1))

        _block_flip(cur_x, start, blen)
        new_val = _autoconv_max_abs(cur_x)
        delta = new_val - cur_val

        frac = it / max(1, n_iter - 1)
        temp = t_start * (t_end / t_start) ** frac

        if delta <= 0 or rng.random() < np.exp(-delta / max(temp, 1e-12)):
            cur_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_x = cur_x.copy()
        else:
            _block_flip(cur_x, start, blen)

    return best_val, best_x


def construct():
    t0 = time.time()
    deadline = 110.0

    rng = np.random.default_rng(_SEED)

    # Generate multiple chaotic initializations and pick the best
    # This provides diverse, structured starting points
    n_chaotic = 10
    best_val = float('inf')
    best_x = None

    for i in range(n_chaotic):
        # Each chaotic sequence uses a different seed derived from the main seed
        seq_seed = _SEED + i * 7919  # Large prime for good mixing
        x = _chaotic_sequence(seq_seed, N)
        val = _autoconv_max_abs(x)
        if val < best_val:
            best_val = val
            best_x = x

    # Also try a pure random initialization as a baseline
    x_rand = rng.choice([-1.0, 1.0], size=N)
    val_rand = _autoconv_max_abs(x_rand)
    if val_rand < best_val:
        best_val = val_rand
        best_x = x_rand

    # Multi-restart simulated annealing within the time budget
    restarts = 0
    while time.time() - t0 < deadline - 5.0:
        restarts += 1

        # Budget for this restart
        if restarts == 1:
            budget = 20000
        elif restarts == 2:
            budget = 15000
        else:
            budget = 8000

        # New starting point: perturb the current best or use a new chaotic sequence
        if restarts == 1:
            start_x = best_x.copy()
        else:
            # Alternate between perturbing the best and using new chaotic sequences
            if restarts % 2 == 0:
                # Perturb the best found so far
                start_x = best_x.copy()
                n_mut = int(rng.integers(20, 80))
                mut_idx = rng.choice(N, size=n_mut, replace=False)
                start_x[mut_idx] = -start_x[mut_idx]
            else:
                # Use a new chaotic sequence
                seq_seed = _SEED + (restarts * 7919)
                start_x = _chaotic_sequence(seq_seed, N)

        val, xx = _sa_run(start_x, budget, rng, t0, deadline)
        if val < best_val:
            best_val = val
            best_x = xx

        if time.time() - t0 > deadline - 5.0:
            break

    return best_x
