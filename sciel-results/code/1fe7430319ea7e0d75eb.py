"""Autoconvolution minimization via block-flip simulated annealing.

We choose 400 signed step heights (+/-1) to minimize the maximum absolute
value of the (linear) autoconvolution, which is the C3 upper bound we want
to lower.  Evaluation is done with a zero-padded FFT so each candidate costs
a single O(L log L) transform (L=1024).
"""
import time
import numpy as np

N = 400
L = 1024  # FFT length >= 2*N-1 = 799
_SEED = 12345

# Precompute the zero-padding indices once.
_PAD = L - N  # amount of trailing zero padding for linear convolution


def _autoconv_max_abs(x):
    """Return max(|linear autoconvolution|) for a real array x of len N."""
    # Pad to length L for linear convolution via circular convolution.
    xf = np.zeros(L, dtype=np.float64)
    xf[:N] = x
    spec = np.fft.rfft(xf)
    conv = np.fft.irfft(spec * spec.conj(), n=L)
    # Linear autoconvolution lives in conv[0:2*N-1] (the rest is wrap-around).
    return float(np.max(np.abs(conv[:2 * N - 1])))


def _block_flip(x, start, length, rng):
    """Flip a contiguous block of signs in-place; return the flipped segment."""
    idx = np.arange(start, start + length)
    x[idx] = -x[idx]


def _sa_run(x, budget, rng, t0, deadline):
    """Simulated annealing with block-flip moves.  Returns best (val, x)."""
    best_val = _autoconv_max_abs(x)
    best_x = x.copy()
    cur_val = best_val
    cur_x = x

    # Temperature schedule: start high enough to accept big jumps, cool down.
    t_start = 0.05
    t_end = 1e-5
    n_iter = int(budget)

    for it in range(n_iter):
        # Deadline guard with safety margin.
        if time.time() - t0 > deadline:
            break

        # Pick a block size: mostly small (4-16), sometimes larger (17-32).
        if rng.random() < 0.75:
            blen = int(rng.integers(4, 17))
        else:
            blen = int(rng.integers(17, 33))
        start = int(rng.integers(0, N - blen + 1))

        _block_flip(cur_x, start, blen, rng)
        new_val = _autoconv_max_abs(cur_x)
        delta = new_val - cur_val

        # Temperature for this iteration (geometric cooling).
        frac = it / max(1, n_iter - 1)
        temp = t_start * (t_end / t_start) ** frac

        if delta <= 0 or rng.random() < np.exp(-delta / max(temp, 1e-12)):
            cur_val = new_val
            if new_val < best_val:
                best_val = new_val
                best_x = cur_x.copy()
        else:
            # Reject: undo the flip.
            _block_flip(cur_x, start, blen, rng)

    return best_val, best_x


def construct():
    t0 = time.time()
    # Safety margin: the evaluator kills at ~150s; stay well under 120s.
    deadline = 110.0

    rng = np.random.default_rng(_SEED)

    # Initial candidate: random signs (better than all-ones for this metric).
    x = rng.choice([-1.0, 1.0], size=N)

    best_val = _autoconv_max_abs(x)
    best_x = x.copy()

    # Multi-restart simulated annealing within the time budget.
    # Each restart gets a shrinking budget; early restarts are longer.
    restarts = 0
    while time.time() - t0 < deadline - 5.0:
        restarts += 1
        # Budget for this restart: longer early, shorter later.
        if restarts == 1:
            budget = 20000
        elif restarts == 2:
            budget = 15000
        else:
            budget = 8000

        # New starting point: mutate the current best a little, or fresh random.
        if restarts == 1:
            start_x = x.copy()
        else:
            # Perturb the best found so far to diversify.
            start_x = best_x.copy()
            n_mut = int(rng.integers(20, 80))
            mut_idx = rng.choice(N, size=n_mut, replace=False)
            start_x[mut_idx] = -start_x[mut_idx]

        val, xx = _sa_run(start_x, budget, rng, t0, deadline)
        if val < best_val:
            best_val = val
            best_x = xx

        if time.time() - t0 > deadline - 5.0:
            break

    return best_x
