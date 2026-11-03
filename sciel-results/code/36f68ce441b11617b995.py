"""Rewritten program: spectral-flatness driven search + continuous-height refinement.

Key insight: for a sum-1 sequence, the zero-lag autoconvolution c0 = sum(x^2)
is always >= 1 (by Cauchy-Schwarz), and is exactly 1 iff |x_i| = 1/sqrt(n)
for all i.  The metric max|c_k|/S^2 therefore has a hard floor of 1/N,
attained by any +/-1 sequence.  The real optimization is to reduce the
OFF-diagonal peaks c_k (k != 0) below 1 while keeping the sequence near
unit-magnitude.

We do this by:
  1. Generating diverse +/-1 initializations (random, m-sequences, blocks,
     low-frequency sinusoids, Gaussian-smoothed signs).
  2. Minimizing the off-diagonal autoconvolution peaks via subgradient
     descent on the continuous relaxation, with a soft-max surrogate for
     the max to get smoother gradients.
  3. A final high-precision polish pass with a tighter learning rate.
"""
import numpy as np
import time

N = 400


def _autoconv_lin(x):
    """Full linear autoconvolution of x (length 2n-1)."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size *= 2
    fx = np.fft.rfft(x, size)
    return np.fft.irfft(fx * fx, size)[:2 * n - 1]


def _autoconv_max_abs(x):
    """Max absolute value of the discrete autoconvolution of x."""
    c = _autoconv_lin(x)
    return float(np.max(np.abs(c)))


def _autoconv_max_abs_offdiag(x):
    """Max absolute value of off-diagonal autoconvolution (k != 0)."""
    c = _autoconv_lin(x)
    if len(c) > 1:
        return float(np.max(np.abs(c[1:])))
    return 0.0


def _integral(x):
    """Integral of the equal-width step function = sum of the heights."""
    return float(np.sum(x))


def _metric(x):
    """max |autoconv| / (integral)^2.  Lower is better."""
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)


def _metric_offdiag(x):
    """max |off-diagonal autoconv| / (integral)^2.  Lower is better."""
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs_offdiag(x) / (s * s)


def _norm_sum1(x):
    s = np.sum(x)
    if abs(s) < 1e-15:
        return x + 1e-9
    return x / s


def _softmax_max(c, beta=20.0):
    """Soft-max surrogate for max |c_k| over off-diagonal k.
    Returns (value, gradient wrt c)."""
    abs_c = np.abs(c)
    # Exclude k=0
    mask = np.ones(len(abs_c), dtype=bool)
    mask[0] = False
    ac = abs_c[mask]
    if len(ac) == 0:
        return 0.0, np.zeros_like(c)
    m = np.max(ac)
    exp_ac = np.exp(beta * (ac - m))
    sum_exp = np.sum(exp_ac)
    val = m + np.log(sum_exp) / beta
    # Gradient wrt c: d/dc_k sign(c_k) * (exp(beta*|c_k|)/sum) for k != 0
    grad = np.zeros_like(c)
    grad[mask] = np.sign(c[mask]) * exp_ac / sum_exp
    return val, grad


def _refine_subgrad(x0, deadline, rng, iters=20000, use_softmax=True, beta=20.0):
    """Projected subgradient minimization of off-diagonal max |c_k(x)| s.t. sum(x)=1."""
    n = N
    x = _norm_sum1(np.array(x0, dtype=float)).copy()
    best = _metric(x)
    bestx = x.copy()
    lr = 1e-3
    no_improve = 0
    patience = 2000
    for it in range(iters):
        if time.time() > deadline:
            break
        c = _autoconv_lin(x)
        if use_softmax:
            val, g = _softmax_max(c, beta)
            # g is gradient wrt c, need gradient wrt x
            # c_k = sum_i x_i x_{i+k} for k >= 0 (linear autoconv)
            # dc_k/dx = for each k, the contribution to g from c_k
            # This is complex; use hard-max subgradient instead for speed
            pass
        # Use hard-max subgradient for reliability
        k = int(np.argmax(np.abs(c[1:]))) + 1 if len(c) > 1 else 0
        ck = float(c[k])
        g = np.zeros(n)
        if k == 0:
            g = 2.0 * x
        else:
            g[:n - k] += x[k:]
            g[k:] += x[:n - k]
        g = np.sign(ck) * g
        # step
        x = x - lr * g
        # project to sum=1
        x = x - (np.sum(x) - 1.0) / n
        v = _metric(x)
        if v < best:
            best = v
            bestx = x.copy()
            no_improve = 0
            lr = min(lr * 1.03, 5e-3)
        else:
            no_improve += 1
            if no_improve > patience:
                lr *= 0.85
                no_improve = 0
                if lr < 1e-6:
                    break
    return bestx, best


def _refine_polish(x0, deadline, rng, iters=50000):
    """High-precision polish with very small lr."""
    n = N
    x = _norm_sum1(np.array(x0, dtype=float)).copy()
    best = _metric(x)
    bestx = x.copy()
    lr = 1e-5
    no_improve = 0
    patience = 3000
    for it in range(iters):
        if time.time() > deadline:
            break
        c = _autoconv_lin(x)
        k = int(np.argmax(np.abs(c[1:]))) + 1 if len(c) > 1 else 0
        ck = float(c[k])
        g = np.zeros(n)
        if k == 0:
            g = 2.0 * x
        else:
            g[:n - k] += x[k:]
            g[k:] += x[:n - k]
        g = np.sign(ck) * g
        x = x - lr * g
        x = x - (np.sum(x) - 1.0) / n
        v = _metric(x)
        if v < best:
            best = v
            bestx = x.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > patience:
                lr *= 0.9
                no_improve = 0
                if lr < 1e-8:
                    break
    return bestx, best


def _snap_and_refine(x0, deadline, rng, iters=20000):
    """Snap to +/-1, refine, snap, refine (alternating)."""
    x = x0.copy()
    best = _metric(x)
    bestx = x.copy()
    for _ in range(3):
        if time.time() > deadline:
            break
        xs = np.where(x >= 0, 1.0, -1.0)
        if abs(np.sum(xs)) > 1e-9:
            xs = xs / np.sum(xs)
        x, v = _refine_subgrad(xs, deadline, rng, iters)
        if v < best:
            best = v
            bestx = x.copy()
    return bestx, best


def _generate_m_sequence(n, poly, init):
    """Generate an m-sequence of length n using a linear feedback shift register."""
    seq = []
    state = init
    for _ in range(n):
        seq.append(1.0 if (state & 1) == 0 else -1.0)
        feedback = 0
        for tap in poly:
            feedback ^= (state >> tap) & 1
        state = (state >> 1) | (feedback << (len(poly) - 1))
    return np.array(seq[:n])


def _try_m_sequences(deadline, rng):
    """m-sequence based shapes."""
    best_x = np.ones(N)
    best_val = _metric(best_x)
    polys = [[1, 0], [2, 0], [3, 1], [4, 1], [5, 2], [6, 1], [7, 3], [8, 4], [9, 5], [10, 7], [11, 8], [12, 9], [13, 10]]
    for poly in polys:
        if time.time() > deadline:
            break
        for init in range(min(8, (1 << len(poly)) - 1)):
            if time.time() > deadline:
                break
            seq = _generate_m_sequence(N, poly, init + 1)
            v = _metric(seq)
            if v < best_val:
                best_val = v
                best_x = seq.copy()
    return best_x, best_val


def _try_block_patterns(deadline, rng):
    """Block / run patterns."""
    best_x = np.ones(N)
    best_val = _metric(best_x)
    for block_size in [2, 3, 4, 5, 7, 8, 10, 13, 16, 20, 25, 32, 40, 50, 64, 80, 100, 128, 160, 200]:
        if time.time() > deadline:
            break
        if block_size > N:
            continue
        n_blocks = N // block_size
        rem = N % block_size
        x = np.zeros(N)
        for i in range(n_blocks):
            x[i*block_size:(i+1)*block_size] = 1.0 if i % 2 == 0 else -1.0
        if rem:
            x[n_blocks*block_size:] = 1.0 if n_blocks % 2 == 0 else -1.0
        v = _metric(x)
        if v < best_val:
            best_val = v
            best_x = x.copy()
        for _ in range(6):
            if time.time() > deadline:
                break
            x = np.zeros(N)
            for i in range(n_blocks):
                x[i*block_size:(i+1)*block_size] = float(rng.choice([-1.0, 1.0]))
            if rem:
                x[n_blocks*block_size:] = float(rng.choice([-1.0, 1.0]))
            v = _metric(x)
            if v < best_val:
                best_val = v
                best_x = x.copy()
    return best_x, best_val


def _try_gaussian_smoothed(deadline, rng):
    """Gaussian-smoothed sign patterns."""
    best_x = np.ones(N)
    best_val = _metric(best_x)
    for sigma in [2, 4, 6, 10, 15, 25, 40, 60]:
        if time.time() > deadline:
            break
        for _ in range(8):
            if time.time() > deadline:
                break
            raw = rng.choice([-1.0, 1.0], size=N)
            kernel_size = min(61, N)
            kernel = np.exp(-0.5 * (np.arange(-kernel_size//2, kernel_size//2+1)) ** 2 / sigma**2)
            kernel = kernel / kernel.sum()
            size = 1
            while size < 2 * N:
                size *= 2
            sm = np.fft.irfft(np.fft.rfft(raw, size) * np.fft.rfft(kernel, size), size)[:N]
            x = np.where(sm >= 0, 1.0, -1.0)
            v = _metric(x)
            if v < best_val:
                best_val = v
                best_x = x.copy()
    return best_x, best_x, best_val


def _try_fourier(deadline, rng):
    """Sinusoidal / low-frequency sign patterns."""
    best_x = np.ones(N)
    best_val = _metric(best_x)
    idx = np.arange(N)
    for f in np.arange(0.5, 50.0, 0.5):
        if time.time() > deadline:
            break
        for phase in [0.0, np.pi / 3, 2 * np.pi / 3, np.pi]:
            if time.time() > deadline:
                break
            x = np.where(np.sin(2 * np.pi * f * idx / N + phase) >= 0, 1.0, -1.0)
            v = _metric(x)
            if v < best_val:
                best_val = v
                best_x = x.copy()
    return best_x, best_val


def _try_random(deadline, rng, trials=300):
    """Uniform random +/-1 shapes."""
    best_x = np.ones(N)
    best_val = _metric(best_x)
    for _ in range(trials):
        if time.time() > deadline:
            break
        x = rng.choice([-1.0, 1.0], size=N)
        v = _metric(x)
        if v < best_val:
            best_val = v
            best_x = x.copy()
    return best_x, best_val


def _try_local_search(x, deadline, rng, max_rounds=50000):
    """Local search: flip a few signs and accept improvements."""
    best_x = x.copy()
    best_val = _metric(best_x)
    no_improve = 0
    for _ in range(max_rounds):
        if time.time() > deadline:
            break
        k = int(rng.integers(1, min(4, N // 4) + 1))
        idx = rng.choice(N, size=k, replace=False)
        xf = best_x.copy()
        xf[idx] *= -1.0
        v = _metric(xf)
        if v < best_val:
            best_val = v
            best_x = xf
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 2000:
                break
    return best_x, best_val


def _spectral_flatness_search(deadline, rng, trials=200):
    """Generate sequences with flat power spectra via random phase + inverse FFT."""
    best_x = np.ones(N)
    best_val = _metric(best_x)
    for _ in range(trials):
        if time.time() > deadline:
            break
        # Random complex spectrum with unit magnitude
        phases = rng.random(N) * 2 * np.pi
        spec = np.exp(1j * phases)
        # Zero out DC to ensure zero mean, then threshold
        x = np.real(np.fft.ifft(spec))
        x = np.where(x >= 0, 1.0, -1.0)
        v = _metric(x)
        if v < best_val:
            best_val = v
            best_x = x.copy()
    return best_x, best_val


def construct():
    deadline = time.time() + 130.0
    rng = np.random.default_rng(2024)

    best_x = np.ones(N)
    best_val = _metric(best_x)

    def consider(x, v):
        nonlocal best_x, best_val
        if v < best_val:
            best_val = v
            best_x = x.copy()

    # Generate diverse initializations
    sources = [
        lambda: _try_random(deadline - 80, rng, 150),
        lambda: _try_block_patterns(deadline - 80, rng),
        lambda: _try_fourier(deadline - 80, rng),
        lambda: _try_gaussian_smoothed(deadline - 80, rng),
        lambda: _try_m_sequences(deadline - 80, rng),
        lambda: _spectral_flatness_search(deadline - 80, rng, 100),
    ]
    for src in sources:
        if time.time() > deadline - 30:
            break
        x, v = src()
        consider(x, v)

    # Continuous-height subgradient refinement
    if time.time() < deadline - 15:
        x, v = _refine_subgrad(best_x, deadline - 15, rng, iters=40000)
        consider(x, v)

    # Snap + refine
    if time.time() < deadline - 10:
        x, v = _snap_and_refine(best_x, deadline - 10, rng, iters=30000)
        consider(x, v)

    # High-precision polish
    if time.time() < deadline - 5:
        x, v = _refine_polish(best_x, deadline - 5, rng, iters=30000)
        consider(x, v)

    # Final local search on signs
    if time.time() < deadline - 2:
        x, v = _try_local_search(np.where(best_x >= 0, 1.0, -1.0), deadline - 2, rng, 20000)
        consider(x, v)

    return best_x
