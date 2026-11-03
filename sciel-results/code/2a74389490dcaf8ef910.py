"""Autoconvolution minimization: continuous gradient-descent seed + discrete SA + polish.

We choose 400 signed step heights (+/-1) to minimize the maximum absolute value
of the linear autoconvolution (the C3 upper bound).  The pipeline is:
  1. Continuous gradient descent on a smooth LSE-peak surrogate of the
     normalized autocorrelation (flat-spectrum optimization) -> threshold to {-1,1}.
  2. Discrete simulated annealing with incremental-FFT single-bit flips and
     occasional block flips.
  3. Greedy multi-scale local polish with incremental-FFT single-bit passes.
Each FFT-based score costs O(L log L); single-bit flips use an O(L) FFT update.
"""
import time
import numpy as np

N = 400
L = 1024  # FFT length >= 2*N-1 = 799
_SEED = 12345
# Precompute the frequency grid for incremental FFT updates
_FREQ_GRID = np.arange(L // 2 + 1)  # 0..L/2
# exp(-2*pi*i*k*i/L) for k in 0..L/2, i in 0..N-1  -> shape (N, L//2+1)
# We compute on the fly per index to save memory; precompute the base exponent.
_PI2I_OVER_L = -2j * np.pi / L


def _autocorr_fast(h):
    """Linear one-sided autocorrelation ac[k] = sum_i h[i] h[i+k], k = 0..N-1."""
    F = np.fft.rfft(h, n=L)
    ac = np.fft.irfft(F * np.conj(F), n=L)[:N]
    return ac


def _hard_score(h):
    """Peak autoconvolution concentration (lower is better)."""
    ac = _autocorr_fast(h)
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9
    return float(np.max(np.abs(ac)) / dc)


def _lse_peak_score(h, beta):
    """Smooth LSE surrogate of the peak of |ac[1:]|/ac[0] (lower is better).
    Returns (score, ac, F) so callers can reuse ac and F."""
    F = np.fft.rfft(h, n=L)
    ac = np.fft.irfft(F * np.conj(F), n=L)[:N]
    dc = ac[0]
    if dc <= 1e-30:
        return 1e9, ac, F
    y = np.abs(ac[1:]) / dc
    m = float(np.max(y))
    z = beta * (y - m)
    s = m + float(np.log(np.sum(np.exp(z)))) / beta
    return s, ac, F


def _lse_peak_gradient(h, beta):
    """Gradient of the LSE-peak surrogate w.r.t. h (continuous).
    Returns (grad, score).  grad has shape (N,)."""
    F = np.fft.rfft(h, n=L)
    ac = np.fft.irfft(F * np.conj(F), n=L)[:N]
    dc = ac[0]
    if dc <= 1e-30:
        return np.zeros(N), 1e9
    y = np.abs(ac[1:]) / dc
    m = float(np.max(y))
    z = beta * (y - m)
    e = np.exp(z)
    w = e / float(np.sum(e))  # softmax weights over lags 1..N-1
    # score = m + (1/beta) log(sum exp(z))
    # d(score)/d(ac[k]) for k>=1: w[k-1] / dc * sign(ac[k])
    #   (ignoring the m term's kink, which is handled by the max)
    # d(ac[k])/d(h[i]) = h[i+k] + h[i-k]  (for the linear autocorr, with zero padding)
    #   More precisely, ac[k] = sum_j h[j] h[j+k], so d(ac[k])/d(h[i]) = h[i+k] + h[i-k]
    #   (with out-of-range terms = 0)
    # We build the gradient by accumulating over k.
    # Use FFT-based approach for efficiency
    w_ext = np.zeros(L)
    w_ext[1:N] = w / dc
    signs = np.zeros(L)
    signs[1:N] = np.sign(ac[1:N])
    w_signed = w_ext * signs
    h_ext = np.zeros(L)
    h_ext[:N] = h
    Ws = np.fft.rfft(w_signed, n=L)
    Hc = np.fft.rfft(h_ext, n=L)
    # conv1[i] = sum_k w_signed[k] * h[i+k]
    conv1 = np.fft.irfft(Ws * np.conj(Hc), n=L)[:N]
    # conv2[i] = sum_k w_signed[k] * h[i-k]
    conv2 = np.fft.irfft(Ws * Hc, n=L)[:N]
    grad = conv1 + conv2
    score = m + float(np.log(np.sum(e))) / beta
    return grad, score


def _l2_offdc_score_and_grad(h):
    """Smooth surrogate: minimize sum_{k>=1} ac[k]^2 (off-DC energy).
    This is the classic flat-spectrum / low-PAPR objective.  Returns (score, grad)."""
    F = np.fft.rfft(h, n=L)
    ac = np.fft.irfft(F * np.conj(F), n=L)[:N]
    # ac[k] = sum_j h[j] h[j+k];  d(ac[k])/d(h[i]) = h[i+k] + h[i-k]
    # score = sum_{k>=1} ac[k]^2  ->  grad[i] = 2 * sum_{k>=1} ac[k] * (h[i+k] + h[i-k])
    # Use FFT-based approach: grad = 2 * (ac_convolved_with_h + h_convolved_with_ac)
    # where ac_convolved_with_h[i] = sum_k ac[k] * h[i+k] (with zero padding)
    # and h_convolved_with_ac[i] = sum_k ac[k] * h[i-k]
    # Both can be computed via FFT convolution
    ac_ext = np.zeros(L)
    ac_ext[:N] = ac
    ac_ext[0] = 0.0  # exclude DC term
    h_ext = np.zeros(L)
    h_ext[:N] = h
    # conv1[i] = sum_k ac[k] * h[i+k] = cross-correlation
    # In frequency domain: IFFT(FFT(ac_ext) * conj(FFT(h_ext)))
    Ac = np.fft.rfft(ac_ext, n=L)
    Hc = np.fft.rfft(h_ext, n=L)
    conv1 = np.fft.irfft(Ac * np.conj(Hc), n=L)[:N]
    # conv2[i] = sum_k ac[k] * h[i-k] = convolution
    # In frequency domain: IFFT(FFT(ac_ext) * FFT(h_ext))
    conv2 = np.fft.irfft(Ac * Hc, n=L)[:N]
    grad = 2.0 * (conv1 + conv2)
    score = float(np.sum(ac[1:] ** 2))
    return score, grad


def _fft_update_single_flip(F, old_val, i):
    """Return new FFT after flipping h[i] -> -h[i].
    F_new = F - 2*h[i]*exp(-2*pi*i*k*i/L)  for k in 0..L/2.
    O(L) operation."""
    k = np.arange(L // 2 + 1)
    phase = np.exp(_PI2I_OVER_L * k * i)
    F_new = F - 2.0 * old_val * phase
    return F_new


def _fft_update_block_flip(F, h, start, size):
    """Return new FFT after flipping h[start:start+size] -> -h[...].
    F_new = F - 2 * sum_{j=0}^{size-1} h[start+j] * exp(-2*pi*i*k*(start+j)/L)
    O(L * size) but with a small constant; much faster than a full rfft when
    size << N and we are doing many flips."""
    idx = np.arange(start, start + size)
    vals = h[idx]
    # Build the phase matrix (size, L//2+1)
    k = np.arange(L // 2 + 1)
    phase = np.exp(_PI2I_OVER_L * np.outer(idx, k))  # (size, L//2+1)
    delta = 2.0 * (vals[:, None] * phase).sum(axis=0)
    F_new = F - delta
    return F_new


def _autocorr_from_fft(F):
    ac = np.fft.irfft(F * np.conj(F), n=L)[:N]
    return ac


def _continuous_seed(rng, deadline):
    """Run continuous gradient descent on the L2 off-DC spectrum surrogate, then
    threshold to {-1,1}.  Optionally do a second short LSE-peak phase to sharpen.
    Returns a {-1,1} array of length N."""
    # Start from a random continuous signal in [-1, 1]
    h = rng.uniform(-1.0, 1.0, size=N)
    best_h = h.copy()
    best_score = float(np.sum(_autocorr_fast(h)[1:] ** 2))

    # Phase A: minimize L2 off-DC energy (flat spectrum) -- cheap, smooth, well-conditioned
    lr = 0.05
    for step in range(2000):
        if time.time() > deadline - 80.0:
            break
        score, grad = _l2_offdc_score_and_grad(h)
        if score < best_score:
            best_score = score
            best_h = h.copy()
        gnorm = float(np.linalg.norm(grad))
        if gnorm < 1e-12:
            break
        h -= lr * grad / (gnorm + 1e-12) * 0.5
        h = np.clip(h, -2.0, 2.0)
        lr = max(lr * 0.9995, 0.005)

    # Threshold
    h_bin = np.sign(best_h)
    h_bin[h_bin == 0] = 1.0

    # Phase B: short LSE-peak refinement on the continuous signal starting from best_h
    if time.time() < deadline - 60.0:
        h = best_h.copy()
        beta = 50.0
        lr = 0.1
        for step in range(500):
            if time.time() > deadline - 60.0:
                break
            grad, score = _lse_peak_gradient(h, beta)
            gnorm = float(np.linalg.norm(grad))
            if gnorm < 1e-12:
                break
            h -= lr * grad / (gnorm + 1e-12) * 0.1
            h = np.clip(h, -2.0, 2.0)
            beta = min(beta * 1.005, 300.0)
            lr = max(lr * 0.999, 0.01)
        # Re-threshold and keep the better of the two thresholds
        h_bin2 = np.sign(h)
        h_bin2[h_bin2 == 0] = 1.0
        if _hard_score(h_bin2) < _hard_score(h_bin):
            h_bin = h_bin2

    return h_bin


def construct():
    try:
        t0 = time.time()
        deadline = t0 + 140.0  # safety margin before 150s limit
        rng = np.random.default_rng(_SEED)

        idx = np.arange(N)

        # ---------- Phase 1: continuous gradient-descent seed ----------
        h_cont_seed = _continuous_seed(rng, deadline)
        s_cont = _hard_score(h_cont_seed)

        # ---------- Phase 2: diverse initializations ----------
        inits = [
            h_cont_seed,
            np.ones(N),
            rng.choice(np.array([-1.0, 1.0]), size=N),
            np.cos(2 * np.pi * idx ** 2 / (2 * N)),
            np.cos(2 * np.pi * idx ** 2 / (N + 1)),
            np.cos(2 * np.pi * idx ** 2 / (3 * N)),
            np.cos(2 * np.pi * idx ** 2 / (N - 1)),
            np.cos(2 * np.pi * idx ** 2 / (5 * N)),
            np.cos(2 * np.pi * idx ** 2 / (7 * N)),
            rng.choice(np.array([-1.0, 1.0]), size=N),
            np.cos(2 * np.pi * idx / N),
            np.sin(2 * np.pi * idx / N),
            np.cos(2 * np.pi * 2 * idx / N),
            np.sin(2 * np.pi * 2 * idx / N),
            np.cos(2 * np.pi * 3 * idx / N),
            np.sin(2 * np.pi * 3 * idx / N),
            np.cos(2 * np.pi * 5 * idx / N),
            np.sin(2 * np.pi * 5 * idx / N),
            # Gold mean / Thue-Morse-like patterns
            np.sign(np.cos(2 * np.pi * idx * np.sqrt(2) / N)),
            np.sign(np.cos(2 * np.pi * idx * np.sqrt(3) / N)),
            # Alternating blocks
            np.sign(np.sin(2 * np.pi * idx * 8 / N)),
            np.sign(np.sin(2 * np.pi * idx * 16 / N)),
            rng.choice(np.array([-1.0, 1.0]), size=N),
        ]
        best_h = None
        best_hard = 1e9
        for h in inits:
            s = _hard_score(h)
            if s < best_hard:
                best_hard = s
                best_h = h.copy()

        h = best_h.copy()
        cur_hard = best_hard
        F_cur = np.fft.rfft(h, n=L)

        # ---------- Phase 3: discrete SA with incremental FFT ----------
        T = 0.08
        T_min = 1e-4
        decay = 0.9999
        beta = 50.0
        sizes = np.array([2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128])

        # Precompute the current smooth score once; update incrementally when we accept.
        def _smooth_from_ac(ac):
            dc = ac[0]
            if dc <= 1e-30:
                return 1e9
            y = np.abs(ac[1:]) / dc
            m = float(np.max(y))
            z = beta * (y - m)
            return m + float(np.log(np.sum(np.exp(z)))) / beta

        ac_cur = _autocorr_from_fft(F_cur)
        cur_smooth = _smooth_from_ac(ac_cur)
        step_count = 0

        while time.time() < deadline - 25.0:
            step_count += 1
            if rng.random() < 0.35:
                # Block flip with incremental FFT
                size = int(rng.choice(sizes))
                start = int(rng.integers(0, N - size + 1))
                sl = slice(start, start + size)
                old = h[sl].copy()
                h[sl] *= -1.0
                F_new = _fft_update_block_flip(F_cur, old, start, size)
                ac_new = _autocorr_from_fft(F_new)
                dc = ac_new[0]
                if dc <= 1e-30:
                    new_hard = 1e9
                else:
                    new_hard = float(np.max(np.abs(ac_new)) / dc)
                if new_hard < cur_hard:
                    cur_hard = new_hard
                    F_cur = F_new
                    ac_cur = ac_new
                    cur_smooth = _smooth_from_ac(ac_new)
                    if new_hard < best_hard:
                        best_hard = new_hard
                        best_h = h.copy()
                else:
                    new_smooth = _smooth_from_ac(ac_new)
                    dS = new_smooth - cur_smooth
                    if dS <= 0 or rng.random() < np.exp(-dS / T):
                        cur_hard = new_hard
                        F_cur = F_new
                        ac_cur = ac_new
                        cur_smooth = new_smooth
                    else:
                        h[sl] = old
            else:
                # Single-bit flip (incremental FFT, O(L))
                i = int(rng.integers(0, N))
                old_val = h[i]
                h[i] *= -1.0
                F_new = _fft_update_single_flip(F_cur, old_val, i)
                ac_new = _autocorr_from_fft(F_new)
                dc = ac_new[0]
                if dc <= 1e-30:
                    new_hard = 1e9
                else:
                    new_hard = float(np.max(np.abs(ac_new)) / dc)
                if new_hard < cur_hard:
                    cur_hard = new_hard
                    F_cur = F_new
                    ac_cur = ac_new
                    cur_smooth = _smooth_from_ac(ac_new)
                    if new_hard < best_hard:
                        best_hard = new_hard
                        best_h = h.copy()
                else:
                    new_smooth = _smooth_from_ac(ac_new)
                    dS = new_smooth - cur_smooth
                    if dS <= 0 or rng.random() < np.exp(-dS / T):
                        cur_hard = new_hard
                        F_cur = F_new
                        ac_cur = ac_new
                        cur_smooth = new_smooth
                    else:
                        h[i] = old_val
            T = max(T_min, T * decay)
            # Periodic restart from best if stuck
            if (step_count % 5000 == 0) and (time.time() < deadline - 30.0):
                if cur_hard > best_hard * 1.05:
                    h = best_h.copy()
                    F_cur = np.fft.rfft(h, n=L)
                    ac_cur = _autocorr_from_fft(F_cur)
                    cur_hard = best_hard
                    cur_smooth = _smooth_from_ac(ac_cur)

        # ---------- Phase 4: greedy multi-scale local polish (incremental FFT for blocks) ----------
        h = best_h.copy()
        F_cur = np.fft.rfft(h, n=L)
        sizes_polish = [1, 2, 4, 8, 16, 32]
        for _pass in range(20):
            if time.time() > deadline - 8.0:
                break
            any_improve = False
            for size in sizes_polish:
                for start in range(0, N - size + 1, size):  # stride by size for speed
                    if time.time() > deadline - 8.0:
                        return best_h
                    sl = slice(start, start + size)
                    old = h[sl].copy()
                    h[sl] *= -1.0
                    F_new = _fft_update_block_flip(F_cur, old, start, size)
                    ac_new = _autocorr_from_fft(F_new)
                    dc = ac_new[0]
                    if dc <= 1e-30:
                        new_hard = 1e9
                    else:
                        new_hard = float(np.max(np.abs(ac_new)) / dc)
                    if new_hard < best_hard - 1e-12:
                        best_hard = new_hard
                        best_h = h.copy()
                        F_cur = F_new
                        any_improve = True
                    else:
                        h[sl] = old
            if not any_improve:
                break

        # ---------- Phase 5: high-precision single-bit polish (incremental FFT) ----------
        h = best_h.copy()
        F_cur = np.fft.rfft(h, n=L)
        for _pass in range(60):
            if time.time() > deadline - 3.0:
                break
            any_improve = False
            for i in range(N):
                if time.time() > deadline - 3.0:
                    return best_h
                old_val = h[i]
                h[i] *= -1.0
                F_new = _fft_update_single_flip(F_cur, old_val, i)
                ac_new = _autocorr_from_fft(F_new)
                dc = ac_new[0]
                if dc <= 1e-30:
                    new_hard = 1e9
                else:
                    new_hard = float(np.max(np.abs(ac_new)) / dc)
                if new_hard < best_hard - 1e-14:
                    best_hard = new_hard
                    best_h = h.copy()
                    F_cur = F_new
                    any_improve = True
                else:
                    h[i] = old_val
            if not any_improve:
                break

        return best_h
    except Exception:
        return np.ones(N)
