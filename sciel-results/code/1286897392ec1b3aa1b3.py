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
    grad = np.zeros(N)
    for k in range(1, N):
        wk = w[k - 1] / dc
        if wk < 1e-15:
            continue
        # d(ac[k])/d(h[i]) = h[i+k] + h[i-k] for valid i
        # i ranges 0..N-1; i+k valid if i+k < N; i-k valid if i-k >= 0
        # Contribution to grad[i] = wk * sign(ac[k]) * (h[i+k] + h[i-k])
        sgn = 1.0 if ac[k] >= 0 else -1.0
        # h[i+k] for i in 0..N-1-k
        if N - k > 0:
            grad[:N - k] += wk * sgn * h[k:]
        # h[i-k] for i in k..N-1
        if N - k > 0:
            grad[k:] += wk * sgn * h[:N - k]
    score = m + float(np.log(np.sum(e))) / beta
    return grad, score


def _fft_update_single_flip(F, h, i):
    """Return new FFT after flipping h[i] -> -h[i].
    F_new = F - 2*h[i]*exp(-2*pi*i*k*i/L)  for k in 0..L/2.
    O(L) operation."""
    old_val = h[i]
    # exp(-2*pi*i*k*i/L) for k = 0..L/2
    k = np.arange(L // 2 + 1)
    phase = np.exp(_PI2I_OVER_L * k * i)
    F_new = F - 2.0 * old_val * phase
    return F_new


def _autocorr_from_fft(F):
    ac = np.fft.irfft(F * np.conj(F), n=L)[:N]
    return ac


def _continuous_seed(rng, deadline):
    """Run continuous gradient descent on the LSE-peak surrogate, then threshold.
    Returns a {-1,1} array of length N."""
    # Start from a random continuous signal in [-1, 1]
    h = rng.uniform(-1.0, 1.0, size=N)
    beta = 50.0
    lr = 0.5
    best_h = h.copy()
    best_score = _lse_peak_score(h, beta)[0]

    for step in range(3000):
        if time.time() > deadline - 60.0:
            break
        grad, score = _lse_peak_gradient(h, beta)
        if score < best_score:
            best_score = score
            best_h = h.copy()
        # Normalize gradient to control step size
        gnorm = float(np.linalg.norm(grad))
        if gnorm < 1e-12:
            break
        h -= lr * grad / gnorm * 0.1
        # Keep h bounded
        h = np.clip(h, -2.0, 2.0)
        # Slowly increase beta to sharpen the peak focus
        beta = min(beta * 1.005, 500.0)
        # Decay learning rate
        lr = max(lr * 0.999, 0.01)

    # Threshold to {-1, 1}
    h_bin = np.sign(best_h)
    h_bin[h_bin == 0] = 1.0
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
            rng.choice(np.array([-1.0, 1.0]), size=N),
            np.cos(2 * np.pi * idx / N),
            np.sin(2 * np.pi * idx / N),
            np.cos(2 * np.pi * 2 * idx / N),
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
        T = 0.05
        T_min = 1e-4
        decay = 0.99985
        beta = 30.0
        sizes = np.array([2, 4, 8, 12, 16, 24, 32, 48, 64])

        while time.time() < deadline - 25.0:
            if rng.random() < 0.30:
                # Block flip (needs full FFT recompute)
                size = int(rng.choice(sizes))
                start = int(rng.integers(0, N - size + 1))
                sl = slice(start, start + size)
                old = h[sl].copy()
                h[sl] *= -1.0
                F_new = np.fft.rfft(h, n=L)
                ac_new = _autocorr_from_fft(F_new)
                dc = ac_new[0]
                if dc <= 1e-30:
                    new_hard = 1e9
                else:
                    new_hard = float(np.max(np.abs(ac_new)) / dc)
                if new_hard < cur_hard:
                    cur_hard = new_hard
                    F_cur = F_new
                    if new_hard < best_hard:
                        best_hard = new_hard
                        best_h = h.copy()
                else:
                    # Accept via SA if smooth score improves or random
                    y = np.abs(ac_new[1:]) / dc
                    m = float(np.max(y))
                    z = beta * (y - m)
                    new_smooth = m + float(np.log(np.sum(np.exp(z)))) / beta
                    # Compute current smooth score
                    ac_cur = _autocorr_from_fft(F_cur)
                    dc_cur = ac_cur[0]
                    if dc_cur > 1e-30:
                        y_cur = np.abs(ac_cur[1:]) / dc_cur
                        m_cur = float(np.max(y_cur))
                        z_cur = beta * (y_cur - m_cur)
                        cur_smooth = m_cur + float(np.log(np.sum(np.exp(z_cur)))) / beta
                    else:
                        cur_smooth = 1e9
                    dS = new_smooth - cur_smooth
                    if dS <= 0 or rng.random() < np.exp(-dS / T):
                        cur_hard = new_hard
                        F_cur = F_new
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
                    if new_hard < best_hard:
                        best_hard = new_hard
                        best_h = h.copy()
                else:
                    y = np.abs(ac_new[1:]) / dc
                    m = float(np.max(y))
                    z = beta * (y - m)
                    new_smooth = m + float(np.log(np.sum(np.exp(z)))) / beta
                    ac_cur = _autocorr_from_fft(F_cur)
                    dc_cur = ac_cur[0]
                    if dc_cur > 1e-30:
                        y_cur = np.abs(ac_cur[1:]) / dc_cur
                        m_cur = float(np.max(y_cur))
                        z_cur = beta * (y_cur - m_cur)
                        cur_smooth = m_cur + float(np.log(np.sum(np.exp(z_cur)))) / beta
                    else:
                        cur_smooth = 1e9
                    dS = new_smooth - cur_smooth
                    if dS <= 0 or rng.random() < np.exp(-dS / T):
                        cur_hard = new_hard
                        F_cur = F_new
                    else:
                        h[i] = old_val
            T = max(T_min, T * decay)

        # ---------- Phase 4: greedy multi-scale local polish ----------
        h = best_h.copy()
        F_cur = np.fft.rfft(h, n=L)
        sizes_polish = [2, 4, 8, 16, 32]
        for _pass in range(30):
            if time.time() > deadline - 8.0:
                break
            any_improve = False
            for size in sizes_polish:
                for start in range(0, N - size + 1):
                    if time.time() > deadline - 8.0:
                        return best_h
                    sl = slice(start, start + size)
                    old = h[sl].copy()
                    h[sl] *= -1.0
                    F_new = np.fft.rfft(h, n=L)
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
        for _pass in range(50):
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
