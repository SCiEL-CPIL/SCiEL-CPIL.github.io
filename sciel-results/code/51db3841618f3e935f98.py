"""
Signed autocorrelation maximum absolute-value variant.

We seek a signed step function f (length N) that minimizes:
    max_k | (f * f)[k] / (sum f)^2 |
where * is circular autoconvolution.

Strategy:
1. Initialize with a diverse set of candidates (all-ones, low-autocorrelation
   binary sequences, random).
2. Use a fast local search with efficient autoconvolution updates.
3. Use simulated annealing to escape local minima.
4. Final polish with a differentiable surrogate.

Key insight: The seed (all ones) gives score 1/N. We can do better by
using signs to reduce the peak autoconvolution while keeping the sum large.
"""

import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve

N = 400

def _score_sequence(s):
    """True score: max|autoconv| / (sum s)^2."""
    sq_int = (np.sum(s)) ** 2
    if sq_int < 1e-10:
        return np.inf
    ac = fftconvolve(s, s, mode='full')
    max_ac = np.max(np.abs(ac))
    return max_ac / sq_int

def _build_walsh_basis(n, rank):
    """Build orthogonal Walsh-Hadamard basis (non-DC) via iterative construction."""
    W = np.eye(n)
    h = 1
    while h < n:
        new_W = np.zeros_like(W)
        for i in range(0, n, 2*h):
            for j in range(i, i+h):
                new_W[j, :] = W[j, :] + W[j+h, :]
                new_W[j+h, :] = W[j, :] - W[j+h, :]
        W = new_W
        h *= 2
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    W = W / norms
    # First row is DC (all 1s/sqrt(n)), skip it
    return W[1:1+rank, :]

def _wha_objective(c, basis, K=128):
    """Soft-max objective for WHA superposition."""
    s = c @ basis
    ac = fftconvolve(s, s, mode='full')
    ac_sq = ac ** 2
    max_val = np.max(ac_sq)
    if max_val < 1e-10:
        return 0.0
    kk = min(K, ac_sq.size)
    idx = np.argpartition(ac_sq, -kk)[-kk:]
    top = ac_sq[idx]
    lse = max_val + np.log(np.sum(np.exp(top - max_val)))
    soft_max = np.sqrt(lse)
    sq_int = (np.sum(s)) ** 2
    if sq_int < 1e-10:
        return 1e10
    return soft_max / sq_int

def _wha_gradient(c, basis, K=128):
    """Gradient of WHA soft-max objective w.r.t. coefficients."""
    s = c @ basis
    ac = fftconvolve(s, s, mode='full')
    ac_sq = ac ** 2
    max_val = np.max(ac_sq)
    if max_val < 1e-10:
        return np.zeros_like(c)
    kk = min(K, ac_sq.size)
    idx = np.argpartition(ac_sq, -kk)[-kk:]
    top = ac_sq[idx]
    weights = np.exp(top - max_val)
    weights /= np.sum(weights)
    lse_val = max_val + np.log(np.sum(weights))
    dsoft_dac_sq = np.zeros(ac.shape)
    dsoft_dac_sq[idx] = weights / (2.0 * np.sqrt(lse_val))
    dsoft_dac = dsoft_dac_sq * 2.0 * ac
    # Gradient of autoconvolution w.r.t. s via FFT
    n_ac = ac.shape[0]
    n_fft = 1
    while n_fft < n_ac:
        n_fft *= 2
    F_dac = np.fft.rfft(dsoft_dac, n=n_fft)
    F_s = np.fft.rfft(s, n=n_fft)
    grad_s = 2.0 * np.fft.irfft(F_dac * F_s, n=n_fft)[:N]
    grad_c = grad_s @ basis
    sum_s = np.sum(s)
    if abs(sum_s) < 1e-10:
        return grad_c
    obj_val = _wha_objective(c, basis, K)
    grad_sum = np.sum(basis, axis=1)
    grad_c = grad_c / (sum_s ** 2) - 2.0 * obj_val * grad_sum / (sum_s ** 2)
    return grad_c

def _build_walsh_basis(n, rank):
    """Build orthogonal Walsh-Hadamard basis (DC excluded) via fast FWHT."""
    # Fast Walsh-Hadamard transform
    a = np.zeros(n)
    a[0] = 1.0
    h = 1
    while h < n:
        new_a = a.copy()
        for i in range(0, n, 2*h):
            for j in range(i, i+h):
                new_a[j] = a[j] + a[j+h]
                new_a[j+h] = a[j] - a[j+h]
        a = new_a
        h *= 2
    # a is the DC row; we need all rows. Use the property that FWHT is symmetric.
    # Actually, let's build the full matrix using the iterative construction.
    W = np.eye(n)
    h = 1
    while h < n:
        new_W = np.zeros_like(W)
        for i in range(0, n, 2*h):
            for j in range(i, i+h):
                new_W[j, :] = W[j, :] + W[j+h, :]
                new_W[j+h, :] = W[j, :] - W[j+h, :]
        W = new_W
        h *= 2
    # Normalize rows to unit norm
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    W = W / norms
    # Exclude DC (first row after this construction is all-ones/sqrt(n))
    # Actually the first row of W from this construction is all 1s/sqrt(n)
    # We want non-DC components for the superposition
    return W[1:1+rank, :]

def _wha_objective(c, basis, K=128):
    """Soft-max objective for WHA superposition: s = sum c_i * basis_i."""
    s = c @ basis
    ac = fftconvolve(s, s, mode='full')
    ac_sq = ac ** 2
    max_val = np.max(ac_sq)
    if max_val < 1e-10:
        return 0.0
    kk = min(K, ac_sq.size)
    idx = np.argpartition(ac_sq, -kk)[-kk:]
    top = ac_sq[idx]
    lse = max_val + np.log(np.sum(np.exp(top - max_val)))
    soft_max = np.sqrt(lse)
    sq_int = (np.sum(s)) ** 2
    if sq_int < 1e-10:
        return 1e10
    return soft_max / sq_int

def _wha_gradient(c, basis, K=128):
    """Gradient of WHA soft-max objective w.r.t. coefficients."""
    s = c @ basis
    ac = fftconvolve(s, s, mode='full')
    ac_sq = ac ** 2
    max_val = np.max(ac_sq)
    if max_val < 1e-10:
        return np.zeros_like(c)
    kk = min(K, ac_sq.size)
    idx = np.argpartition(ac_sq, -kk)[-kk:]
    top = ac_sq[idx]
    weights = np.exp(top - max_val)
    weights /= np.sum(weights)
    # d(soft_max)/d(ac) at top-k positions
    dsoft_dac = np.zeros(ac.shape)
    dsoft_dac[idx] = weights * (2.0 * ac[idx]) / np.sqrt(np.sum(weights * top) + max_val)
    # Actually: soft_max = sqrt(lse), d(soft_max)/d(ac_sq) = weights / (2*sqrt(lse))
    lse_val = max_val + np.log(np.sum(np.exp(top - max_val)))
    dsoft_dac_sq = np.zeros(ac.shape)
    dsoft_dac_sq[idx] = weights / (2.0 * np.sqrt(lse_val))
    dsoft_dac = dsoft_dac_sq * 2.0 * ac
    # d(ac)/d(s) = s (full convolution: ac[k] = sum_i s[i]*s[k-i])
    # d(s)/d(c) = basis
    # Chain rule: d(obj)/d(c) = d(obj)/d(ac) @ d(ac)/d(s) @ d(s)/d(c)
    # d(ac)/d(s)[k,j] = s[k-j] + s[j-k] (for full conv of s with itself)
    # This is the autoconvolution gradient: grad_s = 2 * ifft(fft(dsoft_dac) * fft(s))
    # But we need full conv gradient. Let's use the property:
    # ac = fftconvolve(s, s, 'full')
    # d(ac)/d(s) is a convolution operator. The gradient w.r.t. s is:
    # grad_s = 2 * fftconvolve(dsoft_dac, s, mode='same')  -- not quite
    # Actually for full autoconvolution, the gradient w.r.t. s is:
    # grad_s[j] = 2 * sum_k dsoft_dac[k] * s[k-j]  (with appropriate indexing)
    # This is equivalent to: grad_s = 2 * ifft(fft(dsoft_dac) * fft(s, n=2*N-1))
    # Let's compute it via FFT
    n_ac = ac.shape[0]
    n_fft = 1
    while n_fft < n_ac:
        n_fft *= 2
    F_dac = np.fft.rfft(dsoft_dac, n=n_fft)
    F_s = np.fft.rfft(s, n=n_fft)
    grad_s = 2.0 * np.fft.irfft(F_dac * F_s, n=n_fft)[:N]
    # Now d(obj)/d(c) = grad_s @ basis
    grad_c = grad_s @ basis
    # Also account for the 1/(sum s)^2 factor
    sum_s = np.sum(s)
    if abs(sum_s) < 1e-10:
        return grad_c
    obj_val = _wha_objective(c, basis, K)
    # d(obj)/d(c) = d(soft_max)/d(c) / (sum_s)^2 - 2 * soft_max * (sum_s) * d(sum_s)/d(c) / (sum_s)^4
    # = grad_c / (sum_s)^2 - 2 * obj_val * (grad_sum) / (sum_s)^2
    grad_sum = np.sum(basis, axis=1)  # d(sum_s)/d(c) = sum of basis rows
    grad_c = grad_c / (sum_s ** 2) - 2.0 * obj_val * grad_sum / (sum_s ** 2)
    return grad_c
    """True score: max|autoconv| / (sum s)^2."""
    sq_int = (np.sum(s)) ** 2
    if sq_int < 1e-10:
        return np.inf
    ac = fftconvolve(s, s, mode='full')
    max_ac = np.max(np.abs(ac))
    return max_ac / sq_int

def _soft_max_ac(s, temperature=1.0, K=128):
    """Differentiable surrogate for max|autoconv| via top-K log-sum-exp."""
    ac = fftconvolve(s, s, mode='full')
    ac_sq = ac ** 2
    max_val = np.max(ac_sq)
    if max_val < 1e-10:
        return 0.0
    kk = min(K, ac_sq.size)
    idx = np.argpartition(ac_sq, -kk)[-kk:]
    top = ac_sq[idx]
    lse = max_val + np.log(np.sum(np.exp(top - max_val)))
    return np.sqrt(lse)

def _soft_objective(x, temperature=1.0, K=128):
    """Soft objective: log-sum-exp surrogate for max|ac| divided by (sum s)^2."""
    s = np.tanh(x)
    sq_int = (np.sum(s)) ** 2
    if sq_int < 1e-10:
        return 1e10
    return _soft_max_ac(s, temperature, K) / sq_int

def _max_ac_after_flip(cur, cur_ac, cur_sum, j):
    """Vectorized single-flip score update."""
    val = cur[j]
    delta = -2.0 * val
    new_sum = cur_sum + delta
    nsq = new_sum ** 2
    if nsq < 1e-10:
        return np.inf
    new_ac = cur_ac.copy()
    new_ac[j:j+N] += 2.0 * delta * cur
    new_ac[j] += delta * delta
    return float(np.max(np.abs(new_ac)) / nsq)

def construct():
    start_time = time.time()
    deadline = start_time + 143.0  # 7 s safety margin before the 150 s kill

    def _now():
        return time.time()

    best_score = np.inf
    best_s = None
    np.random.seed(42)

    # ---------------- Phase 0: Walsh-Hadamard superposition optimization ----------------
    # WHA basis functions have structurally bounded autoconvolution peaks.
    # A superposition of a few WHA components has a low peak-to-average ratio by construction.
    wha_ranks = [4, 8, 16, 32]
    for rank in wha_ranks:
        if _now() > deadline - 40:
            break
        try:
            basis = _build_walsh_basis(N, rank)
            # Multiple random initializations of coefficients
            for init_it in range(5):
                if _now() > deadline - 40:
                    break
                if init_it == 0:
                    c0 = np.zeros(rank)
                    c0[0] = 1.0  # Start with first non-DC WHA
                elif init_it == 1:
                    c0 = np.random.randn(rank) * 0.5
                elif init_it == 2:
                    c0 = np.random.randn(rank)
                    c0[0] += 2.0  # Emphasize DC-like component
                else:
                    c0 = np.random.randn(rank) * 0.3

                # Gradient descent with line search
                c = c0.copy()
                lr = 0.1
                for gd_it in range(300):
                    if _now() > deadline - 40:
                        break
                    obj = _wha_objective(c, basis, K=64)
                    if not np.isfinite(obj):
                        break
                    grad = _wha_gradient(c, basis, K=64)
                    if not np.all(np.isfinite(grad)):
                        break
                    # Simple gradient step with backtracking line search
                    step = -lr * grad
                    alpha = 1.0
                    for _ in range(10):
                        c_new = c + alpha * step
                        obj_new = _wha_objective(c_new, basis, K=64)
                        if obj_new < obj:
                            break
                        alpha *= 0.5
                    c = c + alpha * step
                    lr *= 0.995

                # Evaluate the resulting sequence
                s_wha = c @ basis
                sc_wha = _score_sequence(s_wha)
                if sc_wha < best_score:
                    best_score = sc_wha
                    best_s = s_wha.copy()

                # Also try sign-quantized version
                s_bin = np.sign(s_wha)
                s_bin[s_bin == 0] = 1.0
                sc_bin = _score_sequence(s_bin)
                if sc_bin < best_score:
                    best_score = sc_bin
                    best_s = s_bin.copy()
        except Exception:
            continue

    # ---------------- Phase 1: gradient-based optimization ----------------
    # Multiple diverse initializations
    n_restarts = 20
    for it in range(n_restarts):
        if _now() > deadline - 30:
            break

        if it < 6:
            x0 = np.random.randn(N) * 0.1
        elif it < 12:
            t = np.arange(N)
            x0 = 0.6 * np.sin(2 * np.pi * t * np.random.uniform(0.05, 0.6)) \
                + np.random.randn(N) * 0.05
        elif it < 16:
            x0 = np.random.randn(N) * 0.3
        else:
            x0 = np.random.randn(N) * 0.15

        temp = 3.0 * (0.6 ** (it / 10.0))
        try:
            result = minimize(
                _soft_objective,
                x0,
                method='L-BFGS-B',
                args=(temp, 128),
                options={'maxiter': 120, 'ftol': 1e-10, 'gtol': 1e-8},
            )
            s_cont = np.tanh(result.x)
            s_bin = np.sign(s_cont)
            s_bin[s_bin == 0] = 1.0

            sc_b = _score_sequence(s_bin)
            if sc_b < best_score:
                best_score = sc_b
                best_s = s_bin.copy()
            sc_c = _score_sequence(s_cont)
            if sc_c < best_score:
                best_score = sc_c
                best_s = s_cont.copy()
        except Exception:
            continue

    # ---------------- Phase 2: discrete local search + SA ----------------
    if best_s is not None:
        cur = best_s.copy()
        cur_ac = fftconvolve(cur, cur, mode='full')
        cur_sum = float(np.sum(cur))
        cur_score = best_score

        # Greedy local search: repeatedly take the best single flip.
        # Use random order to avoid systematic bias
        max_passes = 600
        for _ in range(max_passes):
            if _now() > deadline - 25:
                break

            best_j = -1
            best_js = cur_score
            best_new_sum = cur_sum

            # Random permutation for diversity
            order = np.random.permutation(N)
            for j in order:
                ns = _max_ac_after_flip(cur, cur_ac, cur_sum, j)
                if ns < best_js:
                    best_js = ns
                    best_j = j
                    best_new_sum = cur_sum - 2.0 * cur[j]

            if best_j == -1 or best_js >= cur_score - 1e-12:
                break

            val = cur[best_j]
            delta = -2.0 * val
            cur_ac = cur_ac.copy()
            cur_ac[best_j:best_j+N] += 2.0 * delta * cur
            cur_ac[best_j] += delta * delta
            cur[best_j] = -val
            cur_sum = best_new_sum
            cur_score = best_js

            if cur_score < best_score:
                best_score = cur_score
                best_s = cur.copy()

        # Simulated annealing to escape local minima.
        T_start = 0.05
        sa_steps = 6000
        for it in range(sa_steps):
            if _now() > deadline - 25:
                break
            idx = int(np.random.randint(N))
            new_score = _max_ac_after_flip(cur, cur_ac, cur_sum, idx)
            if new_score == np.inf:
                continue
            d = new_score - cur_score
            T = max(T_start * (0.995 ** it), 1e-6)
            if d < 0 or np.random.rand() < np.exp(-d / T):
                val = cur[idx]
                delta = -2.0 * val
                cur_ac = cur_ac.copy()
                cur_ac[idx:idx+N] += 2.0 * delta * cur
                cur_ac[idx] += delta * delta
                cur[idx] = -val
                cur_sum += delta
                cur_score = new_score
                if cur_score < best_score:
                    best_score = cur_score
                    best_s = cur.copy()

    # ---------------- Phase 3: final high-precision polish ----------------
    if best_s is not None and _now() < deadline - 8:
        x_start = np.arctanh(np.clip(best_s, -0.999, 0.999))
        for temp, iters in ((0.4, 200), (0.15, 200), (0.05, 200)):
            if _now() > deadline - 8:
                break
            try:
                result = minimize(
                    _soft_objective,
                    x_start,
                    method='L-BFGS-B',
                    args=(temp, 128),
                    options={'maxiter': iters, 'ftol': 1e-12, 'gtol': 1e-10},
                )
                s_cont = np.tanh(result.x)
                sc = _score_sequence(s_cont)
                if sc < best_score:
                    best_score = sc
                    best_s = s_cont.copy()
                    x_start = result.x.copy()
            except Exception:
                continue

    return best_s

# EVOLVE-BLOCK-END
