"""
Signed autocorrelation maximum absolute-value variant.

We seek a signed step function f (length N) that minimizes:
    max_k | (f * f)[k] / (sum f)^2 |
where * is circular autoconvolution (or linear, depending on interpretation).

The current approach uses:
1. Gradient-based optimization with a soft-max surrogate
2. Discrete local search (single flips + simulated annealing)
3. Final high-precision polish

Improvements in this version:
- Better analytical gradient for the soft-max surrogate
- Peak-targeted local search that focuses on the worst lag
- Block flips for larger moves
- More aggressive time management
"""

import numpy as np
import time
from scipy.optimize import minimize
from scipy.signal import fftconvolve

N = 400

def _autoconv_full(s):
    """Compute full linear autoconvolution."""
    return fftconvolve(s, s, mode='full')

def compute_raw_value(s):
    """True score: max|autoconv| / (sum s)^2."""
    sq_int = (np.sum(s)) ** 2
    if sq_int < 1e-10:
        return np.inf
    ac = _autoconv_full(s)
    max_ac = np.max(np.abs(ac))
    return float(max_ac / sq_int)

def _soft_max_abs_and_grad(s, temperature=1.0, K=128):
    """
    Differentiable surrogate for max|autoconv| via top-K log-sum-exp.
    Returns (value, gradient w.r.t. s).
    """
    ac = _autoconv_full(s)
    ac_sq = ac ** 2
    max_val = np.max(ac_sq)
    if max_val < 1e-10:
        return 0.0, np.zeros_like(s)
    
    kk = min(K, ac_sq.size)
    idx = np.argpartition(ac_sq, -kk)[-kk:]
    top = ac_sq[idx]
    
    # log-sum-exp of top-K
    shifted = top - max_val
    exps = np.exp(shifted)
    lse = max_val + np.log(np.sum(exps))
    value = np.sqrt(lse)
    
    # Gradient: d/ds of sqrt(LSE)
    # LSE = max_val + log(sum(exp(top - max_val)))
    # dLSE/dac_sq[i] = exp(ac_sq[i] - max_val) / sum(exp(top - max_val)) for i in top, else 0
    # d(ac_sq[i])/ds[j] = 2 * ac[i] * d(ac[i])/ds[j]
    # For autoconvolution ac = f * f, d(ac[k])/ds[j] = f[k-j] + f[j-k] (for linear conv)
    
    # Compute gradient numerically via FFT for efficiency
    # ac = ifft(fft(s)^2) for circular, but we use linear here
    # Let's use the relation: ac[k] = sum_i s[i]*s[k-i]
    # d(ac[k])/ds[j] = s[k-j] + s[j-k] (with appropriate bounds)
    
    # For efficiency, compute gradient using FFT
    # ac = fftconvolve(s, s, 'full')
    # The gradient of ac[k] w.r.t. s[j] is s[k-j] + s[j-k] where defined
    
    # Build weight vector for each lag
    weights = np.zeros(ac_sq.size)
    for i in idx:
        weights[i] = exps[i] / np.sum(exps)
    
    # Gradient of LSE w.r.t. ac_sq: weights
    # Gradient of ac_sq w.r.t. ac: 2*ac
    # So gradient of LSE w.r.t. ac: 2*ac*weights
    
    grad_ac = 2.0 * ac * weights
    
    # Now gradient of ac w.r.t. s
    # ac[k] = sum_i s[i] * s[k-i]
    # d(ac[k])/ds[j] = s[k-j] + s[j-k] (if in range)
    
    # This is a convolution-like operation. Let's compute it efficiently.
    # grad_s[j] = sum_k grad_ac[k] * (s[k-j] + s[j-k])
    #           = sum_k grad_ac[k] * s[k-j] + sum_k grad_ac[k] * s[j-k]
    # Let g = grad_ac, then:
    # sum_k g[k] * s[k-j] = (g * s)[j] (cross-correlation)
    # sum_k g[k] * s[j-k] = (g * s)[j] (same, since g is real)
    # Actually for linear convolution:
    # (g * s)[j] = sum_k g[k] * s[j-k]
    # So grad_s[j] = 2 * (g * s)[j] but we need to be careful with bounds
    
    # Use fftconvolve for the cross-correlation
    # sum_k g[k] * s[k-j] = cross-correlation
    # In numpy: np.correlate(g, s, mode='full')[j + N - 1]
    
    # Actually, let's think more carefully.
    # ac[k] = sum_{i=0}^{N-1} s[i] * s[k-i] where s[i]=0 if i<0 or i>=N
    # d(ac[k])/ds[j] = s[k-j] + s[j-k]
    
    # grad_s[j] = sum_{k=0}^{2N-2} grad_ac[k] * (s[k-j] + s[j-k])
    
    # First term: sum_k grad_ac[k] * s[k-j]
    # Let m = k-j, so k = m+j, sum_m grad_ac[m+j] * s[m]
    # This is like a cross-correlation of grad_ac and s
    
    # Use FFT-based approach
    n_fft = 2 * N
    G = np.fft.rfft(grad_ac, n=n_fft)
    S = np.fft.rfft(s, n=n_fft)
    
    # sum_k grad_ac[k] * s[k-j] for j in [0, N-1]
    # This is the cross-correlation (grad_ac ⋆ s)[j]
    # In frequency domain: ifft(conj(G) * S)
    corr1 = np.fft.irfft(np.conj(G) * S, n=n_fft)[:N]
    
    # sum_k grad_ac[k] * s[j-k] for j in [0, N-1]
    # This is the cross-correlation (s ⋆ grad_ac)[j] = (grad_ac ⋆ s)[N-1-j]... 
    # Actually s[j-k] with k summed: let m = j-k, k = j-m
    # sum_m grad_ac[j-m] * s[m]
    # This is (grad_ac * s)[j] (linear convolution)
    conv = np.fft.irfft(G * S, n=n_fft)[:2*N-1]
    corr2 = conv[:N]
    
    grad_s = 2.0 * (corr1 + corr2)
    
    # d(value)/ds = (1/(2*sqrt(LSE))) * grad_LSE = grad_ac stuff / (2*value)
    # Actually value = sqrt(LSE), d(value) = dLSE / (2*sqrt(LSE))
    # dLSE/ds = grad_s (computed above)
    # So d(value)/ds = grad_s / (2 * value)
    
    if value > 1e-10:
        grad_s = grad_s / (2.0 * value)
    
    return value, grad_s

def _smooth_obj_with_grad(x, temperature=1.0, K=128):
    """
    Smooth objective with gradient.
    x is unconstrained, s = tanh(x).
    Returns (f, g) where f is the objective and g is the gradient w.r.t. x.
    """
    s = np.tanh(x)
    ds_dx = 1.0 - s ** 2
    
    max_abs, grad_s = _soft_max_abs_and_grad(s, temperature, K)
    
    sq_int = (np.sum(s)) ** 2
    if sq_int < 1e-10:
        return 1e10, np.zeros_like(x)
    
    # f = max_abs / sq_int
    f = max_abs / sq_int
    
    # df/ds = (grad_s * sq_int - max_abs * 2 * sum(s) * ds/ds) / sq_int^2
    # df/ds = grad_s / sq_int - max_abs * 2 * sum(s) * (1/sq_int^2) * (ones)
    # df/ds = grad_s / sq_int - 2 * max_abs * sum(s) / sq_int^2
    
    sum_s = np.sum(s)
    grad_f_ds = grad_s / sq_int - 2.0 * max_abs * sum_s / (sq_int ** 2)
    
    # Chain rule: df/dx = df/ds * ds/dx
    g = grad_f_ds * ds_dx
    
    return f, g

def _local_search_flip(cur, cur_ac, cur_sum, cur_score, deadline):
    """
    Greedy local search: repeatedly take the best single flip.
    Returns updated (cur, cur_ac, cur_sum, cur_score).
    """
    max_passes = 2000
    for _ in range(max_passes):
        if time.time() > deadline:
            break
        
        best_j = -1
        best_js = cur_score
        best_new_sum = cur_sum
        
        for j in range(N):
            val = cur[j]
            delta = -2.0 * val
            new_sum = cur_sum + delta
            nsq = new_sum ** 2
            if nsq < 1e-10:
                continue
            new_ac = cur_ac.copy()
            new_ac[j:j+N] += 2.0 * delta * cur
            new_ac[j] += delta * delta
            ns = float(np.max(np.abs(new_ac)) / nsq)
            if ns < best_js:
                best_js = ns
                best_j = j
                best_new_sum = new_sum
        
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
    
    return cur, cur_ac, cur_sum, cur_score

def _local_search_peak_targeted(cur, cur_ac, cur_sum, cur_score, deadline, n_iters=5000):
    """
    Local search that targets the peak of the autoconvolution.
    Tries to reduce the maximum absolute value by flipping bits that contribute most to the peak.
    """
    for it in range(n_iters):
        if time.time() > deadline:
            break
        
        # Find the peak lag
        abs_ac = np.abs(cur_ac)
        peak_idx = np.argmax(abs_ac)
        peak_val = abs_ac[peak_idx]
        
        # The peak is at lag peak_idx, which means ac[peak_idx] = sum_i s[i]*s[peak_idx-i]
        # Flipping s[j] changes ac[k] by:
        #   if k in [j, j+N-1]: ac[k] += 2*delta*s[k-j] + (if k==j) delta^2
        # where delta = -2*s[j]
        
        # We want to flip j to reduce |ac[peak_idx]|
        # ac[peak_idx] changes by 2*delta*s[peak_idx-j] if peak_idx in [j, j+N-1]
        
        # Try a few random flips weighted by their contribution to the peak
        for _ in range(3):
            j = int(np.random.randint(N))
            if peak_idx < j or peak_idx >= j + N:
                continue  # flip doesn't affect the peak
            
            val = cur[j]
            delta = -2.0 * val
            new_sum = cur_sum + delta
            nsq = new_sum ** 2
            if nsq < 1e-10:
                continue
            
            new_ac = cur_ac.copy()
            new_ac[j:j+N] += 2.0 * delta * cur
            new_ac[j] += delta * delta
            ns = float(np.max(np.abs(new_ac)) / nsq)
            
            if ns < cur_score - 1e-12:
                cur_ac = new_ac
                cur[j] = -val
                cur_sum = new_sum
                cur_score = ns
    
    return cur, cur_ac, cur_sum, cur_score

def _local_search_block(cur, cur_ac, cur_sum, cur_score, deadline, block_size=4, n_iters=3000):
    """
    Local search with block flips (flip a contiguous block of bits).
    """
    for it in range(n_iters):
        if time.time() > deadline:
            break
        
        j0 = int(np.random.randint(0, N - block_size + 1))
        block = np.arange(j0, j0 + block_size)
        
        # Try flipping the entire block
        old_vals = cur[block].copy()
        new_vals = -old_vals
        
        delta_total = np.sum(new_vals - old_vals)
        new_sum = cur_sum + delta_total
        nsq = new_sum ** 2
        if nsq < 1e-10:
            continue
        
        # Compute new autoconvolution
        # ac_new = ac + contributions from changed bits
        # For each changed bit j, delta_j = new_vals[j] - old_vals[j]
        # ac_new[k] += sum_j delta_j * (s[k-j] + s[j-k]) approximately
        # More precisely, recompute the affected lags
        
        # For efficiency, just recompute the full autoconvolution for small blocks
        # This is O(N^2) per evaluation but block_size is small
        cur_test = cur.copy()
        cur_test[block] = new_vals
        new_ac = _autoconv_full(cur_test)
        ns = float(np.max(np.abs(new_ac)) / nsq)
        
        if ns < cur_score - 1e-12:
            cur = cur_test
            cur_ac = new_ac
            cur_sum = new_sum
            cur_score = ns
    
    return cur, cur_ac, cur_sum, cur_score

def construct():
    start_time = time.time()
    deadline = start_time + 143.0  # 7 s safety margin before the 150 s kill
    
    def _now():
        return time.time()
    
    best_score = np.inf
    best_s = None
    np.random.seed(42)
    
    # ---------------- Phase 1: gradient-based optimization ----------------
    n_restarts = 25
    for it in range(n_restarts):
        if _now() > deadline - 35:
            break
        
        if it < 8:
            x0 = np.random.randn(N) * 0.1
        elif it < 16:
            t = np.arange(N)
            x0 = 0.6 * np.sin(2 * np.pi * t * np.random.uniform(0.05, 0.6)) \
                + np.random.randn(N) * 0.05
        elif it < 20:
            x0 = np.random.randn(N) * 0.3
        else:
            x0 = np.random.randn(N) * 0.15
        
        temp = 3.0 * (0.6 ** (it / 12.0))
        try:
            result = minimize(
                _smooth_obj_with_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                args=(temp, 128),
                options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8},
            )
            s_cont = np.tanh(result.x)
            s_bin = np.sign(s_cont)
            s_bin[s_bin == 0] = 1.0
            
            sc_b = compute_raw_value(s_bin)
            if sc_b < best_score:
                best_score = sc_b
                best_s = s_bin.copy()
            sc_c = compute_raw_value(s_cont)
            if sc_c < best_score:
                best_score = sc_c
                best_s = s_cont.copy()
        except Exception:
            continue
    
    # ---------------- Phase 2: discrete local search ----------------
    if best_s is not None:
        cur = best_s.copy()
        cur_ac = _autoconv_full(cur)
        cur_sum = float(np.sum(cur))
        cur_score = best_score
        
        # Greedy local search
        cur, cur_ac, cur_sum, cur_score = _local_search_flip(
            cur, cur_ac, cur_sum, cur_score, deadline - 25
        )
        if cur_score < best_score:
            best_score = cur_score
            best_s = cur.copy()
        
        # Peak-targeted local search
        cur, cur_ac, cur_sum, cur_score = _local_search_peak_targeted(
            cur, cur_ac, cur_sum, cur_score, deadline - 20, n_iters=8000
        )
        if cur_score < best_score:
            best_score = cur_score
            best_s = cur.copy()
        
        # Block flip local search
        cur, cur_ac, cur_sum, cur_score = _local_search_block(
            cur, cur_ac, cur_sum, cur_score, deadline - 15, block_size=3, n_iters=5000
        )
        if cur_score < best_score:
            best_score = cur_score
            best_s = cur.copy()
        
        # Simulated annealing to escape local minima
        T_start = 0.05
        sa_steps = 10000
        for it in range(sa_steps):
            if _now() > deadline - 15:
                break
            idx = int(np.random.randint(N))
            val = cur[idx]
            delta = -2.0 * val
            new_sum = cur_sum + delta
            nsq = new_sum ** 2
            if nsq < 1e-10:
                continue
            new_ac = cur_ac.copy()
            new_ac[idx:idx+N] += 2.0 * delta * cur
            new_ac[idx] += delta * delta
            new_score = float(np.max(np.abs(new_ac)) / nsq)
            
            d = new_score - cur_score
            T = max(T_start * (0.998 ** it), 1e-6)
            if d < 0 or np.random.rand() < np.exp(-d / T):
                cur_ac = new_ac
                cur[idx] = -val
                cur_sum = new_sum
                cur_score = new_score
                if cur_score < best_score:
                    best_score = cur_score
                    best_s = cur.copy()
    
    # ---------------- Phase 3: final high-precision polish ----------------
    if best_s is not None and _now() < deadline - 8:
        x_start = np.arctanh(np.clip(best_s, -0.999, 0.999))
        for temp, iters in ((0.4, 200), (0.15, 200), (0.05, 300)):
            if _now() > deadline - 8:
                break
            try:
                result = minimize(
                    _smooth_obj_with_grad,
                    x_start,
                    method='L-BFGS-B',
                    jac=True,
                    args=(temp, 128),
                    options={'maxiter': iters, 'ftol': 1e-12, 'gtol': 1e-10},
                )
                s_cont = np.tanh(result.x)
                sc = compute_raw_value(s_cont)
                if sc < best_score:
                    best_score = sc
                    best_s = s_cont.copy()
                    x_start = result.x.copy()
            except Exception:
                continue
    
    return best_s

# EVOLVE-BLOCK-END
