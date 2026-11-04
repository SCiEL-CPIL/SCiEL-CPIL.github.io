"""Second autocorrelation inequality -- COMPLEX-SPACE OPTIMIZATION.

Parameterizes h = |g|^2 where g is complex (real + imaginary parts).
This removes non-negativity constraints and enables smooth optimization.
Uses soft-max for the autoconvolution peak and L-BFGS-B for polishing.
"""

import time
import math
import numpy as np
from scipy.optimize import minimize

N = 50
DH = 1.0 / 20.0
OFFSET = 10  # 1/2 / DH

_DEADLINE = 150.0
_SAFETY = 18.0
_start = time.time()


def _elapsed():
    return time.time() - _start


def _time_left():
    return _DEADLINE - _SAFETY - _elapsed()


def _ratio_from_h(h):
    """R = ||h||_2^2 / (||h||_1 * ||h*h||_inf) from a precomputed nonneg h."""
    h = np.maximum(h, 0.0)
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 1e-300 or l2sq <= 1e-300:
        return 0.0
    full = np.convolve(h, h, mode='full')
    ac = DH * full[OFFSET:OFFSET + N]
    acmax = float(np.max(ac))
    if acmax <= 1e-300:
        return 0.0
    return l2sq / (l1 * acmax)


def _logR_from_g(g_flat, smooth_beta=50.0):
    """Compute log R from complex parameterization g (flattened).
    
    g_flat has shape (2*N,): first N are real parts, next N are imaginary parts.
    h = |g|^2 = real^2 + imag^2
    """
    g_real = g_flat[:N]
    g_imag = g_flat[N:]
    h = g_real**2 + g_imag**2
    
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 1e-300 or l2sq <= 1e-300:
        return -1e30
    
    full = np.convolve(h, h, mode='full')
    ac = DH * full[OFFSET:OFFSET + N]
    ac_pos = np.maximum(ac, 1e-300)
    ac_max = np.max(ac_pos)
    if ac_max <= 1e-300:
        return -1e30
    
    # Soft-max for smooth log(max)
    inv_beta = 1.0 / smooth_beta
    shifted = ac_pos - ac_max
    exp_shifted = np.exp(shifted * inv_beta)
    sum_exp = np.sum(exp_shifted)
    if sum_exp <= 1e-300:
        return -1e30
    log_sum_exp = ac_max + inv_beta * math.log(sum_exp)
    
    R = l2sq / (l1 * math.exp(log_sum_exp))
    if R <= 1e-300:
        return -1e30
    return math.log(R)


def _logR_and_grad_from_g(g_flat, smooth_beta=50.0):
    """Analytic gradient of log R w.r.t. g_flat using chain rule.
    
    h = |g|^2 = gr^2 + gi^2
    d h_k / d gr_k = 2 * gr_k
    d h_k / d gi_k = 2 * gi_k
    
    Then chain through d logR / d h_k.
    """
    g_real = g_flat[:N]
    g_imag = g_flat[N:]
    h = g_real**2 + g_imag**2
    
    l2sq = float(np.dot(h, h))
    l1 = float(np.sum(h))
    if l1 <= 1e-300 or l2sq <= 1e-300:
        return -1e30, np.zeros(2 * N)
    
    full = np.convolve(h, h, mode='full')
    ac = DH * full[OFFSET:OFFSET + N]
    ac_pos = np.maximum(ac, 1e-300)
    ac_max = np.max(ac_pos)
    if ac_max <= 1e-300:
        return -1e30, np.zeros(2 * N)
    
    # Soft-max
    inv_beta = 1.0 / smooth_beta
    shifted = ac_pos - ac_max
    exp_shifted = np.exp(shifted * inv_beta)
    sum_exp = np.sum(exp_shifted)
    if sum_exp <= 1e-300:
        return -1e30, np.zeros(2 * N)
    log_sum_exp = ac_max + inv_beta * math.log(sum_exp)
    
    R = l2sq / (l1 * math.exp(log_sum_exp))
    if R <= 1e-300:
        return -1e30, np.zeros(2 * N)
    logR = math.log(R)
    
    # d logR / d h_k
    softmax_weights = exp_shifted / sum_exp
    
    # d log(acmax_smooth) / d h_k = (1/beta) * sum_i softmax_weights[i] * d ac_i / d h_k
    # d ac_i / d h_k = 2 * DH * h_{i-k} (if valid)
    dlog_acmax_dh = np.zeros(N)
    for k in range(N):
        dlog_acmax_dh[k] = (2.0 * DH * inv_beta) * float(np.dot(softmax_weights[k:], h[:N-k]))
    
    dlogR_dh = (2.0 * h / l2sq) - (1.0 / l1) - dlog_acmax_dh
    
    # Chain: d logR / d gr_k = dlogR_dh[k] * 2 * gr_k
    #         d logR / d gi_k = dlogR_dh[k] * 2 * gi_k
    grad_real = 2.0 * g_real * dlogR_dh
    grad_imag = 2.0 * g_imag * dlogR_dh
    
    grad = np.concatenate([grad_real, grad_imag])
    
    if not np.all(np.isfinite(grad)):
        grad = np.zeros(2 * N)
        eps = 1e-6
        for p in range(2 * N):
            gp = g_flat.copy(); gp[p] += eps
            gm = g_flat.copy(); gm[p] -= eps
            lp = _logR_from_g(gp, smooth_beta)
            lm = _logR_from_g(gm, smooth_beta)
            grad[p] = (lp - lm) / (2.0 * eps)
    
    return logR, grad


def _neg_logR_from_g(g_flat, smooth_beta=50.0):
    return -_logR_from_g(g_flat, smooth_beta)


def _neg_logR_jac_from_g(g_flat, smooth_beta=50.0):
    v, g = _logR_and_grad_from_g(g_flat, smooth_beta)
    return v, -g


def _h_from_g(g_flat):
    """Convert complex parameterization to height vector."""
    g_real = g_flat[:N]
    g_imag = g_flat[N:]
    h = g_real**2 + g_imag**2
    return np.maximum(h, 0.0)


def _polish_complex(g0, budget, smooth_beta=50.0):
    """Local polish in complex space using L-BFGS-B."""
    best_g = np.array(g0, dtype=float)
    best_log = _logR_from_g(best_g, smooth_beta)
    t0 = time.time()
    
    try:
        res = minimize(
            lambda g: _neg_logR_from_g(g, smooth_beta),
            best_g,
            method='L-BFGS-B',
            jac=lambda g: _neg_logR_jac_from_g(g, smooth_beta),
            options={'maxiter': 300, 'maxfun': 3000, 'ftol': 1e-10,
                     'gtol': 1e-8, 'disp': False},
            callback=lambda x: (np.nan if (time.time() - t0) > budget else x),
        )
        if res.x is not None:
            rv = _logR_from_g(res.x, smooth_beta)
            if rv > best_log:
                best_log = rv
                best_g = res.x.copy()
    except Exception:
        pass
    
    h_best = _h_from_g(best_g)
    R_best = _ratio_from_h(h_best)
    return best_g, best_log, R_best


def _generate_starts():
    """Generate diverse starting points in complex parameter space."""
    rng = np.random.default_rng(20240501)
    starts = []
    
    # (a) Uniform h -> g = sqrt(1) real, 0 imag
    g_uniform = np.concatenate([np.ones(N), np.zeros(N)])
    starts.append(g_uniform)
    
    # (b) Gaussian-like shapes
    x = np.linspace(-1, 1, N)
    for sigma in [0.2, 0.5, 1.0, 2.0]:
        h_shape = np.exp(-x**2 / (2 * sigma**2))
        g_real = np.sqrt(h_shape)
        g_imag = np.zeros(N)
        starts.append(np.concatenate([g_real, g_imag]))
    
    # (c) Bimodal
    for sep in [0.3, 0.5, 0.7]:
        h_shape = np.exp(-((x - sep)**2) / 0.1) + np.exp(-((x + sep)**2) / 0.1)
        g_real = np.sqrt(h_shape)
        g_imag = np.zeros(N)
        starts.append(np.concatenate([g_real, g_imag]))
    
    # (d) Triangle / tent
    h_tri = 1 - np.abs(x)
    h_tri = np.maximum(h_tri, 0)
    g_real = np.sqrt(h_tri)
    g_imag = np.zeros(N)
    starts.append(np.concatenate([g_real, g_imag]))
    
    # (e) Cosine window
    h_cos = 0.5 + 0.5 * np.cos(np.pi * x)
    g_real = np.sqrt(h_cos)
    g_imag = np.zeros(N)
    starts.append(np.concatenate([g_real, g_imag]))
    
    # (f) Random smooth
    for seed in [1, 2, 3, 4, 5, 6, 7, 8]:
        rng_local = np.random.RandomState(seed)
        h_rand = np.abs(rng_local.randn(N))
        kernel = np.ones(5) / 5
        h_smooth = np.convolve(h_rand, kernel, mode='same')
        g_real = np.sqrt(h_smooth)
        g_imag = np.zeros(N)
        starts.append(np.concatenate([g_real, g_imag]))
    
    # (g) Peak in center
    h_center = np.zeros(N)
    h_center[N//2 - 2:N//2 + 3] = 1.0
    g_real = np.sqrt(h_center)
    g_imag = np.zeros(N)
    starts.append(np.concatenate([g_real, g_imag]))
    
    # (h) Peak at edges
    h_edge = np.zeros(N)
    h_edge[0:3] = 1.0
    h_edge[-3:] = 1.0
    g_real = np.sqrt(h_edge)
    g_imag = np.zeros(N)
    starts.append(np.concatenate([g_real, g_imag]))
    
    # (i) Alternating
    h_alt = np.ones(N)
    h_alt[::2] = 0.5
    g_real = np.sqrt(h_alt)
    g_imag = np.zeros(N)
    starts.append(np.concatenate([g_real, g_imag]))
    
    # (j) Linear ramp
    for direction in [1, -1]:
        h_ramp = np.linspace(0.1, 1.0, N) if direction == 1 else np.linspace(1.0, 0.1, N)
        g_real = np.sqrt(h_ramp)
        g_imag = np.zeros(N)
        starts.append(np.concatenate([g_real, g_imag]))
    
    # (k) DFT-based: h = |FFT(z)|^2
    for seed in [10, 20, 30, 40, 50]:
        rng_local = np.random.RandomState(seed)
        z = rng_local.randn(N) + 1j * rng_local.randn(N)
        H = np.fft.fft(z)
        h_dft = np.abs(H)**2
        h_dft = h_dft / np.max(h_dft)
        g_real = np.sqrt(h_dft)
        g_imag = np.zeros(N)
        starts.append(np.concatenate([g_real, g_imag]))
    
    # (l) Structured patterns
    h_sin = np.sin(np.pi * (x + 1) / 2)**2
    g_real = np.sqrt(h_sin)
    g_imag = np.zeros(N)
    starts.append(np.concatenate([g_real, g_imag]))
    
    h_cos2 = np.cos(np.pi * x / 2)**2
    g_real = np.sqrt(h_cos2)
    g_imag = np.zeros(N)
    starts.append(np.concatenate([g_real, g_imag]))
    
    # (m) Random complex g
    for _ in range(30):
        g_real = rng.normal(0.0, 0.5, size=N)
        g_imag = rng.normal(0.0, 0.5, size=N)
        starts.append(np.concatenate([g_real, g_imag]))
    
    # (n) Multi-peak with valleys
    for npeaks in [3, 4, 5]:
        for trial in range(10):
            h_shape = np.zeros(N)
            peak_positions = np.linspace(0, N-1, npeaks).astype(int)
            for pi in peak_positions:
                width = rng.uniform(2, 5)
                h_shape += np.exp(-0.5 * ((np.arange(N) - pi) / width)**2)
            g_real = np.sqrt(h_shape)
            g_imag = np.zeros(N)
            starts.append(np.concatenate([g_real, g_imag]))
    
    return starts


def construct():
    global _start
    _start = time.time()
    
    best_g = None
    best_log = -1e30
    best_R = 0.0
    
    starts = _generate_starts()
    
    # Score all starts, sort best-first
    scored = []
    for g0 in starts:
        v = _logR_from_g(g0)
        scored.append((v, g0))
        if v > best_log:
            best_log = v
            best_g = g0.copy()
            h_test = _h_from_g(g0)
            best_R = _ratio_from_h(h_test)
    
    scored.sort(key=lambda t: -t[0])
    
    # Polish top starts within budget
    top_k = min(len(scored), 30)
    for k in range(top_k):
        v0, g0 = scored[k]
        if _time_left() <= 10.0:
            break
        if v0 < best_log - 2.0 and best_log > -1e29:
            continue
        
        budget = min(4.0, max(0.5, _time_left() - 15.0))
        try:
            g_opt, v_opt, R_opt = _polish_complex(g0, budget)
            if v_opt > best_log:
                best_log = v_opt
                best_g = g_opt.copy()
                best_R = R_opt
        except Exception:
            pass
        
        if _time_left() <= 30.0:
            break
    
    # Additional polish with different smooth_beta values
    if best_g is not None and _time_left() > 20.0:
        for beta in [20.0, 100.0, 200.0]:
            if _time_left() <= 15.0:
                break
            budget = min(3.0, max(0.5, _time_left() - 18.0))
            try:
                g_opt, v_opt, R_opt = _polish_complex(best_g, budget, smooth_beta=beta)
                if v_opt > best_log:
                    best_log = v_opt
                    best_g = g_opt.copy()
                    best_R = R_opt
            except Exception:
                pass
    
    # Direct optimization on heights for final polish
    if best_g is not None and _time_left() > 25.0:
        h_init = _h_from_g(best_g)
        if float(np.sum(h_init)) > 0:
            try:
                def _direct_logR(h, beta=50.0):
                    h = np.maximum(h, 0.0)
                    l2sq = float(np.dot(h, h))
                    l1 = float(np.sum(h))
                    if l1 <= 1e-300 or l2sq <= 1e-300:
                        return -1e30
                    full = np.convolve(h, h, mode='full')
                    ac = DH * full[OFFSET:OFFSET + N]
                    ac_pos = np.maximum(ac, 1e-300)
                    ac_max = np.max(ac_pos)
                    if ac_max <= 1e-300:
                        return -1e30
                    inv_beta = 1.0 / beta
                    shifted = ac_pos - ac_max
                    exp_shifted = np.exp(shifted * inv_beta)
                    sum_exp = np.sum(exp_shifted)
                    if sum_exp <= 1e-300:
                        return -1e30
                    log_sum_exp = ac_max + inv_beta * math.log(sum_exp)
                    R = l2sq / (l1 * math.exp(log_sum_exp))
                    if R <= 1e-300:
                        return -1e30
                    return math.log(R)
                
                def _direct_grad(h, beta=50.0):
                    h_pos = np.maximum(h, 0.0)
                    l2sq = float(np.dot(h_pos, h_pos))
                    l1 = float(np.sum(h_pos))
                    if l1 <= 1e-300 or l2sq <= 1e-300:
                        return np.zeros(N)
                    full = np.convolve(h_pos, h_pos, mode='full')
                    ac = DH * full[OFFSET:OFFSET + N]
                    ac_pos = np.maximum(ac, 1e-300)
                    ac_max = np.max(ac_pos)
                    if ac_max <= 1e-300:
                        return np.zeros(N)
                    inv_beta = 1.0 / beta
                    shifted = ac_pos - ac_max
                    exp_shifted = np.exp(shifted * inv_beta)
                    sum_exp = np.sum(exp_shifted)
                    if sum_exp <= 1e-300:
                        return np.zeros(N)
                    softmax_weights = exp_shifted / sum_exp
                    dlog_acmax_dh = np.zeros(N)
                    for k in range(N):
                        dlog_acmax_dh[k] = (2.0 * DH * inv_beta) * float(np.dot(softmax_weights[k:], h_pos[:N-k]))
                    grad = (2.0 * h_pos / l2sq) - (1.0 / l1) - dlog_acmax_dh
                    grad[h <= 1e-10] = 0.0
                    return grad
                
                bounds_direct = [(0.0, None)] * N
                t0_direct = time.time()
                budget_direct = min(_time_left() - 25.0, 25.0)
                
                res_direct = minimize(
                    lambda h: -_direct_logR(h, 50.0), h_init,
                    method='L-BFGS-B',
                    jac=lambda h: -_direct_grad(h, 50.0),
                    bounds=bounds_direct,
                    options={'maxiter': 200, 'maxfun': 2000, 'ftol': 1e-12,
                             'gtol': 1e-9, 'disp': False},
                    callback=lambda x: (np.nan if (time.time() - t0_direct) > budget_direct else x),
                )
                if res_direct.x is not None:
                    h_direct = np.maximum(res_direct.x, 0.0)
                    R_direct = _ratio_from_h(h_direct)
                    if R_direct > best_R:
                        best_R = R_direct
                        best_log = math.log(R_direct) if R_direct > 0 else -1e30
            except Exception:
                pass
    
    # Materialize the height vector
    if best_g is None:
        h = np.ones(N)
    else:
        h = _h_from_g(best_g)
        if not np.all(np.isfinite(h)) or float(np.sum(h)) <= 0.0:
            h = np.ones(N)
    h = np.maximum(h, 0.0)
    return [float(x) for x in h]


if __name__ == '__main__':
    t0 = time.time()
    h = construct()
    print("N =", len(h))
    print("ratio =", _ratio_from_h(h))
    print("time =", time.time() - t0)
