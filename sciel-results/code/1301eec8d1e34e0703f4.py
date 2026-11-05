import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.fft import dct, idct
import time

N = 400

def _evaluate(h):
    s = np.sum(h)
    if abs(s) < 1e-10 or not np.all(np.isfinite(h)):
        return np.inf
    conv = np.convolve(h, h, mode='full')
    return abs(2.0 * N * np.max(conv)) / (s * s)

def _logsumexp(x, axis=None):
    m = np.max(x, axis=axis, keepdims=True)
    if np.any(np.isinf(m)):
        return m
    out = np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True)) + m
    return np.squeeze(out, axis=axis) if axis is not None else out

def _grad_autoconv(w, h):
    L = 2 * N
    w_pad = np.zeros(L)
    w_pad[:len(w)] = w
    h_pad = np.zeros(L)
    h_pad[:N] = h
    W = np.fft.fft(w_pad, n=L)
    H = np.fft.fft(h_pad, n=L)
    cross = np.fft.ifft(W * np.conj(H)).real
    return 2.0 * cross[:N]

def _soft_max_and_weights(conv, beta):
    m = np.max(conv)
    if m <= 0:
        return m, np.zeros_like(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -500, 500)
    exp_shifted = np.exp(shifted)
    weights = exp_shifted / np.sum(exp_shifted)
    soft_max = m + np.log(np.sum(exp_shifted)) / beta
    return soft_max, weights

def _objective_soft(h, beta=50.0):
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = np.convolve(h, h, mode='full')
    soft_max, _ = _soft_max_and_weights(conv, beta)
    return 2.0 * N * soft_max / (s * s)

def _objective_soft_grad(h, beta=50.0):
    s = np.sum(h)
    if abs(s) < 1e-10:
        return np.zeros_like(h)
    conv = np.convolve(h, h, mode='full')
    soft_max, weights = _soft_max_and_weights(conv, beta)
    if np.max(weights) < 1e-15:
        return np.zeros_like(h)
    grad_sm = _grad_autoconv(weights, h)
    return 2.0 * N * (grad_sm * s - 2.0 * soft_max) / (s * s * s)

def _spectral_penalty(h, lam=0.1):
    dct_coeffs = dct(h, type=2, norm='ortho')
    return lam * np.sum(np.abs(dct_coeffs)**4) / N

def _spectral_penalty_grad(h, lam=0.1):
    dct_coeffs = dct(h, type=2, norm='ortho')
    return 4.0 * lam / N * idct(dct_coeffs**3, type=2, norm='ortho')

def _combined_obj(h, beta=50.0, lam=0.05):
    return _objective_soft(h, beta) + _spectral_penalty(h, lam)

def _combined_grad(h, beta=50.0, lam=0.05):
    return _objective_soft_grad(h, beta) + _spectral_penalty_grad(h, lam)

def _refine(h, deadline):
    h_curr = h.copy()
    
    if time.time() < deadline - 10.0:
        try:
            res = minimize(_combined_obj, h_curr, method='L-BFGS-B',
                           args=(30.0, 0.1), jac=_combined_grad,
                           options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-7})
            h_curr = res.x
        except Exception:
            pass
    
    if time.time() < deadline - 5.0:
        try:
            res = minimize(_combined_obj, h_curr, method='L-BFGS-B',
                           args=(80.0, 0.01), jac=_combined_grad,
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-9})
            h_curr = res.x
        except Exception:
            pass
    
    if time.time() < deadline - 2.0:
        try:
            res = minimize(_objective_soft, h_curr, method='L-BFGS-B',
                           args=(150.0,), jac=_objective_soft_grad,
                           options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
            h_curr = res.x
        except Exception:
            pass
            
    return h_curr

def _scale_optimal(h):
    """Find the optimal scaling factor alpha that minimizes the objective for the shape h."""
    conv = np.convolve(h, h, mode='full')
    max_conv = np.max(conv)
    if max_conv <= 0:
        return 1.0
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1.0
    
    # value(alpha) = |2*N*alpha^2*max_conv / (alpha*s)^2| = 2*N*max_conv/s^2 (if alpha>0)
    # Wait, the objective is scale-invariant!
    # value = |2 * N * max(alpha^2 * conv) / (alpha * sum(h))^2|
    # = |2 * N * alpha^2 * max(conv) / (alpha^2 * sum(h)^2)|
    # = 2 * N * max(conv) / sum(h)^2
    # So scaling doesn't change the value. The value is completely scale-invariant.
    return 1.0

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    rng = np.random.default_rng(12345)
    best_h = np.ones(N)
    best_val = _evaluate(best_h)
    
    i = np.arange(N)
    initializations = [np.ones(N)]
    
    for k in [1, 2, 3, 4, 5, 6, 8, 10]:
        for a in [0.3, 0.7, 1.5]:
            initializations.append(a + np.sin(2*np.pi*k*i/N))
            initializations.append(a + np.cos(2*np.pi*k*i/N))
    
    for period in [2, 3, 4, 5, 7, 11, 13]:
        h0 = np.where((i % period) < period // 2, 1.0, -1.0)
        h0 += 0.15 * rng.standard_normal(N)
        if abs(np.sum(h0)) > 0.5:
            initializations.append(h0)
    
    for _ in range(8):
        h0 = rng.standard_normal(N)
        kernel = np.ones(7) / 7.0
        h0 = np.convolve(h0, kernel, mode='same')
        h0 += 1.0
        initializations.append(h0)
    
    for _ in range(8):
        freq = rng.uniform(0.1, 3.0)
        phase = rng.uniform(0, 2*np.pi)
        h0 = np.sin(2*np.pi*freq*i/N + phase)
        h0 += 1.0
        initializations.append(h0)
    
    for _ in range(5):
        h0 = rng.standard_normal(N)
        # Apply a Gaussian blur
        sigma = rng.uniform(2, 10)
        x = np.arange(-20, 21)
        kernel = np.exp(-0.5 * (x/sigma)**2)
        kernel /= np.sum(kernel)
        h0 = np.convolve(h0, kernel, mode='same')
        h0 += 1.0
        initializations.append(h0)
    
    # Optimize each initialization
    for h0 in initializations:
        if time.time() > deadline - 15.0:
            break
        h_refined = _refine(h0, deadline)
        val = _evaluate(h_refined)
        if val < best_val:
            best_val = val
            best_h = h_refined.copy()
    
    if time.time() < deadline - 1.0:
        h_final = _refine(best_h, deadline)
        val_final = _evaluate(h_final)
        if val_final < best_val:
            best_h = h_final
    
    return best_h
