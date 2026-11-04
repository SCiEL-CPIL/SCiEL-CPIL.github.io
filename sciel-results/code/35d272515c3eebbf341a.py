"""Rewritten program: continuous-height optimization via gradient descent on smoothed max autoconv."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = _START_TIME + 115  # safety margin before 120s limit


def _fft_autocorr(x):
    """Compute full linear autocorrelation of 1D array via FFT."""
    n = len(x)
    size = 1
    while size < 2 * n - 1:
        size <<= 1
    X = np.fft.rfft(x, n=size)
    acf = np.fft.irfft(X * np.conj(X), n=size)
    return acf[:2 * n - 1]


def _smooth_max(x, t=1.0):
    """Softplus-based smoothed max."""
    if t < 1e-4:
        return np.max(x)
    m = np.max(x)
    return m + t * np.log(np.mean(np.exp((x - m) / t)))


def _objective_and_grad(h, tau=1.0):
    """
    Objective: smoothed_max(|autoconv(h)|) / (sum(h))^2
    Returns (value, grad)
    """
    # Normalize so sum(h) = 1 to fix the scale (denominator becomes 1)
    s = np.sum(h)
    if abs(s) < 1e-12:
        return 1e10, np.zeros_like(h)
    x = h / s
    acf = _fft_autocorr(x)
    abs_acf = np.abs(acf)
    val = _smooth_max(abs_acf, tau)
    
    # Gradient of smoothed max
    m = np.max(abs_acf)
    e = np.exp((abs_acf - m) / tau)
    e /= np.mean(e)
    g_abs = e / len(abs_acf)  # d(smooth_max)/d(abs_acf_i)
    
    # Chain rule: abs_acf_i = |acf_i|
    # d|acf_i|/dacf_i = sign(acf_i)
    sign_acf = np.sign(acf)
    g_acf = g_abs * sign_acf
    
    # acf = IFFT(|FFT(x)|^2)
    # This is a circular correlation. For linear, we padded.
    # Grad w.r.t. x: 
    # acf_k = sum_j x_j x_{j+k}
    # d acf_k / d x_i = x_{i-k} + x_{i+k} (for valid indices, 0 otherwise)
    # This is equivalent to: grad_x = IFFT( sign_acf * conj(FFT(x)) )? 
    # Let's use the FFT domain directly for efficiency.
    # acf = IFFT( X * conj(X) )
    # d acf / d X = conj(X), d acf / d conj(X) = X
    # But X depends on x.
    # It's easier to use the time-domain convolution gradient:
    # g_x = convolve(g_acf, x) + convolve(x, g_acf_reversed)?
    # Actually, autoconv is (x * x)[k] = sum_i x[i] x[k-i]
    # d/dx_j (x*x)[k] = x[k-j] + x[j-k]
    # So g_x[j] = sum_k g_acf[k] (x[k-j] + x[j-k])
    # This is 2 * (g_acf * x)[j] if we treat it as correlation?
    # Let's just use FFT for the gradient too.
    # In frequency domain: 
    # Let A = X * conj(X). acf = ifft(A).
    # L = sum_k g_acf[k] acf[k] = sum_k g_acf[k] ifft(A)[k]
    # = sum_m A[m] ifft(g_acf)[m] (by Parseval/definition)
    # Let G = fft(g_acf) (zero-padded to size)
    # Then L = sum_m A[m] conj(G[m]) ? No, depends on conventions.
    # Simpler: 
    # grad_x = ifft( 2 * conj(X) * fft(g_acf) ) ? 
    # Let's stick to time domain for correctness, N=400 is small enough.
    # g_x[j] = sum_k g_acf[k] * (x[k-j] if 0<=k-j<N else 0) + (x[j-k] if 0<=j-k<N else 0)
    # This is effectively: g_x = correlate(g_acf, x) + correlate(g_acf_reversed, x) ?
    # Actually, since g_acf is real, and autoconv is symmetric...
    # Let's use scipy.signal or just manual FFT trick.
    
    size = 1
    while size < 2 * N - 1:
        size <<= 1
    X_padded = np.fft.rfft(x, n=size)
    G_padded = np.fft.rfft(g_acf, n=size)
    # The gradient of (x*x) w.r.t x in frequency domain:
    # if y = x * x (linear conv), Y = X*conj(X) (circular, padded)
    # dy/dx = 2 * ifft(conj(X) * G)  (complex) -> take real part
    grad_x = np.fft.irfft(2.0 * np.conj(X_padded) * G_padded, n=size)[:N]
    
    # Now chain rule for h -> x = h/s
    # x = h/s
    # dx/dh_j = (1/s) delta_{ij} - (h_i/s^2)
    # grad_h = (1/s) * grad_x - (h/s^2) * sum(grad_x)
    grad_h = (1.0 / s) * grad_x - (h / (s * s)) * np.sum(grad_x)
    
    return val, grad_h


def _line_search(h, grad, lr, max_iter=10):
    """Simple backtracking line search."""
    f0, _ = _objective_and_grad(h)
    if not np.isfinite(f0):
        return h, f0
    step = lr
    for _ in range(max_iter):
        h_new = h - step * grad
        f_new, _ = _objective_and_grad(h_new)
        if np.isfinite(f_new) and f_new < f0:
            return h_new, f_new
        step *= 0.5
    return h, f0


def construct():
    # Initialize with random continuous values
    rng = np.random.default_rng(42)
    h = rng.uniform(-1, 1, size=N)
    
    # Initial objective
    f, grad = _objective_and_grad(h)
    if not np.isfinite(f):
        return np.ones(N)
    
    best_h = h.copy()
    best_f = f
    
    lr = 1e-3
    tau = 1.0
    
    while time.time() < _DEADLINE:
        # Anneal temperature
        tau = max(0.01, tau * 0.99)
        
        h, f = _line_search(h, grad, lr)
        
        if f < best_f:
            best_f = f
            best_h = h.copy()
        
        _, grad = _objective_and_grad(h, tau)
        
        # Adapt learning rate
        if f < best_f * 0.99:
            lr *= 1.2
        else:
            lr *= 0.8
        lr = np.clip(lr, 1e-6, 1e-1)
        
        # If gradient is very small, break
        if np.linalg.norm(grad) < 1e-6:
            break
    
    # Final cleanup: renormalize to sum=1 is not strictly needed as objective handles it,
    # but the task asks for "signed step heights". The score is the raw value.
    # The evaluator computes max_abs_autocorr / sum^2.
    # Returning best_h is correct.
    
    # Ensure finite
    if not np.all(np.isfinite(best_h)):
        return np.ones(N)
        
    return best_h
# EVOLVE-BLOCK-END
