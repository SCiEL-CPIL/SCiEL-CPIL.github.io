"""Minimize the normalized max autoconvolution of a nonnegative 600-step function on [-1/4,1/4]."""
import time
import numpy as np
from scipy.optimize import minimize

N = 600
HALF = N // 2
DT = 0.5 / N

# EVOLVE-BLOCK-START

def _fft_peak(a):
    """Peak of normalized autoconvolution (a * a) / ||a||_2^2."""
    a = np.asarray(a, dtype=float)
    nrm = np.sqrt(np.dot(a, a))
    if nrm <= 0:
        return 0.0
    conv = np.fft.irfft(np.fft.rfft(a, 2*N)**2, 2*N)[:2*N-1]
    return float(np.max(conv) / (nrm * nrm))

def _fourier(a):
    """Nonnegative, even function from Fourier parameters a = [c0, c1..c_{M-1}]."""
    M = len(a)
    idx = np.arange(N)
    m = idx - HALF
    x = m * DT
    y = np.zeros(N)
    y += a[0]
    for k in range(1, M):
        y += a[k] * np.cos(2.0 * np.pi * k * x)
    return y

def _fourier_grad(a):
    M = len(a)
    idx = np.arange(N)
    m = idx - HALF
    x = m * DT
    y = np.zeros(N)
    y += a[0]
    g = np.zeros(M)
    g[0] = 1.0
    for k in range(1, M):
        c = np.cos(2.0 * np.pi * k * x)
        y += a[k] * c
        g[k] = c
    return y, g

def _fourier_obj(a):
    y, g = _fourier_grad(a)
    y = np.maximum(y, 0.0)
    nrm = np.dot(y, y)
    if nrm <= 0:
        return 1e30, np.zeros_like(a)
    val = 1.0 / nrm
    grad = np.zeros_like(a)
    for k in range(len(a)):
        grad[k] = -2.0 * np.dot(y, y * g[k]) / nrm
    return val, grad

def _fourier_optimize(deadline):
    M = 16
    a0 = np.zeros(M)
    a0[0] = 1.0
    best_a = a0.copy()
    best_val = _fourier_obj(a0)[0]
    for _ in range(3):
        if time.time() > deadline - 20:
            break
        try:
            res = minimize(_fourier_obj, best_a, jac=True, method="L-BFGS-B",
                           options={"maxiter": 400, "maxls": 40, "ftol": 1e-12, "gtol": 1e-8})
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_a = res.x.copy()
        except Exception:
            break
    y = _fourier(best_a)
    y = np.maximum(y, 0.0)
    return y

def _direct_polish(y, deadline, steps=200):
    """Direct minimization of the autoconvolution peak over 600 nonnegative vars."""
    best = y.copy()
    best_score = _fft_peak(best)
    step = max(1e-3, 1e-3 * best_score)
    for it in range(steps):
        if time.time() > deadline - 15:
            break
        s = _fft_peak(best)
        grad = np.zeros(N)
        for j in range(N):
            save = best[j]
            best[j] = save + step
            gp = _fft_peak(best)
            best[j] = save - step
            if best[j] < 0:
                best[j] = 0.0
            gm = _fft_peak(best)
            best[j] = save
            grad[j] = (gp - gm) / (2.0 * step)
        # projected gradient step
        lr = step / (np.linalg.norm(grad) + 1e-12)
        cand = best - lr * grad
        cand = np.maximum(cand, 0.0)
        cs = _fft_peak(cand)
        if cs < s:
            best = cand
            best_score = cs
        else:
            step *= 0.7
    return best, best_score

def _double_box_optimize(deadline):
    """
    Optimize a 'double-box' (hollow) structure.
    f(x) = A * (1_{|x| <= R} - B * 1_{|x| <= r})
    Parameters: [A, R, B, r]
    Constraints: A > 0, B >= 0, 0 <= r <= R <= 0.25.
    """
    def param_to_signal(params):
        A, R, B, r = params
        A = max(A, 0.0)
        B = max(B, 0.0)
        R = np.clip(R, 0.0, 0.25)
        r = np.clip(r, 0.0, R)
        
        idx = np.arange(N)
        m = idx - HALF
        x = m * DT
        # x represents the center of the bin.
        # A bin is in the outer box if its center is within R.
        # A bin is in the inner box if its center is within r.
        outer = (np.abs(x) <= R).astype(float)
        inner = (np.abs(x) <= r).astype(float)
        
        y = A * outer - B * inner
        y = np.maximum(y, 0.0)
        return y

    def obj(params):
        y = param_to_signal(params)
        return _fft_peak(y)
    
    # Initial guesses
    # 1. Single box (B=0)
    inits = [
        [1.0, 0.25, 0.0, 0.0],
        [1.0, 0.20, 0.5, 0.10],
        [1.0, 0.15, 0.8, 0.05],
        [1.0, 0.25, 1.0, 0.15],
        [1.0, 0.25, 1.0, 0.20],
        [1.0, 0.10, 0.0, 0.0],
        [1.0, 0.05, 0.0, 0.0],
    ]
    
    best_y = None
    best_val = float('inf')
    
    for p0 in inits:
        if time.time() > deadline - 10:
            break
        try:
            res = minimize(obj, p0, method="Nelder-Mead", 
                           options={"maxiter": 200, "xatol": 1e-5, "fatol": 1e-8})
            if res.fun < best_val:
                best_val = res.fun
                best_y = param_to_signal(res.x)
        except Exception:
            pass
            
    if best_y is not None:
        return best_y
    return None

def construct():
    start = time.time()
    deadline = start + 140.0   # safety margin under 150s kill
    best = np.ones(N)
    best_score = _fft_peak(best)

    # 1) Fourier parametric optimization (fast, low-dim)
    try:
        y = _fourier_optimize(deadline)
        if y is not None:
            y = np.maximum(y, 0.0)
            s = _fft_peak(y)
            if s < best_score:
                best, best_score = y, s
    except Exception:
        pass

    # 2) Double-box optimization (structural search for hollow shapes)
    try:
        y = _double_box_optimize(deadline)
        if y is not None:
            y = np.maximum(y, 0.0)
            s = _fft_peak(y)
            if s < best_score:
                best, best_score = y, s
    except Exception:
        pass

    # 3) Direct 600-variable polish to escape parametric constraints
    try:
        best, best_score = _direct_polish(best, deadline)
    except Exception:
        pass

    best = np.maximum(best, 0.0)
    return list(best)

# EVOLVE-BLOCK-END
