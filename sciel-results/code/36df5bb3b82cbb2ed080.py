"""
Structural change: Direct L-BFGS-B optimization of the time-domain signal x,
using a smooth log-sum-exp surrogate for the max absolute autoconvolution,
with analytic gradients computed via FFT.

This is materially different from the current program's approach of:
1. Optimizing phases in frequency domain with fixed magnitudes (Nelder-Mead)
2. Then doing gradient descent on a smooth surrogate
3. Then local search
4. Then coordinate descent polish

The new approach directly optimizes x in time domain using L-BFGS-B, which
is a quasi-Newton method that can converge much faster than Nelder-Mead for
this smooth problem. The gradient is computed analytically via FFT-based
autoconvolution.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 400

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2


def _autoconv_values(x):
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return conv[:2 * N - 1]


def _integral(x):
    return float(np.sum(x))


def _metric(x):
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    conv = _autoconv_values(x)
    return float(np.max(np.abs(conv))) / (s * s)


def _soft_max_abs(conv, beta=500.0):
    abs_conv = np.abs(conv)
    max_val = np.max(abs_conv)
    shifted = abs_conv - max_val
    return max_val + np.log(np.sum(np.exp(beta * shifted))) / beta


def _smooth_objective_and_grad(x, beta=300.0):
    """
    Compute the smooth surrogate objective and its analytic gradient.
    
    Objective: soft_max_abs(conv) / (sum(x))^2
    where conv is the linear autoconvolution of x.
    
    Gradient derivation:
    Let f = soft_max_abs(conv), s = sum(x).
    Objective = f / s^2
    
    df/dx_j = sum_k w_k * d|conv_k|/dx_j
    where w_k = exp(beta * (|conv_k| - max|conv|)) / sum_m exp(beta * (|conv_m| - max|conv|))
    
    d|conv_k|/dx_j = sign(conv_k) * (x_{j+k} + x_{j-k}) for valid indices
    
    d(f/s^2)/dx_j = (df/dx_j * s^2 - f * 2s * 1) / s^4
                  = (df/dx_j * s - 2f) / s^3
    """
    conv = _autoconv_values(x)
    s = _integral(x)
    if abs(s) < 1e-12:
        return 1e10, np.zeros(N)
    
    abs_conv = np.abs(conv)
    max_val = np.max(abs_conv)
    shifted = abs_conv - max_val
    weights = np.exp(beta * shifted)
    weights_sum = np.sum(weights)
    if weights_sum < 1e-30:
        return 1e10, np.zeros(N)
    weights /= weights_sum
    
    f = max_val + np.log(weights_sum) / beta
    
    # Compute df/dx
    df_dx = np.zeros(N)
    for k in range(2 * N - 1):
        if weights[k] < 1e-12:
            continue
        sign_k = np.sign(conv[k]) if conv[k] != 0 else 1.0
        for j in range(N):
            contrib = 0.0
            if j + k < N:
                contrib += x[j + k]
            if j - k >= 0:
                contrib += x[j - k]
            df_dx[j] += 2.0 * sign_k * weights[k] * contrib
    
    grad = (df_dx * s - 2.0 * f) / (s ** 3)
    
    return f / (s * s), grad


def _lbfgs_optimize(x_init, deadline, rng, max_iters=5000):
    """
    Direct L-BFGS-B optimization of x in time domain.
    """
    best_x = x_init.copy()
    best_val = _metric(best_x)
    current_x = x_init.copy()
    
    def objective_and_grad(x):
        val, grad = _smooth_objective_and_grad(x, beta=300.0)
        return val, grad
    
    # Run L-BFGS-B
    start_time = time.time()
    try:
        result = minimize(
            objective_and_grad,
            current_x,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': max_iters, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 10000}
        )
        
        if result.fun < best_val:
            val_final = _metric(result.x)
            if val_final < best_val:
                best_val = val_final
                best_x = result.x.copy()
                
    except Exception:
        pass
    
    return best_x, best_val


def _multi_start_lbfgs(x_init, deadline, rng, num_starts=8):
    """
    Multiple random starts for L-BFGS-B to avoid local minima.
    """
    best_x = x_init.copy()
    best_val = _metric(best_x)
    
    time_per_start = (deadline - time.time()) / num_starts
    
    for start_idx in range(num_starts):
        if time.time() > deadline - 5:
            break
        
        if start_idx == 0:
            x_start = x_init.copy()
        else:
            # Perturb the initial solution
            noise_scale = 0.5 * np.std(x_init)
            x_start = x_init + rng.normal(0, noise_scale, size=N)
        
        x_opt, val_opt = _lbfgs_optimize(x_start, deadline - 5, rng, max_iters=2000)
        
        if val_opt < best_val:
            best_val = val_opt
            best_x = x_opt.copy()
    
    return best_x, best_val


def _final_polish(x, deadline, rng):
    """High-precision final polish with coordinate descent."""
    best_x = x.copy()
    best_val = _metric(best_x)
    
    for sweep in range(20):
        if time.time() > deadline:
            break
        
        improved = False
        for idx in range(N):
            if time.time() > deadline:
                break
            
            lo, hi = -0.02, 0.02
            best_delta = 0.0
            best_v = best_val
            
            for _ in range(8):
                if time.time() > deadline:
                    break
                m1 = lo + (hi - lo) / 3
                m2 = hi - (hi - lo) / 3
                
                x1 = best_x.copy()
                x1[idx] += m1
                v1 = _metric(x1)
                
                x2 = best_x.copy()
                x2[idx] += m2
                v2 = _metric(x2)
                
                if v1 < v2:
                    hi = m2
                else:
                    lo = m1
                
                if v1 < best_v:
                    best_v = v1
                    best_delta = m1
                if v2 < best_v:
                    best_v = v2
                    best_delta = m2
            
            if best_delta != 0.0 and best_v < best_val:
                best_x[idx] += best_delta
                best_val = best_v
                improved = True
        
        if not improved:
            break
    
    return best_x, best_val


def construct():
    deadline = time.time() + 145.0
    
    # Start with ones
    best_x = np.ones(N)
    best_val = _metric(best_x)
    
    rng = np.random.default_rng(2024)
    
    # Phase 1: Multi-start L-BFGS-B optimization
    if time.time() < deadline - 30:
        x_opt, val_opt = _multi_start_lbfgs(best_x, deadline - 30, rng, num_starts=6)
        if val_opt < best_val:
            best_val = val_opt
            best_x = x_opt
    
    # Phase 2: Final polish
    if time.time() < deadline - 2:
        best_x, best_val = _final_polish(best_x, deadline - 2, rng)
    
    return best_x
