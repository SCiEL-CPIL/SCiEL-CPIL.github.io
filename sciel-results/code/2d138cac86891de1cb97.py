"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

def _autoconv_max_abs(x):
    """Max absolute value of the discrete autoconvolution of x."""
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)
    return float(np.max(np.abs(conv[:2 * N - 1])))

def _integral(x):
    """Integral of the equal-width step function = sum of the heights."""
    return float(np.sum(x))

def _metric(x):
    """max |autoconv| / (integral)^2.  Lower is better."""
    s = _integral(x)
    if abs(s) < 1e-12:
        return float('inf')
    return _autoconv_max_abs(x) / (s * s)

def _soft_max_abs(conv, beta=100.0):
    """Smooth approximation of max |conv| using log-sum-exp."""
    abs_conv = np.abs(conv)
    max_val = np.max(abs_conv)
    if max_val == 0:
        return 0.0, np.zeros_like(conv)
    
    # Numerical stability: shift by max
    shifted = abs_conv - max_val
    exp_shifted = np.exp(beta * shifted)
    sum_exp = np.sum(exp_shifted)
    
    # Value: max + (1/beta) * log(sum(exp(beta * (x - max))))
    val = max_val + np.log(sum_exp) / beta
    
    # Gradient with respect to abs_conv
    # d/dx [ max + 1/beta * log(sum(exp(beta*(x-max))) ) ]
    # = 1/beta * (1/sum_exp) * sum( exp(beta*(x_i-max)) * beta * sign(x_i) )
    # = sum( exp(beta*(x_i-max)) * sign(x_i) ) / sum_exp
    
    grad_abs = (exp_shifted * np.sign(conv)) / sum_exp
    
    # Gradient with respect to conv (chain rule for abs)
    # d/dconv |conv| = sign(conv)
    # But we already used sign(conv) in grad_abs? 
    # Let's re-derive carefully.
    # f(conv) = max_val + (1/beta) log( sum exp(beta (|conv_i| - max_val)) )
    # Let u_i = |conv_i|. f = max + 1/beta log(sum exp(beta(u_i - max)))
    # df/du_i = exp(beta(u_i - max)) / sum(exp(beta(u_j - max)))
    # du_i/dconv_i = sign(conv_i)
    # df/dconv_i = df/du_i * sign(conv_i)
    
    # My previous grad_abs calculation:
    # exp_shifted = exp(beta(u_i - max))
    # sum_exp = sum(exp_shifted)
    # grad_abs = (exp_shifted * sign(conv)) / sum_exp
    # This is exactly df/dconv_i.
    
    return val, grad_abs

def _smooth_metric_and_grad(x, beta=100.0):
    """
    Returns the smooth surrogate value and its gradient.
    Objective: soft_max |autoconv| / (integral)^2
    """
    fx = np.fft.rfft(x, _FFT_SIZE)
    conv = np.fft.irfft(fx * fx, _FFT_SIZE)[:2 * N - 1]
    s = _integral(x)
    
    if abs(s) < 1e-12:
        return 1e10, np.zeros(N)
    
    val, grad_conv = _soft_max_abs(conv, beta)
    
    # Objective J = val / s^2
    # dJ/dx = (dval/dx * s^2 - val * d(s^2)/dx) / s^4
    # d(s^2)/dx = 2s * (ds/dx) = 2s * 1 = 2s
    # dJ/dx = (dval/dx * s^2 - val * 2s) / s^4
    #        = dval/dx / s^2 - 2*val / s^3
    #        = (dval/dx * s - 2*val) / s^3
    
    # Now we need dval/dx.
    # val depends on conv. conv is the linear operator T(x) = x * x (autoconv).
    # We need the adjoint of T.
    # Let c = T(x). 
    # We have grad_c = dval/dc.
    # We want grad_x = T^* (grad_c).
    
    # T(x) = ifft( fft(x) * fft(x) ) truncated.
    # Let F(x) = fft(x, _FFT_SIZE).
    # C = F(x) * F(x).
    # c = ifft(C) (first 2N-1).
    
    # dval/dx = sum_k (grad_c[k] * dc[k]/dx)
    # dc[k]/dx is the adjoint of the autoconvolution operator.
    
    # For linear conv y = x * h, adjoint is x * h_flipped.
    # For autoconv y = x * x, it's nonlinear.
    # y_l = sum_k x_k x_{l-k}
    # dy_l / dx_m = x_{l-m} + x_{m-l} (with appropriate bounds/zero padding)
    # In FFT domain:
    # Y = X^2
    # dY / dX = 2X
    # But we are in time domain for the gradient accumulation.
    
    # Let's use the FFT adjoint trick.
    # Let v be the vector in the frequency domain corresponding to the gradient wrt C.
    # v = fft(grad_conv_padded, _FFT_SIZE)
    # Then dval/dX = 2 * X * v (element-wise)
    # Then dval/dx = ifft(dval/dX)
    
    # Pad grad_conv to _FFT_SIZE
    grad_conv_full = np.zeros(_FFT_SIZE)
    # Note: autoconv is symmetric/real. The FFT of a real sequence is conjugate symmetric.
    # We need to be careful with the padding.
    # conv is length 2N-1.
    # grad_conv is length 2N-1.
    # We need to map this back to the frequency domain gradient.
    
    # The operation was:
    # 1. x -> fft -> X
    # 2. X -> X*X -> C
    # 3. C -> ifft -> conv_full (length _FFT_SIZE)
    # 4. conv_full[:2N-1] -> conv
    # 5. conv -> val
    
    # So grad_conv_full is the gradient wrt the full ifft output.
    # We set grad_conv_full[:2N-1] = grad_conv.
    # The rest are 0 because we didn't use them in val.
    
    # Now, dval/dC = fft(grad_conv_full, _FFT_SIZE)
    # dval/dX = 2 * X * dval/dC
    # dval/dx = ifft(dval/dX, _FFT_SIZE)[:N]
    
    dval_dC = np.fft.rfft(grad_conv_full, _FFT_SIZE)
    dval_dX = 2 * fx * dval_dC
    dval_dx = np.fft.irfft(dval_dX, _FFT_SIZE)[:N]
    
    # Combine with the integral part
    grad_J = (dval_dx * s - 2 * val) / (s * s * s)
    
    return val / (s * s), grad_J

def _try_gradient_descent(x_init, deadline, rng):
    """
    Use L-BFGS-B to minimize the smooth surrogate.
    """
    x = x_init.copy()
    best_x = x.copy()
    best_val = _metric(x)
    
    # We minimize the smooth metric.
    # L-BFGS-B requires bounds. The problem doesn't explicitly bound heights, 
    # but large heights increase autoconv. 
    # Let's try unbounded first, or large bounds.
    
    def objective_and_grad(x_flat):
        x = x_flat.reshape(N)
        val, grad = _smooth_metric_and_grad(x, beta=100.0)
        return val, grad.reshape(-1)
    
    # Initial gradient check? Maybe not needed for speed, but good for debugging.
    # We will run minimize.
    
    start_time = time.time()
    try:
        # Use L-BFGS-B
        # maxiter is a hard cap, but we check time in callback
        stop_flag = [False]
        
        def callback(xk):
            if time.time() > deadline - 5:
                stop_flag[0] = True
                return True
            return False
        
        res = minimize(
            objective_and_grad, 
            x, 
            jac=True, 
            method='L-BFGS-B', 
            options={'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-6, 'maxls': 50},
            callback=callback
        )
        
        if res.x is not None:
            x_new = res.x.reshape(N)
            val_new = _metric(x_new)
            if val_new < best_val:
                best_val = val_new
                best_x = x_new.copy()
                
    except Exception as e:
        pass
        
    return best_x, best_val

def _try_local_search_fast(x, deadline, rng, max_rounds=50000):
    """Fast local search with in-place modifications."""
    best_x = x.copy()
    best_val = _metric(x)
    no_improve = 0
    step_scale = 0.1
    
    for round_idx in range(max_rounds):
        if time.time() > deadline:
            break
        
        if no_improve > 50:
            step_scale *= 0.9
            no_improve = 0
            if step_scale < 0.001:
                break
        
        # Modify a random subset
        k = int(rng.integers(1, 5))
        indices = rng.choice(N, size=k, replace=False)
        
        best_delta = 0
        best_idx = None
        
        # Try a few perturbations
        for trial in range(5):
            x_test = best_x.copy()
            for idx in indices:
                x_test[idx] += rng.normal(0, step_scale)
            v = _metric(x_test)
            if v < best_val + best_delta:
                best_delta = v - best_val
                best_idx = x_test
        
        if best_idx is not None and best_delta < 0:
            best_val += best_delta
            best_x = best_idx
            no_improve = 0
        else:
            no_improve += 1
            
    return best_x, best_val

def construct():
    deadline = time.time() + 140.0 # Safety margin
    
    # Start with a good initial guess.
    # Random initialization often works well for GD, but let's try a few starts.
    rng = np.random.default_rng(42)
    
    best_x = None
    best_val = float('inf')
    
    # Start 1: Random
    x_init = rng.normal(0, 1, N)
    x_gd, val_gd = _try_gradient_descent(x_init, deadline - 20, rng)
    if val_gd < best_val:
        best_val = val_gd
        best_x = x_gd
        
    # Start 2: Sine wave
    t = np.linspace(0, 4*np.pi, N)
    x_init2 = np.sin(t)
    x_gd2, val_gd2 = _try_gradient_descent(x_init2, deadline - 20, rng)
    if val_gd2 < best_val:
        best_val = val_gd2
        best_x = x_gd2
        
    # Start 3: Random sparse
    x_init3 = np.zeros(N)
    mask = rng.random(N) < 0.1
    x_init3[mask] = rng.normal(0, 1, mask.sum())
    x_gd3, val_gd3 = _try_gradient_descent(x_init3, deadline - 20, rng)
    if val_gd3 < best_val:
        best_val = val_gd3
        best_x = x_gd3

    # Final Polish: Discrete local search on the best GD result
    if best_x is not None and time.time() < deadline - 5:
        best_x, best_val = _try_local_search_fast(best_x, deadline, rng, max_rounds=10000)
        
    if best_x is None:
        return np.ones(N)
        
    return best_x
# EVOLVE-BLOCK-END
