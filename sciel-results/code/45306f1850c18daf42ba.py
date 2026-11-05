"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _fft_autoconv(h):
    """Compute full autoconvolution using FFT for speed."""
    n = len(h)
    size = 2 * n
    fh = np.fft.rfft(h, size)
    conv = np.fft.irfft(fh * fh, size)[:n + n - 1]
    return conv

def _compute_value(h):
    s = np.sum(h)
    if s == 0:
        return np.inf
    conv = _fft_autoconv(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _soft_max_obj_and_grad(h, beta=50.0):
    """
    Compute soft-max smoothed objective and its gradient.
    Objective: 2*N*soft_max(conv) / (sum(h))^2
    where soft_max approximates max(conv).
    """
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(h)
    
    conv = _fft_autoconv(h)
    m = np.max(conv)
    # Numerically stable soft max
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -500, 500)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    soft_max = m + np.log(sum_exp) / beta
    weights = exp_shifted / sum_exp
    
    # Gradient of soft_max w.r.t. h
    # d(soft_max)/d(h_i) = sum_j weights_j * d(conv_j)/d(h_i)
    # d(conv_j)/d(h_i) = h_{j-i} + h_{i-j} (with 0 outside [0,N-1])
    
    grad_sm = np.zeros(N)
    for i in range(N):
        # term1: sum over k where 0<=k<N and 0<=i+k<2N-1
        k_min = max(0, 0)
        k_max = min(N-1, 2*N-2-i)
        if k_min <= k_max:
            term1 = np.dot(weights[i+k_min:i+k_max+1], h[k_min:k_max+1])
        else:
            term1 = 0.0
        # term2: sum over k where 0<=k<N and 0<=i-k<2N-1, i.e., i-N+1<=k<=i
        k_min2 = max(0, i-(2*N-2))
        k_max2 = min(N-1, i)
        if k_min2 <= k_max2:
            term2 = np.dot(weights[k_min2:k_max2+1], h[k_min2:k_max2+1])
        else:
            term2 = 0.0
        grad_sm[i] = term1 + term2
    
    # Full objective: f = 2*N*soft_max / s^2
    # df/dh_i = 2*N * (grad_sm_i * s^2 - soft_max * 2*s) / s^4
    #          = 2*N * (grad_sm_i * s - 2*soft_max) / s^3
    obj = 2.0 * N * soft_max / (s * s)
    grad = 2.0 * N * (grad_sm * s - 2.0 * soft_max) / (s * s * s)
    
    return obj, grad

def _objective_wrapper(h, beta):
    obj, _ = _soft_max_obj_and_grad(h, beta)
    return obj

def _grad_wrapper(h, beta):
    _, grad = _soft_max_obj_and_grad(h, beta)
    return grad

def _objective_hard(x):
    """Hard objective for final evaluation."""
    h = x
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10
    conv = _fft_autoconv(h)
    return abs(2.0 * N * np.max(conv) / (s * s))

def _chirp_value(params):
    """
    Objective for chirp parameters (theta, c).
    h_i = cos(theta * i) + c
    """
    theta, c = params
    i = np.arange(N)
    h = np.cos(theta * i) + c
    return _objective_hard(h)

def _chirp_h(params):
    theta, c = params
    i = np.arange(N)
    return np.cos(theta * i) + c

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s limit
    
    best_h = np.ones(N)
    best_val = _compute_value(best_h)
    
    rng = np.random.default_rng(42)
    
    # 1. Optimize Chirp Parameters
    # h_i = cos(theta * i) + c
    # We expect theta to be smallish to keep S large, but non-zero to reduce peak.
    # c should be positive to ensure S != 0 and help with the sum.
    
    # Initial guesses for theta, c
    # theta in [0, pi], c in [0, 2]
    inits = [
        (0.1, 0.5),
        (0.2, 0.5),
        (0.5, 0.5),
        (1.0, 0.5),
        (0.05, 1.0),
        (0.1, 1.0),
    ]
    
    for init in inits:
        if time.time() > deadline - 60.0:
            break
        try:
            res = minimize(_chirp_value, init, method='Nelder-Mead', options={'maxiter': 500})
            h_chirp = _chirp_h(res.x)
            val_chirp = _objective_hard(h_chirp)
            if val_chirp < best_val:
                best_val = val_chirp
                best_h = h_chirp.copy()
        except Exception:
            pass

    # 2. Refine with L-BFGS-B on the full vector
    # Use the best chirp result as starting point
    h_curr = best_h.copy()
    
    # Also try a few other diverse starts if time permits
    other_starts = []
    # Comb
    h_comb = np.array([1.0 if (i % 2 == 0) else -0.5 for i in range(N)])
    other_starts.append(h_comb)
    # Random
    h_rand = rng.standard_normal(N)
    if abs(np.sum(h_rand)) < 1e-6: h_rand[0] += 1
    other_starts.append(h_rand)
    
    all_starts = [h_curr] + other_starts
    
    for h0 in all_starts:
        if time.time() > deadline - 20.0:
            break
        
        # Ensure sum is not zero
        s0 = np.sum(h0)
        if abs(s0) < 1e-10:
            h0 = h0 + 1.0
            
        h_curr = h0.copy()
        
        # Run L-BFGS-B
        try:
            res = minimize(_objective_wrapper, h_curr, 
                          method='L-BFGS-B',
                          jac=lambda x: _grad_wrapper(x, 50.0),
                          args=(50.0,),
                          options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8, 'maxfun': 1000})
            h_new = res.x
        except Exception:
            h_new = h_curr
            
        val_new = _objective_hard(h_new)
        if val_new < best_val:
            best_val = val_new
            best_h = h_new.copy()
            
        # Second pass with higher beta if improved
        if val_new < best_val + 1e-6 and time.time() < deadline - 10.0:
            try:
                res2 = minimize(_objective_wrapper, h_new, 
                              method='L-BFGS-B',
                              jac=lambda x: _grad_wrapper(x, 100.0),
                              args=(100.0,),
                              options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 500})
                h_new2 = res2.x
                val_new2 = _objective_hard(h_new2)
                if val_new2 < best_val:
                    best_val = val_new2
                    best_h = h_new2.copy()
            except Exception:
                pass

    # Safety: ensure valid output
    s_final = np.sum(best_h)
    if abs(s_final) < 1e-10:
        return np.ones(N)
    
    return best_h
# EVOLVE-BLOCK-END
