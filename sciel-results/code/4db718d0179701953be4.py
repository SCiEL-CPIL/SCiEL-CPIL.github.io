"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_FFT_SIZE = 1
while _FFT_SIZE < 2 * N - 1:
    _FFT_SIZE *= 2

def _autoconv_fft(h):
    """Compute linear autoconvolution using FFT."""
    H = np.fft.rfft(h, _FFT_SIZE)
    ac = np.fft.irfft(H * H, _FFT_SIZE)
    return ac[:2*N-1]

def _autoconv_grad(h):
    """
    Gradient of autoconvolution w.r.t. h.
    d/dh_i [h*h]_k = sum_j (delta_{k,i+j} + delta_{k,j+i}) h_j
    This is equivalent to: grad = 2 * correlate(h, h) but for linear conv:
    For linear autoconv ac[k] = sum_j h[j]*h[k-j] (with h out of range = 0),
    d ac[k]/d h[i] = sum_j (h[j] * (delta_{k,i+j} + delta_{k,j+i}))
                    = h[k-i] + h[i-k] (when in range)
    So grad[i] for ac[k] is h[k-i] + h[i-k] if indices valid.
    More simply: the Jacobian of ac w.r.t. h has entries J[k,i] = h[k-i] + h[i-k] (if in [0,N-1])
    For soft-max, we need d(softmax)/d h = sum_k w_k * J[k,i] where w_k = softmax weights.
    """
    # We'll compute this on-the-fly in the objective gradient
    pass

def _score(h):
    """Compute the objective value: max|autoconv| / (sum h)^2."""
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    return np.max(np.abs(ac)) / (s * s)

def _soft_max_abs(ac, beta=100.0):
    """
    Smooth approximation of max(|ac|) via log-sum-exp.
    Returns (value, weights) where weights are softmax weights over |ac|.
    """
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    if m < 1e-15:
        n_ac = len(ac)
        return 0.0, np.ones(n_ac) / n_ac
    scaled = beta * (abs_ac - m)
    scaled = np.clip(scaled, -100, 100)
    exp_scaled = np.exp(scaled)
    sum_exp = np.sum(exp_scaled)
    if sum_exp < 1e-300:
        n_ac = len(ac)
        return 0.0, np.ones(n_ac) / n_ac
    weights = exp_scaled / sum_exp
    log_sum = np.log(sum_exp)
    value = m + log_sum / beta
    return value, weights

def _objective_smooth(h, beta=50.0):
    """
    Smooth objective for optimization.
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10
    ac = _autoconv_fft(h)
    sm, _ = _soft_max_abs(ac, beta)
    return sm / (s * s)

def _objective_smooth_grad(h, beta=50.0):
    """
    Smooth objective and its gradient for L-BFGS-B.
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10, np.zeros_like(h)
    
    ac = _autoconv_fft(h)
    sm, weights = _soft_max_abs(ac, beta)
    
    # d(sm)/d(h[i]) = sum_k weights[k] * d(ac[k])/d(h[i])
    # d(ac[k])/d(h[i]) = h[k-i] + h[i-k] (if indices in range)
    # This is the (circular) correlation, but for linear conv we need to be careful.
    # For linear autoconvolution ac[k] = sum_j h[j]*h[k-j] for 0<=k<=2N-2
    # d(ac[k])/d(h[i]) = h[k-i] (if 0<=k-i<N) + h[i-k] (if 0<=i-k<N)
    
    # Compute gradient efficiently:
    # grad[i] = sum_k weights[k] * (h[k-i] + h[i-k])
    # Let w = weights (length 2N-1)
    # grad[i] = sum_k w[k]*h[k-i] + sum_k w[k]*h[i-k]
    # First term: for fixed i, k-i ranges from 0 to N-1, so k from i to i+N-1
    #   = sum_{j=0}^{N-1} w[i+j] * h[j]
    # Second term: for fixed i, i-k ranges from 0 to N-1, so k from i-N+1 to i
    #   = sum_{j=0}^{N-1} w[i-j] * h[j]
    # Note w[k] = 0 for k < 0 or k >= 2N-1
    
    w = weights  # length 2N-1
    # Pad w to handle boundary conditions
    # First term: conv of w with h, but only linear part
    # grad1[i] = sum_{j=0}^{N-1} w[i+j] * h[j]
    # This is like (w * h) but with w indexed from i
    # Use FFT for efficiency
    
    # Actually, let's compute it directly since N=400 is small
    # grad[i] = sum_k w[k] * (h[k-i] if 0<=k-i<N else 0 + h[i-k] if 0<=i-k<N else 0)
    
    grad = np.zeros(N)
    for i in range(N):
        # First part: h[k-i] where 0<=k-i<N => i<=k<=i+N-1
        k_start = max(0, i)
        k_end = min(2*N-2, i+N-1)
        if k_start <= k_end:
            j_vals = np.arange(k_start, k_end+1) - i  # k-i values
            grad[i] += np.dot(w[k_start:k_end+1], h[j_vals])
        
        # Second part: h[i-k] where 0<=i-k<N => i-N+1<=k<=i
        k_start2 = max(0, i-N+1)
        k_end2 = min(2*N-2, i)
        if k_start2 <= k_end2:
            j_vals2 = i - np.arange(k_start2, k_end2+1)  # i-k values
            grad[i] += np.dot(w[k_start2:k_end2+1], h[j_vals2])
    
    dsm_dh = grad
    
    # d(sm/(s^2))/dh = (dsm_dh * s^2 - sm * 2*s * 1) / s^4
    #                 = dsm_dh / s^2 - 2*sm/(s^3)
    obj_val = sm / (s * s)
    grad_obj = dsm_dh / (s * s) - 2.0 * sm / (s * s * s)
    
    return obj_val, grad_obj

def _gen_chirp(n):
    """Generate a chirp sequence."""
    i = np.arange(n)
    phase = np.pi * i**2 / n
    return np.cos(phase)

def _gen_sinc(n):
    """Generate a windowed sinc sequence."""
    i = np.arange(n)
    center = n / 2
    x = (i - center)
    sigma = n / 4
    return np.exp(-0.5 * (x / sigma) ** 2)

def _local_search_fast(h, max_passes=30, deadline=None):
    """
    Local search with incremental autoconvolution updates.
    Flipping sign at index i: new_h = h with h[i] -> -h[i]
    new_ac = old_ac - 2*h[i]*(h convolved with delta_i)
    The delta_i contribution to ac is: ac_delta[k] = h[k-i] + h[i-k] (if in range)
    So new_ac[k] = old_ac[k] - 2*h[i]*(h[k-i] + h[i-k])
    """
    h = h.copy()
    ac = _autoconv_fft(h)
    best_val = np.max(np.abs(ac)) / (np.sum(h) * np.sum(h))
    s = np.sum(h)
    
    for pass_num in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        
        # Try flipping signs
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            
            old_hi = h[i]
            new_hi = -old_hi
            
            # Update autoconvolution
            # delta_ac[k] = -2 * old_hi * (h[k-i] + h[i-k])
            # Compute this incrementally
            delta_ac = np.zeros(2*N-1)
            # h[k-i] part: k from i to i+N-1
            k_start = max(0, i)
            k_end = min(2*N-2, i+N-1)
            if k_start <= k_end:
                j_vals = np.arange(k_start, k_end+1) - i
                delta_ac[k_start:k_end+1] -= 2.0 * old_hi * h[j_vals]
            
            # h[i-k] part: k from max(0,i-N+1) to min(2N-2,i)
            k_start2 = max(0, i-N+1)
            k_end2 = min(2*N-2, i)
            if k_start2 <= k_end2:
                j_vals2 = i - np.arange(k_start2, k_end2+1)
                delta_ac[k_start2:k_end2+1] -= 2.0 * old_hi * h[j_vals2]
            
            new_ac = ac + delta_ac
            new_s = s - 2.0 * old_hi
            if abs(new_s) < 1e-9:
                continue
            
            new_val = np.max(np.abs(new_ac)) / (new_s * new_s)
            
            if new_val < best_val:
                # Accept
                h[i] = new_hi
                ac = new_ac
                s = new_s
                best_val = new_val
                improved = True
            # else: keep old h[i], ac unchanged
        
        if not improved:
            break
    
    return h, best_val

def construct():
    t_start = time.time()
    deadline = t_start + 130.0  # Safety margin
    
    best_score = np.inf
    best_h = np.ones(N)
    
    base_score = _score(best_h)
    best_score = base_score
    best_h = best_h.copy()

    inits = []
    
    # 1. Constant
    inits.append(np.ones(N))
    
    # 2. Gaussian
    x = np.arange(N)
    mu = N/2
    sigma = N/4
    inits.append(np.exp(-0.5 * ((x - mu) / sigma) ** 2))
    
    # 3. Chirp
    inits.append(_gen_chirp(N))
    
    # 4. Sinc (windowed)
    inits.append(_gen_sinc(N))
    
    # 5. Random smoothed
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(N)
    kernel = np.ones(10)/10
    smooth = np.convolve(noise, kernel, mode='same')
    inits.append(smooth)
    
    # 6. Alternating signs with Gaussian envelope
    x = np.arange(N)
    mu = N/2
    sigma = N/3
    gauss_env = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    alt_signs = np.where(np.arange(N) % 2 == 0, 1.0, -1.0)
    inits.append(gauss_env * alt_signs)
    
    # 7. Barker-like sequence
    barker_like = np.ones(N)
    for i in range(1, N):
        barker_like[i] = 1.0 if ((i * 7 + 3) % 11) < 5 else -1.0
    inits.append(barker_like)
    
    # 8. Cosine-modulated chirp
    x = np.arange(N)
    chirp_cos = np.cos(np.pi * x**2 / N) * np.cos(2 * np.pi * x / N)
    inits.append(chirp_cos)
    
    # 9. Another random init
    rng2 = np.random.default_rng(42)
    inits.append(rng2.standard_normal(N))
    
    for h_init in inits:
        if time.time() > deadline - 15:
            break
        
        s0 = np.sum(h_init)
        if abs(s0) < 1e-6:
            continue
        
        h_norm = h_init / s0
        
        try:
            for beta in [30, 60, 120]:
                if time.time() > deadline - 8:
                    break
                
                def obj(x):
                    return _objective_smooth(x, beta=beta)
                
                def obj_grad(x):
                    val, grad = _objective_smooth_grad(x, beta=beta)
                    return val, grad
                
                res = minimize(obj_grad, h_norm, method='L-BFGS-B', 
                               options={'maxiter': 200, 'disp': False, 'gtol': 1e-7, 'ftol': 1e-10})
                h_opt = res.x
                
                if abs(np.sum(h_opt)) < 1e-6:
                    continue
                    
                val = _score(h_opt)
                if val < best_score:
                    best_score = val
                    best_h = h_opt.copy()
        except Exception:
            pass

    # Final local search with incremental updates
    if time.time() < deadline - 3:
        h_refined, val_refined = _local_search_fast(best_h, max_passes=25, deadline=deadline-1)
        if val_refined < best_score:
            best_score = val_refined
            best_h = h_refined.copy()

    # Final check
    if best_score > base_score:
        best_score = base_score
        best_h = np.ones(N)

    return best_h

# EVOLVE-BLOCK-END
