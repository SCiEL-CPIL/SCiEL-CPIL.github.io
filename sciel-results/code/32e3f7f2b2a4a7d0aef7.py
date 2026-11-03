"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 400

# EVOLVE-BLOCK-START

def _get_fft_size(n):
    """Get the next power of 2 size for FFT of linear convolution of length n."""
    size = 2 * n - 1
    fsize = 1
    while fsize < size:
        fsize *= 2
    return fsize

def _autoconv_fft(h):
    """Compute autoconvolution using FFT."""
    n = len(h)
    fsize = _get_fft_size(n)
    H = np.fft.fft(h, fsize)
    ac = np.fft.ifft(H * H, fsize)
    return ac[:2*n-1]

def _score(h):
    """
    Compute the raw value: max |autoconv| / (integral)^2.
    """
    integral = np.sum(h)
    if abs(integral) < 1e-12:
        return 1e10
    ac = _autoconv_fft(h)
    max_abs_ac = np.max(np.abs(ac))
    return max_abs_ac / (integral * integral)

def _soft_max_abs(ac, temp):
    """
    Smooth surrogate for max(|ac|) using log-sum-exp.
    """
    z = np.abs(ac)
    z_max = np.max(z)
    if z_max == 0:
        return 0.0
    exp_term = np.exp((z - z_max) / temp)
    log_sum = np.log(np.sum(exp_term))
    return z_max + temp * log_sum

def _objective_smooth(x, temp):
    """
    Objective function for the optimizer.
    x is the full vector of length N.
    """
    ac = _autoconv_fft(x)
    soft_max_val = _soft_max_abs(ac, temp)
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return 1e10
    return soft_max_val / (integral * integral)

def _gradient_smooth(x, temp):
    """
    Analytical gradient of the soft-max surrogate objective.
    f(x) = S(x) / I^2, where S(x) = soft_max(|ac|), I = sum(x).
    dS/dx_j = sum_k p_k * d|ac_k|/dx_j
    p_k = exp((z_k - z_max)/temp) / sum(exp((z - z_max)/temp))
    d|ac_k|/dx_j = sign(ac_k) * (h_{k-j} + h_{j-k}) for valid k, j (0-indexed in full conv)
    Actually, ac_k = sum_i h_i h_{k-i}.
    d ac_k / d x_j = h_{k-j} + h_{j-k} (if indices valid, else one term or zero)
    Since h is real, ac is symmetric.
    Let's compute dS/dx via matrix multiplication or direct loop?
    N=400, so 400x400 matrix is fine.
    But we can be smarter.
    
    Let P = vector of p_k (length 2N-1).
    Let S = soft_max_val.
    Gradient of S w.r.t x:
    g_S[j] = sum_k p_k * sign(ac_k) * (h_{k-j} + h_{j-k})
    Note: ac_k = sum_i h_i h_{k-i}.
    If we define h as index 0..N-1.
    ac_k is defined for k=0..2N-2.
    Term h_{k-j} exists if 0 <= k-j < N.
    Term h_{j-k} exists if 0 <= j-k < N.
    
    We can compute this efficiently.
    Let H_full be zero-padded h to length 2N-1? No.
    Let's just use the fact that the gradient of sum_k w_k ac_k is w * h + reverse(w) * h (convolved?).
    Actually, let u_k = p_k * sign(ac_k).
    Then dS/dx_j = sum_k u_k (h_{k-j} + h_{j-k}).
    Let v_k = u_k.
    Then sum_k v_k h_{k-j} is (v * h)_j? No, it's cross-correlation.
    Let's define correlation c[j] = sum_k v_k h_{k-j}.
    This is equivalent to np.correlate(v, h, mode='valid')? No.
    v has length 2N-1. h has length N.
    c[j] = sum_{k} v_k h_{k-j}.
    If we pad h to length 2N-1 with zeros, let H_pad.
    Then c[j] = sum_k v_k H_pad[k-j].
    This is convolution of v and reverse(H_pad)?
    Let's just compute it directly for N=400. 400*400 = 160k ops. Very fast.
    """
    ac = _autoconv_fft(x)
    z = np.abs(ac)
    z_max = np.max(z)
    if z_max == 0:
        return np.zeros_like(x)
    
    exp_term = np.exp((z - z_max) / temp)
    sum_exp = np.sum(exp_term)
    p = exp_term / sum_exp  # length 2N-1
    
    signs = np.sign(ac)
    u = p * signs  # length 2N-1
    
    # g_S[j] = sum_k u_k (x_{k-j} + x_{j-k})
    # Term 1: sum_k u_k x_{k-j}
    # Term 2: sum_k u_k x_{j-k}
    
    g1 = np.zeros(N)
    g2 = np.zeros(N)
    
    # Efficient computation using convolution/correlation
    # g1[j] = sum_k u_k x_{k-j}
    # Let U be u padded to length 2N-1? No, u is already length 2N-1.
    # This is cross-correlation of u and x.
    # np.correlate(u, x, mode='valid')?
    # u length 2N-1, x length N.
    # 'valid' mode returns length (2N-1) - N + 1 = N.
    # c[k] = sum_i u[k+i] x[i].
    # We want sum_k u_k x_{k-j}. Let i = k-j => k = i+j.
    # sum_i u_{i+j} x_i.
    # This is exactly np.correlate(u, x, 'valid')[j]?
    # Let's check: np.correlate(a, b, 'valid') computes sum a[k+i] b[i].
    # So g1 = np.correlate(u, x, mode='valid')
    
    # g2[j] = sum_k u_k x_{j-k}
    # Let i = j-k => k = j-i.
    # sum_i u_{j-i} x_i.
    # This is convolution?
    # np.convolve(u, x, 'full')?
    # Let's use np.correlate(u, x, 'full') and extract?
    # Or note that g2[j] = g1_rev?
    # u is not symmetric.
    # Let's compute g2 as np.correlate(u, x, 'full')?
    # Actually, g2[j] = sum_i x_i u_{j-i}.
    # This is (x * u)[j] (convolution).
    # But u has length 2N-1.
    # Let's just do the direct loop if it's simpler, or use FFT.
    # Since we already use FFT, let's do it in frequency domain?
    # No, direct is fine for N=400.
    
    # Let's use numpy's correlate for g1.
    # u length 2N-1, x length N.
    # 'valid' output length N.
    g1 = np.correlate(u, x, mode='valid')
    
    # For g2:
    # g2[j] = sum_i x_i u_{j-i}.
    # This is the convolution of x and u.
    # We can compute full convolution and take the middle part?
    # Conv length: N + (2N-1) - 1 = 3N-2.
    # We need indices 0..N-1?
    # Let's check j=0: sum_i x_i u_{-i}. u_{-i} is 0 for i>0 (since u starts at 0).
    # So g2[0] = x_0 u_0.
    # Convolution c[m] = sum_i x_i u_{m-i}.
    # So g2[j] = c[j].
    # We need c[0..N-1].
    
    # We can compute c = np.convolve(x, u, mode='full')
    # But u is length 2N-1.
    # Let's just do it.
    c = np.convolve(x, u, mode='full')
    g2 = c[:N]
    
    g_S = g1 + g2
    
    # Now total gradient of f = S / I^2
    # df/dx_j = (I^2 * g_S[j] - S * 2 * I) / I^4
    #          = g_S[j] / I^2 - 2 * S / I^3
    
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return np.zeros_like(x)
        
    S = z_max + temp * np.log(sum_exp)
    
    grad = g_S / (integral * integral) - 2 * S / (integral * integral * integral)
    
    return grad

def _lbfgs_optimize(h0, temp, deadline):
    """Run L-BFGS-B optimization with given temperature."""
    def obj(x):
        return _objective_smooth(x, temp)
    
    def grad(x):
        return _gradient_smooth(x, temp)
    
    res = minimize(
        obj,
        h0,
        jac=grad,
        method='L-BFGS-B',
        bounds=[(-10, 10)] * N,
        options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    return res.x

def _single_flip_local_search(h, deadline, max_iters=1000):
    """
    Local search by flipping signs of individual elements.
    Only valid for +/- 1 vectors? No, we can adjust continuous values.
    But the task says "signed step heights". It doesn't strictly say +/- 1.
    However, "signed" often implies discrete.
    Let's stick to continuous optimization primarily.
    If we want to do local search, we can try small perturbations.
    But L-BFGS with analytical gradient should be sufficient.
    """
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    best_score = 1e10
    best_h = np.ones(N)
    
    np.random.seed(42)
    
    # Initializations
    inits = []
    inits.append(np.ones(N))
    for _ in range(5):
        inits.append(np.random.choice([-1, 1], N))
    for _ in range(5):
        inits.append(np.random.uniform(-1, 1, N))
    
    # Temperatures for annealing
    temps = [10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.01]
    
    for h0 in inits:
        if time.time() > deadline - 10:
            break
            
        current_h = h0.copy()
        
        for temp in temps:
            if time.time() > deadline - 10:
                break
                
            # Optimize with current temp
            new_h = _lbfgs_optimize(current_h, temp, deadline)
            
            if new_h is not None and not np.any(np.isnan(new_h)):
                current_h = new_h
            
            # Check actual score
            actual_score = _score(current_h)
            if actual_score < best_score:
                best_score = actual_score
                best_h = current_h.copy()
                
            # Optional: If we have time, do a quick local search?
            # For now, rely on L-BFGS.
            
    # Final high-precision polish
    if time.time() < deadline - 5:
        # Use very small temperature
        new_h = _lbfgs_optimize(best_h, 0.001, deadline)
        if new_h is not None and not np.any(np.isnan(new_h)):
            actual_score = _score(new_h)
            if actual_score < best_score:
                best_score = actual_score
                best_h = new_h
    
    return best_h

# EVOLVE-BLOCK-END
