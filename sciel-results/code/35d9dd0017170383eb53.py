"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 400
_FSIZE = 1
while _FSIZE < 2 * N - 1:
    _FSIZE *= 2

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

N = 400
_FSIZE = 1
while _FSIZE < 2 * N - 1:
    _FSIZE *= 2

def _autoconv_full(h):
    """Full linear autoconvolution via FFT."""
    H = np.fft.rfft(h, _FSIZE)
    ac = np.fft.irfft(H * H, _FSIZE)
    return ac[:2*N-1]

def compute_raw_value(x):
    """
    Computes the raw value: max(|autoconvolution|) / (sum(x))^2.
    Assumes x is a 1D numpy array.
    """
    X = np.fft.rfft(x, _FSIZE)
    conv = np.fft.irfft(X * X, _FSIZE)
    conv = conv[:2*N-1]
    max_abs_conv = np.max(np.abs(conv))
    integral = np.sum(x)
    if abs(integral) < 1e-12:
        return float('inf')
    return max_abs_conv / (integral * integral)

def _soft_max_abs_and_grad(h, beta=100.0):
    """
    Computes the soft-max absolute autoconvolution value and its analytical gradient.
    Returns: value, gradient
    """
    s = np.sum(h)
    if abs(s) < 1e-9:
        return 1e10, np.zeros_like(h)
    
    ac = _autoconv_full(h)
    abs_ac = np.abs(ac)
    m = np.max(abs_ac)
    
    if m < 1e-15:
        S = 0.0
        grad_S = np.zeros_like(h)
    else:
        # Soft max approximation: max(x) ~ (1/beta) * log(sum(exp(beta * x)))
        # We want max(|ac|). Let y = |ac|.
        # To avoid overflow, we subtract m from y before exp.
        scaled = beta * (abs_ac - m)
        # Clip to prevent underflow/overflow issues in exp
        scaled = np.clip(scaled, -50, 50)
        exp_scaled = np.exp(scaled)
        sum_exp = np.sum(exp_scaled)
        log_sum = np.log(sum_exp)
        S = m + log_sum / beta
        
        # Weights for the soft-max
        w = exp_scaled / sum_exp
        
        # Gradient of S with respect to ac_k
        # dS/dac_k = w_k * sign(ac_k)
        # Note: sign(0) is 0, which is fine as the derivative of |x| at 0 is undefined 
        # but in soft max it's smoothed.
        
        v = w * np.sign(ac)
        
        # Gradient of S w.r.t h
        # S = f(ac)
        # ac_k = sum_i h_i h_{k-i}
        # dS/dh_j = sum_k (dS/dac_k) * (dac_k/dh_j)
        # dac_k/dh_j = h_{k-j} + h_{j-k} (since autoconv is symmetric, and derivative of x*x is 2x shifted)
        # Actually, ac = h * h. 
        # d(ac)/dh_j is a vector where element k is h_{k-j} if k>=j else h_{j-k} ? No.
        # ac_k = sum_m h_m h_{k-m}.
        # d(ac_k)/dh_j = h_{k-j} (if k>=j) + h_{j-k} (if k<j)? 
        # Let's use correlation.
        # dS/dh_j = 2 * sum_k v_k * (d ac_k / d h_j) / 2 ? No, v already includes the derivative chain from S to ac.
        # dS/dh_j = sum_k v_k * (h_{k-j} + h_{j-k}) ? 
        # Wait, ac_k = sum_m h_m h_{k-m}.
        # If we vary h_j, it affects terms where m=j or k-m=j.
        # Term 1: m=j -> h_j h_{k-j}. Deriv w.r.t h_j is h_{k-j}.
        # Term 2: k-m=j -> m=k-j. h_{k-j} h_j. Deriv w.r.t h_j is h_{k-j}.
        # So d ac_k / d h_j = h_{k-j} + h_{j-k} ?
        # Since h is real, h_{k-j} is defined for k>=j. h_{j-k} is defined for k<j.
        # This looks like 2 * h_{|k-j|} if we consider even/odd symmetry? No, h is not necessarily symmetric.
        # Let's use the property that autoconvolution gradient is 2 * cross_correlation(v, h).
        # Let C = v * h (cross correlation).
        # C_j = sum_k v_k h_{k-j} ?
        
        # Let's use FFT for efficiency.
        # We want G_j = sum_k v_k * (h_{k-j} + h_{j-k})
        # Note that h_{j-k} = h_{-(k-j)}.
        # This is effectively 2 * (v * h)_j if we consider the full convolution?
        # Actually, gradient of ||h*h||_inf approx is related to correlation of v and h.
        
        # Let's compute Cross Correlation R = np.correlate(v, h, mode='full')
        # R[k] = sum_n v[n+k] conj(h[n])
        # We need sum_k v_k h_{k-j}.
        # Let m = k-j => k = m+j.
        # Sum_m v[m+j] h[m].
        # This is exactly the cross correlation of v and h at lag j.
        # R[j] = sum_m v[m+j] h[m].
        
        # So dS/dh_j = sum_k v_k h_{k-j} + sum_k v_k h_{j-k}
        # The second term: sum_k v_k h_{j-k}. Let l = j-k => k = j-l.
        # Sum_l v[j-l] h[l].
        # This is cross correlation of v and h at lag -j? Or j?
        # R[-j] would be sum_m v[m-j] h[m].
        # It turns out that for real vectors, the gradient of the autoconvolution sum is 2 * R[j] if we define R appropriately, 
        # but we must be careful with indices.
        
        # Let's just use np.correlate for clarity and correctness, N=400 is small.
        # v has length 2N-1, h has length N.
        # We need the correlation values for lags 0 to N-1.
        
        # np.correlate(v, h, mode='full') returns length (2N-1) + N - 1 = 3N-2.
        # The lag 0 is at index N-1.
        # Positive lags j > 0 are at index N-1 + j.
        # Negative lags j < 0 are at index N-1 + j.
        
        corr = np.correlate(v, h, mode='full')
        # We need dS/dh_j for j=0..N-1.
        # Term 1: sum_k v_k h_{k-j}. This is corr at lag j? 
        # corr[k] = sum_n v[n+k] h[n].
        # If we want sum_k v_k h_{k-j}, let n = k-j => k = n+j.
        # Sum_n v[n+j] h[n]. This is corr[j].
        
        # Term 2: sum_k v_k h_{j-k}. Let m = j-k => k = j-m.
        # Sum_m v[j-m] h[m].
        # This is sum_m v[-(m-j)] h[m].
        # This is corr[-j]?
        # corr[-j] = sum_n v[n-j] h[n]. Let m=n-j => n=m+j.
        # Sum_m v[m] h[m+j]. Not quite.
        
        # Let's stick to the FFT method which is cleaner for "gradient of quadratic form".
        # S = sum_k v_k ac_k
        # ac = ifft(fft(h)^2)
        # grad_S = 2 * ifft(fft(v) * fft(h)) ?
        # Let's verify.
        # L = sum_k v_k (h*h)_k = <v, h*h> = <v * h, h> (where * is convolution? No, inner product).
        # <v, h*h> = sum_k v_k sum_m h_m h_{k-m} = sum_m h_m sum_k v_k h_{k-m}
        # Inner sum is cross correlation of v and h at lag m?
        # C_m = sum_k v_k h_{k-m}.
        # So grad_S_m = 2 * C_m ? (Since h_m appears in two terms in the sum? No, h is the variable).
        # If L = sum_m h_m C_m, then dL/dh_m = C_m.
        # But C_m depends on h? No, v is fixed (weights from soft max).
        # So grad_S = C.
        
        # How to compute C_m = sum_k v_k h_{k-m} for all m?
        # This is cross correlation of v and h.
        # Using FFT:
        # V = fft(v, FSIZE_CORR)
        # H = fft(h, FSIZE_CORR)
        # C = ifft(V * conj(H)) ?
        # Standard correlation: r[m] = sum_n a[n+m] b[n].
        # Here a=v, b=h.
        # r[m] = sum_n v[n+m] h[n].
        # We want C_m = sum_k v_k h_{k-m}.
        # Let n = k-m => k = n+m.
        # C_m = sum_n v[n+m] h[n].
        # This is exactly r[m].
        
        # So we need r[0..N-1].
        
        fsize_corr = 1
        while fsize_corr < (2*N-1) + N - 1:
            fsize_corr *= 2
            
        V = np.fft.rfft(v, fsize_corr)
        H = np.fft.rfft(h, fsize_corr)
        # Correlation via FFT: ifft(fft(a) * conj(fft(b))) gives sum a[n+m] b[n]
        R = np.fft.irfft(V * np.conj(H), fsize_corr)
        
        # R[m] corresponds to lag m.
        # We need m from 0 to N-1.
        grad_S = 2 * R[:N]
        
    # Objective f = S / s^2
    # df/dh_i = (dS/dh_i * s^2 - S * 2 * s) / s^4
    # = (dS/dh_i * s - 2 * S) / s^3
    
    grad = (grad_S * s - 2 * S) / (s ** 3)
    
    return S / (s * s), grad

def _smooth_obj_with_grad(h, beta=100.0):
    """
    Objective and gradient for scipy.optimize.minimize.
    """
    val, grad = _soft_max_abs_and_grad(h, beta)
    # Add a small regularization to keep h away from zero sum? 
    # The problem requires sum != 0. 
    # If sum is close to 0, value is inf. 
    # We can add a penalty if |sum| < 1.
    s = np.sum(h)
    if abs(s) < 1.0:
        # Penalty to push sum away from 0
        pen = 1.0 / (abs(s) + 1e-6)
        # Gradient of penalty
        # d/dh (1/(|s|+eps)) = -sign(s) / (|s|+eps)^2
        gp = -np.sign(s) / (abs(s) + 1e-6)**2
        return val + pen, grad + gp
    return val, grad

def _local_search_flip(h, max_passes=20, deadline=None):
    """
    Local optimization: try flipping each element's sign, keep if improves score.
    Uses FFT incrementally for speedup.
    """
    h = h.copy()
    h = np.sign(h)
    h[h == 0] = 1
    best_val = compute_raw_value(h)
    H = np.fft.rfft(h, _FSIZE)
    
    # Precompute frequency basis vectors for speed
    freqs = np.arange(_FSIZE//2+1)
    basis_cache = {}
    
    for _ in range(max_passes):
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        for i in range(N):
            if deadline is not None and time.time() > deadline:
                break
            # Flip sign: h[i] -> -h[i]
            old_val = h[i]
            h[i] = -old_val
            # Update FFT incrementally
            if i not in basis_cache:
                basis_cache[i] = np.exp(-2j * np.pi * freqs * i / _FSIZE)
            delta = -2 * old_val * basis_cache[i]
            H_new = H + delta
            conv = np.fft.irfft(H_new * H_new, _FSIZE)[:2*N-1]
            integral = np.sum(h)
            if abs(integral) < 1e-12:
                new_val = float('inf')
            else:
                new_val = np.max(np.abs(conv)) / (integral * integral)
            
            if new_val < best_val:
                best_val = new_val
                H = H_new
                improved = True
            else:
                h[i] = old_val  # revert
        if not improved:
            break
    return h, best_val

def construct():
    deadline = time.time() + 140.0  # safety margin before 150s kill
    
    # Phase 1: Continuous Optimization
    # Initialize with random signs, then relax to continuous
    rng = np.random.RandomState(42)
    h_init = rng.choice([-1.0, 1.0], size=N)
    
    # Try a few random initializations for the continuous phase
    best_cont_h = None
    best_cont_val = float('inf')
    
    for _ in range(5):
        h0 = rng.choice([-1.0, 1.0], size=N)
        try:
            res = minimize(_smooth_obj_with_grad, h0, method='L-BFGS-B', 
                           jac=True, options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-8})
            if res.success or res.fun < best_cont_val:
                best_cont_val = res.fun
                best_cont_h = res.x
        except Exception:
            continue
        
        if time.time() > deadline - 60:
            break
            
    if best_cont_h is None:
        # Fallback to random
        best_cont_h = h_init
        
    # Phase 2: Discretization
    h_binary = np.sign(best_cont_h)
    h_binary[h_binary == 0] = 1.0
    
    # Phase 3: Local Search Refinement
    # Use the fast incremental FFT local search
    h_final, final_val = _local_search_flip(h_binary, max_passes=50, deadline=deadline)
    
    # If we have time, try a few random restarts of the whole pipeline
    while time.time() < deadline - 20:
        h0 = rng.choice([-1.0, 1.0], size=N)
        # Quick continuous relax
        try:
            res = minimize(_smooth_obj_with_grad, h0, method='L-BFGS-B', 
                           jac=True, options={'maxiter': 100, 'ftol': 1e-7, 'gtol': 1e-7})
            h_c = res.x
            h_b = np.sign(h_c)
            h_b[h_b == 0] = 1.0
            
            h_test, val_test = _local_search_flip(h_b, max_passes=10, deadline=deadline)
            if val_test < final_val:
                final_val = val_test
                h_final = h_test
        except Exception:
            pass
            
    return h_final
# EVOLVE-BLOCK-END
