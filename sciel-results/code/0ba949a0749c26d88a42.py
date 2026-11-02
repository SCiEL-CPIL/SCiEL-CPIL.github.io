"""Spectral shaping + coordinate descent for autoconvolution minimization."""
import numpy as np
import time
from scipy.optimize import minimize

N = 600
DEADLINE = 145.0

def _autoconvolve_max_ratio(h):
    """Compute max_x (f*f)(x) / (∫f)^2 for step function h on N equal intervals."""
    h = np.asarray(h, dtype=np.float64)
    w = 1.0 / (2.0 * N)
    integral = w * np.sum(h)
    if integral <= 1e-12:
        return 1e10
    n_fft = 4 * N
    h_padded = np.zeros(n_fft)
    h_padded[:N] = h
    F = np.fft.fft(h_padded)
    conv = np.real(np.fft.ifft(F * np.conj(F)))[:2*N-1]
    max_conv = np.max(conv)
    return max_conv * w / (integral * integral)

def _autoconvolve(h):
    """Return the autoconvolution array."""
    h = np.asarray(h, dtype=np.float64)
    n_fft = 4 * N
    h_padded = np.zeros(n_fft)
    h_padded[:N] = h
    F = np.fft.fft(h_padded)
    conv = np.real(np.fft.ifft(F * np.conj(F)))[:2*N-1]
    return conv

def _project_simplex(v, s=1.0):
    """
    Project vector v onto the simplex {x >= 0, sum(x) = s}.
    Uses the algorithm from Duchi et al. (2010).
    """
    v = np.asarray(v, dtype=np.float64)
    n = len(v)
    if s <= 0:
        return np.zeros(n)
    
    # Sort v in descending order
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
    theta = (cssv[rho] - s) / (rho + 1.0)
    w = np.maximum(v - theta, 0)
    return w

def _subgradient_descent(h_init, deadline):
    """
    Projected Subgradient Descent for minimizing max(autoconvolution).
    """
    h = np.asarray(h_init, dtype=np.float64).copy()
    # Normalize to simplex
    s = np.sum(h)
    if s < 1e-12:
        h = np.ones(N) / N
    else:
        h = h / s
        
    # Initial step size
    # The Lipschitz constant of the gradient of the max autoconvolution is related to the norm of h.
    # Let's start with a moderate step size and decay it.
    alpha = 0.01
    alpha_decay = 0.99
    best_h = h.copy()
    best_val = _autoconvolve_max_ratio(h)
    
    n_iter = 0
    while time.time() < deadline:
        conv = _autoconvolve(h)
        max_val = np.max(conv)
        # Find indices of near-max values (to handle flat peaks)
        # Use a small tolerance
        threshold = max_val * 0.999
        mask = conv >= threshold
        # If no mask (shouldn't happen), take the argmax
        if not np.any(mask):
            mask = np.zeros(len(conv), dtype=bool)
            mask[np.argmax(conv)] = True
        
        # Compute subgradient
        # grad c_k = h_{k-i} + h_i (shifted)
        # We need sum over k in mask of grad c_k
        # grad_h_j = sum_{k in mask} (h_{k-j} + h_j)
        # This can be computed via cross-correlation of the mask with h.
        
        # Let M be the mask vector (length 2N-1)
        M = mask.astype(np.float64)
        # We want g_j = sum_k M_k (h_{k-j} + h_j)
        # Note: h is defined on 0..N-1. conv is on 0..2N-2.
        # h_{k-j} is non-zero only if 0 <= k-j < N.
        
        # Term 1: sum_k M_k h_{k-j}
        # This is cross-correlation of M and h.
        # Let's use FFT for efficiency.
        n_fft = 4 * N
        M_padded = np.zeros(n_fft)
        M_padded[:len(M)] = M
        h_padded = np.zeros(n_fft)
        h_padded[:N] = h
        
        # Cross correlation: ifft(fft(M) * conj(fft(h)))
        # This gives sum_i M_i h_{i-j} ?? No, standard correlation.
        # Let's verify indices.
        # conv_k = sum_i h_i h_{k-i}
        # d conv_k / d h_j = h_{k-j} + h_j (if j in range, k-j in range)
        # Sum_k M_k h_{k-j} is the cross-correlation of M and h at lag j.
        # Correlation R_Mh[j] = sum_k M_k h_{k-j}
        # This is ifft(fft(M) * conj(fft(h))) if M and h are zero-padded.
        # Let's check the shift.
        # Standard FFT correlation: ifft(fft(a) * conj(fft(b)))_k = sum_i a_{i} b_{i-k}
        # We want sum_k M_k h_{k-j}. Let i=k. sum_i M_i h_{i-j}.
        # This matches R_Mh[j] if we define R_Mh[j] = sum_i M_i h_{i-j}.
        # So we need ifft(fft(M) * conj(fft(h))).
        
        cross_corr = np.real(np.fft.ifft(np.fft.fft(M_padded) * np.conj(np.fft.fft(h_padded))))
        # The result is circular. We need the linear part.
        # The length of M is 2N-1. Length of h is N.
        # The correlation length is (2N-1) + N - 1 = 3N-1? No.
        # M has support [0, 2N-2]. h has support [0, N-1].
        # h_{k-j} is non-zero for k-j in [0, N-1] => j in [k-N+1, k].
        # We want g_j for j in [0, N-1].
        # The term sum_k M_k h_{k-j} for a fixed j is the correlation at lag j.
        # The FFT method gives circular correlation.
        # Since we padded to 4N, and max support sum is 3N-1, linear correlation is preserved in the first 3N-1 points?
        # Actually, ifft(fft(M)*conj(fft(h))) gives sum_i M_i h_{i-j} (circular).
        # With zero padding, it becomes linear.
        # The index j in the result corresponds to the lag.
        # We need the values for lags 0 to N-1.
        # In the FFT output, the lags are 0, 1, ..., N_fft-1.
        # The linear correlation R[j] is at index j.
        # So we take cross_corr[0:N].
        
        term1 = cross_corr[:N]
        
        # Term 2: sum_k M_k h_j = h_j * sum_k M_k
        sum_M = np.sum(M)
        term2 = h * sum_M
        
        grad = term1 + term2
        
        # Normalize gradient?
        # The objective is max conv. We want to decrease it.
        # Update: h = h - alpha * grad
        # Then project to simplex.
        
        h_new = h - alpha * grad
        h_new = _project_simplex(h_new, s=1.0)
        
        # Check improvement
        val_new = _autoconvolve_max_ratio(h_new)
        if val_new < best_val:
            best_val = val_new
            best_h = h_new.copy()
            # Maybe increase alpha slightly if we are making progress?
            # Or keep it.
            h = h_new
        else:
            # If no improvement, decay alpha more aggressively
            alpha *= 0.9
            # Also, maybe take a smaller step
            h_new2 = h - alpha * grad
            h_new2 = _project_simplex(h_new2, s=1.0)
            val_new2 = _autoconvolve_max_ratio(h_new2)
            if val_new2 < best_val:
                best_val = val_new2
                best_h = h_new2.copy()
                h = h_new2
            else:
                # If still no improvement, just decay
                pass
                
        alpha *= alpha_decay
        n_iter += 1
        if n_iter % 100 == 0:
            # Periodic check
            pass
            
    return best_h, best_val

def construct():
    start_time = time.time()
    
    # Phase 1: Discrete support search (kept from original, but simplified)
    x = np.linspace(-0.5, 0.5, N, endpoint=False)
    
    best_h = np.ones(N)
    best_val = _autoconvolve_max_ratio(best_h)
    
    # Try Gaussian
    h_gauss = np.exp(-0.5 * (x / 0.2) ** 2)
    val_gauss = _autoconvolve_max_ratio(h_gauss)
    if val_gauss < best_val:
        best_val = val_gauss
        best_h = h_gauss.copy()
        
    # Try Triangle
    h_tri = np.maximum(0, 1 - np.abs(x) / 0.3)
    val_tri = _autoconvolve_max_ratio(h_tri)
    if val_tri < best_val:
        best_val = val_tri
        best_h = h_tri.copy()
        
    # Phase 2: Subgradient Descent from best initial
    # Use a generous deadline for SGD
    sgd_deadline = DEADLINE - 10
    h_sgd, val_sgd = _subgradient_descent(best_h, sgd_deadline)
    if val_sgd < best_val:
        best_val = val_sgd
        best_h = h_sgd.copy()
        
    # Phase 3: Quick L-BFGS-B polish if time remains
    if time.time() < DEADLINE - 5:
        try:
            h_norm = best_h / np.sum(best_h)
            def objective(hf):
                h_local = np.maximum(hf, 0.0)
                sl = np.sum(h_local)
                if sl < 1e-12: return 1e10
                h_local = h_local / sl
                conv = _autoconvolve(h_local)
                t = 0.01
                m = np.max(conv)
                if m < 1e-300: return 0.0
                shifted = np.clip(conv - m, -100, 100)
                return m + t * np.log(np.sum(np.exp(shifted / t)))
                
            result = minimize(objective, h_norm, method='L-BFGS-B', 
                             bounds=[(0, None)]*N, options={'maxiter': 100, 'ftol': 1e-12})
            if result.success and result.fun < best_val:
                h_new = np.maximum(result.x, 0.0)
                s_new = np.sum(h_new)
                if s_new > 1e-12:
                    h_new = h_new / s_new
                    val_new = _autoconvolve_max_ratio(h_new)
                    if val_new < best_val:
                        best_val = val_new
                        best_h = h_new.copy()
        except Exception:
            pass
            
    # Ensure non-negative and normalized
    best_h = np.maximum(best_h, 0.0)
    s = np.sum(best_h)
    if s < 1e-12:
        best_h = np.ones(N)
    else:
        best_h = best_h / s
        
    return best_h.tolist()
