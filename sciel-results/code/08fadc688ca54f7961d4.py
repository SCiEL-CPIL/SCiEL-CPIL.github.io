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

def _objective_and_grad(h, beta):
    """
    Compute soft-max smoothed objective and gradient.
    Objective: 2*N*soft_max(conv) / (sum(h))^2
    We minimize this (without abs since we'll handle sign).
    """
    s = np.sum(h)
    if abs(s) < 1e-10:
        return 1e10, np.zeros_like(h)
    
    conv = _fft_autoconv(h)
    # Soft max with parameter beta
    m = np.max(conv)
    shifted = beta * (conv - m)
    shifted = np.clip(shifted, -500, 500)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted)
    soft_max_val = m + np.log(sum_exp) / beta
    
    # Gradient of soft_max w.r.t. conv
    weights = exp_shifted / sum_exp  # softmax weights
    
    # Objective: 2*N*soft_max / s^2
    obj = 2.0 * N * soft_max_val / (s * s)
    
    # Gradient: d(obj)/d(h_i)
    # d(soft_max)/d(h_i) = sum_k weights[k] * d(conv[k])/d(h_i)
    # conv[k] = sum_j h[j] * h[j+k-N+1] (for the full convolution indexing)
    # Actually conv is the full convolution, so conv[m] = sum_j h[j]*h[j+m-(N-1)]
    # Let's use the FFT gradient approach
    
    # Gradient of autoconvolution w.r.t. h:
    # If conv = ifft(fft(h)^2), then d(conv[k])/d(h[i]) = sum_j (h[j] if j+k=N-1+i else 0) + h[i+k-(N-1)]... 
    # This is complex. Let's use a different approach.
    
    # d(conv[k])/d(h[i]) for full convolution:
    # conv[k] = sum_j h[j] * h[j + k - (N-1)]
    # d(conv[k])/d(h[i]) = h[i + k - (N-1)] (if 0 <= i+k-(N-1) < N) + h[i] * delta(k-(N-1)-i)... 
    
    # Actually simpler: conv = h * h (linear convolution)
    # d(conv)/d(h) is the convolution of the gradient.
    # Using FFT: grad_conv = ifft(2*fft(h) * fft(grad_h)) but we need per-element.
    
    # Let's compute gradient numerically or use the structure:
    # d(conv[k])/d(h[i]) = sum over j where j = i: h[k-(N-1)+i] + h[i] * [k-(N-1)+i == j]
    # This is getting complex. Let me use a simpler gradient computation.
    
    # Alternative: use the fact that for soft max, 
    # d(soft_max)/d(h) = sum_k weights[k] * d(conv[k])/d(h)
    # d(conv[k])/d(h[i]) = h[i+k-(N-1)] + h[i+k-(N-1)]... 
    
    # For linear convolution conv = np.convolve(h, h):
    # conv[k] = sum_{j=0}^{N-1} h[j] * h[j+k-(N-1)] where h is zero-padded outside [0,N-1]
    # d(conv[k])/d(h[i]) = h[i+k-(N-1)] + (if i+k-(N-1) is in range, that's it, plus if j=i then h[k-(N-1)+i])
    # Actually: d(conv[k])/d(h[i]) = h[i+k-(N-1)] if 0<=i+k-(N-1)<N, plus h[i] * delta(k-(N-1))... 
    
    # Let me just do: for each k, d(conv[k])/d(h[i]) = h[i+k-(N-1)] (when valid) + h[i+k-(N-1)] (when j=i gives h[i+k-(N-1)])
    # Wait, conv[k] = sum_j h[j]*h[j+k-(N-1)]
    # d/d(h[i]) = sum_j [delta(j,i)*h[j+k-(N-1)] + h[j]*delta(j+k-(N-1),i)]
    # = h[i+k-(N-1)] + h[i-k+(N-1)] (when indices valid)
    
    # So d(conv[k])/d(h[i]) = h[i+k-(N-1)] + h[i-k+(N-1)] (with h out of range = 0)
    
    # This means the gradient of soft_max w.r.t. h[i] is:
    # sum_k weights[k] * (h[i+k-(N-1)] + h[i-k+(N-1)])
    
    # This is a convolution of weights with h!
    # grad_sm[i] = sum_k weights[k] * h[i+k-(N-1)] + sum_k weights[k] * h[i-k+(N-1)]
    # = (weights * h)[i] + (weights_rev * h)[i] where * is appropriate convolution
    
    # Let's compute this via FFT for efficiency
    # First term: sum_k weights[k] * h[i+k-(N-1)]
    # Let l = k-(N-1), then sum_l weights[l+N-1] * h[i+l] for l in [-(N-1), N-1]
    # This is cross-correlation or convolution depending on indexing
    
    # Simpler: just compute directly for N=400 (it's fast enough)
    grad_sm = np.zeros(N)
    for k in range(2*N-1):
        if weights[k] > 1e-15:
            # d(conv[k])/d(h[i]) = h[i+k-(N-1)] + h[i-k+(N-1)]
            # First term: h[i+k-(N-1)] for i in [0,N-1], valid when 0<=i+k-(N-1)<N
            lo1 = max(0, N-1-k)
            hi1 = min(N, N-1-k+N)
            if lo1 < hi1:
                idx1 = np.arange(lo1, hi1)
                grad_sm[idx1] += weights[k] * h[idx1+k-(N-1)]
            # Second term: h[i-k+(N-1)] for i in [0,N-1], valid when 0<=i-k+(N-1)<N
            lo2 = max(0, k-(N-1))
            hi2 = min(N, k-(N-1)+N)
            if lo2 < hi2:
                idx2 = np.arange(lo2, hi2)
                grad_sm[idx2] += weights[k] * h[idx2-k+(N-1)]
    
    # Now obj = 2*N*soft_max / s^2
    # d(obj)/d(h[i]) = 2*N*(d(soft_max)/d(h[i]) * s^2 - soft_max * 2*s * 1) / s^4
    # = 2*N*(grad_sm[i] / s - 2*soft_max / s^2)
    grad_obj = 2.0 * N * (grad_sm / s - 2.0 * soft_max_val / (s * s))
    
    return obj, grad_obj

def _refine(h0, deadline):
    """Use scipy minimize with gradient to refine."""
    h0 = h0.copy()
    s0 = np.sum(h0)
    if abs(s0) < 1e-10:
        return h0
    
    best_h = h0.copy()
    best_val = _compute_value(h0)
    
    for beta in [20.0, 50.0, 100.0, 200.0]:
        if time.time() > deadline:
            break
        
        def obj_grad(x):
            o, g = _objective_and_grad(x, beta)
            return o, g
        
        try:
            result = minimize(obj_grad, h0, jac=True, method='L-BFGS-B',
                            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            if result.success or result.fun < best_val:
                h_candidate = result.x
                val_candidate = _compute_value(h_candidate)
                if val_candidate < best_val:
                    best_val = val_candidate
                    best_h = h_candidate.copy()
        except Exception:
            pass
    
    return best_h

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_h = np.ones(N)
    best_val = _compute_value(best_h)
    
    # Generate diverse initializations
    x = np.linspace(0, 1, N)
    initializations = []
    
    # Chirp signals (key structural change)
    for theta in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
        z = np.exp(1j * theta)
        h_chirp = np.real(z ** np.arange(N))
        # Add small positive offset to ensure sum > 0
        h_chirp = h_chirp + 0.5
        initializations.append(h_chirp)
    
    # Cosine window
    initializations.append(0.5 + 0.5 * np.cos(2 * np.pi * x))
    
    # Sine oscillatory
    initializations.append(1.0 + 0.5 * np.sin(2 * np.pi * x))
    initializations.append(1.0 + 0.3 * np.sin(4 * np.pi * x))
    initializations.append(1.0 + 0.7 * np.sin(2 * np.pi * 3 * x))
    
    # Two-lobe
    initializations.append(1.0 + 0.5 * np.sin(np.pi * x))
    
    # Uniform with noise
    np.random.seed(42)
    initializations.append(np.ones(N) + 0.05 * np.random.randn(N))
    
    # Triangular
    tri = np.minimum(x, 1 - x) * 2
    initializations.append(0.5 + 0.5 * tri)
    
    # Gaussian
    initializations.append(np.exp(-0.5 * ((x - 0.5) * 4) ** 2))
    
    # Bump functions
    for center in [0.25, 0.5, 0.75]:
        width = 0.15
        initializations.append(np.exp(-0.5 * ((x - center) / width) ** 2))
    
    # Multi-bump
    initializations.append(0.5 * np.exp(-0.5 * ((x - 0.3) / 0.1) ** 2) + 0.5 * np.exp(-0.5 * ((x - 0.7) / 0.1) ** 2))
    
    # Cosine with phase shift
    for phase in [0.25, 0.5, 0.75]:
        initializations.append(0.5 + 0.5 * np.cos(2 * np.pi * (x - phase)))
    
    # Random
    for seed in [1, 2, 3, 4, 5]:
        np.random.seed(seed)
        initializations.append(1.0 + 0.3 * np.random.randn(N))
    
    best_overall = best_h.copy()
    best_overall_val = best_val
    
    for h0 in initializations:
        if time.time() > deadline:
            break
        
        # Ensure sum is not zero and positive
        s0 = np.sum(h0)
        if abs(s0) < 1e-10:
            h0 = h0 + 1.0
            s0 = np.sum(h0)
        
        # Normalize so sum = N for consistency
        h0 = h0 / s0 * N
        
        # Evaluate initial
        val0 = _compute_value(h0)
        if val0 < best_overall_val:
            best_overall_val = val0
            best_overall = h0.copy()
        
        # Refine with gradient-based optimization
        h_refined = _refine(h0, deadline)
        val_refined = _compute_value(h_refined)
        
        if val_refined < best_overall_val:
            best_overall_val = val_refined
            best_overall = h_refined.copy()
    
    # Final refinement on best
    if time.time() < deadline - 10:
        h_final = _refine(best_overall, deadline)
        val_final = _compute_value(h_final)
        if val_final < best_overall_val:
            best_overall_val = val_final
            best_overall = h_final.copy()
    
    # Safety: ensure valid output
    s_final = np.sum(best_overall)
    if abs(s_final) < 1e-10:
        return np.ones(N)
    
    return best_overall
# EVOLVE-BLOCK-END
