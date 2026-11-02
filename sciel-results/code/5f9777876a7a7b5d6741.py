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


def _spectral_to_heights(sigma, center=0.0, n_bins=N):
    """
    Generate step heights from a Gaussian power spectrum.
    The autocorrelation of a Gaussian is a Gaussian, so this flattens the peak.
    sigma controls the width in frequency domain.
    """
    freqs = np.fft.fftfreq(n_bins, d=1.0/(2*N))
    # Power spectrum: Gaussian centered at 'center' frequency
    power = np.exp(-0.5 * ((freqs - center) / sigma) ** 2)
    # Amplitude is sqrt of power (for non-negative signal, use real part)
    # Actually, we need h such that |FFT(h)|^2 ~ power
    # For a real non-negative signal, h itself should be positive
    # Use the inverse FFT of the power spectrum's sqrt as amplitude
    # Simpler: just use the Gaussian in time domain as a good starting point
    x = np.linspace(-0.5, 0.5, n_bins, endpoint=False)
    # In time domain, a Gaussian with width sigma_t gives Gaussian autocorrelation
    # The optimal is to make autocorrelation as flat as possible
    # A Gaussian h gives Gaussian autocorrelation - not flat
    # Better: use a "top-hat" or other shapes
    # For now, return a Gaussian
    h = np.exp(-0.5 * (x / (sigma * 0.3)) ** 2)
    return np.maximum(h, 0.0)


def _coordinate_descent_block(h, block_size, n_rounds, deadline):
    """
    Coordinate descent: optimize each block of 'block_size' consecutive bins
    by treating the rest as fixed and minimizing the max autoconvolution.
    """
    h = np.asarray(h, dtype=np.float64).copy()
    n_blocks = N // block_size
    
    for round_idx in range(n_rounds):
        if time.time() > deadline:
            break
        for b in range(n_blocks):
            if time.time() > deadline:
                break
            start = b * block_size
            end = start + block_size
            block = h[start:end].copy()
            
            # Compute the "background" contribution to autoconvolution from fixed parts
            # conv_k = sum_i h_i h_{k-i}
            # Split h into block part and rest
            rest = h.copy()
            rest[start:end] = 0.0
            
            # conv = autoconv(block) + cross(block, rest)
            # cross(block, rest) = sum over block indices i, rest indices j: block_i rest_j at position i+j
            # and symmetric
            
            # For efficiency, compute the contribution to max from the block
            # We'll just do a simple 1D search over block values
            
            # Actually, let's do a simpler approach: perturb the block
            # and keep if it improves
            best_block = block.copy()
            best_val = _autoconvolve_max_ratio(h)
            
            # Try scaling the block
            for scale in [0.5, 0.7, 0.8, 0.9, 1.1, 1.2, 1.5]:
                h_test = h.copy()
                h_test[start:end] = block * scale
                val = _autoconvolve_max_ratio(h_test)
                if val < best_val:
                    best_val = val
                    best_block = block * scale
            
            # Try shifting the block (moving mass)
            # This is expensive, skip for now
            
            h[start:end] = best_block
    
    return h


def _discrete_support_search(deadline):
    """
    Search over discrete support configurations.
    Key idea: if we put mass only on certain bins, the autoconvolution
    has a specific structure that can be analyzed.
    """
    best_h = None
    best_val = 1e10
    
    # Try various support patterns
    # Pattern 1: uniform (baseline)
    h = np.ones(N)
    val = _autoconvolve_max_ratio(h)
    if val < best_val:
        best_val = val
        best_h = h.copy()
    
    # Pattern 2: two separated bumps
    for gap_frac in [0.1, 0.15, 0.2, 0.25, 0.3]:
        if time.time() > deadline:
            break
        gap = int(gap_frac * N)
        half = (N - gap) // 2
        h = np.zeros(N)
        h[:half] = 1.0
        h[N-half:] = 1.0
        val = _autoconvolve_max_ratio(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Pattern 3: three bumps
    for gap_frac in [0.1, 0.15, 0.2]:
        if time.time() > deadline:
            break
        gap = int(gap_frac * N)
        seg = (N - 2*gap) // 3
        if seg <= 0:
            continue
        h = np.zeros(N)
        h[:seg] = 1.0
        h[N-seg:] = 1.0
        mid_start = (N - seg) // 2
        h[mid_start:mid_start+seg] = 1.0
        val = _autoconvolve_max_ratio(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Pattern 4: Gaussian-like
    x = np.linspace(-0.5, 0.5, N, endpoint=False)
    for width in [0.1, 0.15, 0.2, 0.25, 0.3]:
        if time.time() > deadline:
            break
        h = np.exp(-0.5 * (x / width) ** 2)
        val = _autoconvolve_max_ratio(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    # Pattern 5: Triangle
    for width in [0.2, 0.25, 0.3, 0.35, 0.4]:
        if time.time() > deadline:
            break
        h = np.maximum(0, 1 - np.abs(x) / width)
        val = _autoconvolve_max_ratio(h)
        if val < best_val:
            best_val = val
            best_h = h.copy()
    
    return best_h, best_val


def _refine_with_optimization(h_init, deadline):
    """
    Use scipy minimize to refine the heights.
    """
    h = np.asarray(h_init, dtype=np.float64).copy()
    s = np.sum(h)
    if s < 1e-12:
        return h
    
    h_norm = h / s
    bounds = [(0, None)] * N
    
    # Use a smooth max surrogate for differentiability
    def objective(hf):
        h_local = np.maximum(hf, 0.0)
        sl = np.sum(h_local)
        if sl < 1e-12:
            return 1e10
        h_local = h_local / sl
        conv = _autoconvolve(h_local)
        # Smooth max
        t = 0.01
        m = np.max(conv)
        if m < 1e-300:
            return 0.0
        shifted = np.clip(conv - m, -100, 100)
        return m + t * np.log(np.sum(np.exp(shifted / t)))
    
    def grad(hf):
        h_local = np.maximum(hf, 0.0)
        sl = np.sum(h_local)
        if sl < 1e-12:
            return np.zeros_like(hf)
        h_local = h_local / sl
        conv = _autoconvolve(h_local)
        t = 0.01
        m = np.max(conv)
        shifted = np.clip(conv - m, -100, 100)
        exp_vals = np.exp(shifted / t)
        sum_exp = np.sum(exp_vals)
        grad_conv = exp_vals / sum_exp
        
        # Gradient via FFT
        n_fft = 4 * N
        h_pad = np.zeros(n_fft)
        h_pad[:N] = h_local
        gc_pad = np.zeros(n_fft)
        gc_pad[:len(grad_conv)] = grad_conv
        
        # d(conv_k)/d(h_j) = h_{k-j} + h_j (with zero padding)
        # sum_k grad_conv_k * h_{k-j} = cross-correlation
        cc = np.real(np.fft.ifft(np.fft.fft(gc_pad) * np.conj(np.fft.fft(h_pad))))[:N]
        sum_grad = np.sum(grad_conv)
        grad_h = cc + h_local * sum_grad
        
        # Normalization gradient
        dot = np.dot(grad_h, h_local)
        grad_hvec = (grad_h / sl) - (h_local * dot) / (sl * sl)
        return grad_hvec
    
    try:
        result = minimize(objective, h_norm, method='L-BFGS-B', jac=grad,
                         bounds=bounds, options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
        if result.x is not None:
            h_new = np.maximum(result.x, 0.0)
            s_new = np.sum(h_new)
            if s_new > 1e-12:
                h_new = h_new / s_new
                return h_new
    except Exception:
        pass
    
    return h


def construct():
    start_time = time.time()
    
    # Phase 1: Discrete support search
    h_disc, val_disc = _discrete_support_search(DEADLINE - 60)
    best_h = h_disc.copy()
    best_val = val_disc
    
    # Phase 2: Refine with optimization from multiple starts
    inits = [best_h]
    
    # Add some structured initializations
    x = np.linspace(-0.5, 0.5, N, endpoint=False)
    
    # Gaussian
    h_gauss = np.exp(-0.5 * (x / 0.2) ** 2)
    inits.append(h_gauss)
    
    # Two bumps
    for gap in [0.15, 0.2, 0.25]:
        g = int(gap * N)
        half = (N - g) // 2
        h_2b = np.zeros(N)
        h_2b[:half] = 1.0
        h_2b[N-half:] = 1.0
        inits.append(h_2b)
    
    # Three bumps
    for gap in [0.1, 0.15, 0.2]:
        g = int(gap * N)
        seg = (N - 2*g) // 3
        if seg > 0:
            h_3b = np.zeros(N)
            h_3b[:seg] = 1.0
            h_3b[N-seg:] = 1.0
            mid = (N - seg) // 2
            h_3b[mid:mid+seg] = 1.0
            inits.append(h_3b)
    
    # Bimodal Gaussian
    h_bi = np.exp(-0.5 * ((x - 0.2) / 0.1) ** 2) + np.exp(-0.5 * ((x + 0.2) / 0.1) ** 2)
    inits.append(h_bi)
    
    # Triangle
    h_tri = np.maximum(0, 1 - np.abs(x) / 0.3)
    inits.append(h_tri)
    
    # Wide flat
    h_flat = np.ones(N) * 0.5 + 0.5 * np.cos(2 * np.pi * x)
    h_flat = np.maximum(h_flat, 0)
    inits.append(h_flat)
    
    # Optimize each initialization
    for h_init in inits:
        if time.time() > DEADLINE - 15:
            break
        
        h_init = np.maximum(h_init, 0.0)
        s = np.sum(h_init)
        if s < 1e-12:
            continue
        h_init = h_init / s
        
        # Quick coordinate descent first
        h_cd = _coordinate_descent_block(h_init, block_size=10, n_rounds=2, deadline=DEADLINE - 20)
        val_cd = _autoconvolve_max_ratio(h_cd)
        
        # Then full optimization
        h_opt = _refine_with_optimization(h_cd, DEADLINE - 15)
        val_opt = _autoconvolve_max_ratio(h_opt)
        
        if val_opt < best_val:
            best_val = val_opt
            best_h = h_opt.copy()
        elif val_cd < best_val:
            best_val = val_cd
            best_h = h_cd.copy()
    
    # Final coordinate descent polish
    if time.time() < DEADLINE - 5:
        h_polish = _coordinate_descent_block(best_h, block_size=5, n_rounds=3, deadline=DEADLINE - 2)
        val_polish = _autoconvolve_max_ratio(h_polish)
        if val_polish < best_val:
            best_h = h_polish
            best_val = val_polish
    
    # Ensure non-negative
    best_h = np.maximum(best_h, 0.0)
    s = np.sum(best_h)
    if s < 1e-12:
        best_h = np.ones(N)
    else:
        best_h = best_h / s
    
    return best_h.tolist()
