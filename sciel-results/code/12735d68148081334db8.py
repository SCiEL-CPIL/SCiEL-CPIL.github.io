import numpy as np
import time

N = 600
DELTA = 1.0 / N

# Precompute FFT size
_NFFT = 1 << (2 * N - 1).bit_length()

def compute_ratio_fast(h):
    """Compute max(f*f) / (int f)^2 for step function with heights h, using FFT."""
    h = np.maximum(np.asarray(h, dtype=np.float64), 0.0)
    s = np.sum(h) * DELTA
    if s < 1e-15:
        return 1e15
    h_norm = h / s  # normalize so int f = 1
    # FFT-based autoconvolution
    F = np.fft.rfft(h_norm, _NFFT)
    conv = np.fft.irfft(F * F, _NFFT)[:2*N-1]
    # For step functions, the continuous autoconvolution at grid points is DELTA * conv
    max_val = np.max(conv) * DELTA
    return max_val

def spectral_optimize(h0, max_iters=500, deadline=None):
    """Optimize in spectral domain: minimize L∞ norm of autocorrelation.
    
    Key insight: f*f in time domain corresponds to |F|² in frequency domain.
    We want to minimize max of inverse FFT of |F|² subject to F being the FFT
    of a nonnegative function. We use gradient descent on the log-spectrum
    with projection back to time domain (soft-thresholding for nonnegativity).
    """
    h = np.maximum(np.asarray(h0, dtype=np.float64), 0.0).copy()
    s = np.sum(h) * DELTA
    if s < 1e-15:
        return h, 1e15
    
    h = h / s  # normalize so ∫f = 1
    best_ratio = compute_ratio_fast(h)
    best_h = h.copy()
    
    # Work with the power spectrum
    F = np.fft.rfft(h, _NFFT)
    power = np.abs(F) ** 2
    phase = np.angle(F)
    
    # Optimize the log-power spectrum
    log_power = np.log(power + 1e-30)
    
    lr = 0.01
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        # Current time-domain function
        F_curr = np.sqrt(np.maximum(power, 0)) * np.exp(1j * phase)
        h_curr = np.fft.irfft(F_curr, _NFFT)[:N]
        h_curr = np.maximum(h_curr, 0)
        s_curr = np.sum(h_curr) * DELTA
        if s_curr < 1e-15:
            break
        h_curr = h_curr / s_curr
        
        # Compute autocorrelation
        F_h = np.fft.rfft(h_curr, _NFFT)
        autocorr = np.fft.irfft(F_h * F_h, _NFFT)[:2*N-1]
        max_val = np.max(autocorr) * DELTA
        
        if max_val < best_ratio - 1e-12:
            best_ratio = max_val
            best_h = h_curr.copy()
        
        # Gradient: we want to decrease the peak of autocorrelation
        # The peak is at index j*. To reduce it, we should decrease power at
        # frequencies that contribute constructively to that peak.
        # Use the phase information: contribution to autocorr[j] from freq k is
        # |F[k]|² * exp(2i*k*j*Δω)
        
        j_star = np.argmax(autocorr)
        
        # Compute gradient of max_val w.r.t. log_power
        # autocorr[j] = sum_k |F[k]|² * exp(2πi*k*(j-N+1)/NFFT) * Δω_factor
        # d(autocorr[j])/d(log_power[k]) = |F[k]|² * exp(2πi*k*(j_star-N+1)/NFFT)
        
        k_indices = np.arange(len(power))
        # The phase factor for each frequency's contribution to autocorr[j_star]
        # autocorr[j] = sum_k F[k]*conj(F[k-j]) ... this is more complex
        # Simpler: use finite difference on a few key frequencies
        
        # Alternative: use the fact that for the peak to decrease, we want
        # to reduce coherence. Add noise to the spectrum.
        
        # Spectral shaping: apply a smooth filter that reduces high-frequency
        # components (which cause sharp peaks in time domain)
        # But we also need to maintain the DC component (total integral)
        
        # Use a simple approach: multiply power by a smooth function that
        # suppresses frequencies contributing to the peak
        # The contribution of freq k to autocorr[j] has phase exp(2πi*k*(j-N+1)/N)
        # We want to reduce power where this phase aligns with the peak direction
        
        # Simpler spectral approach: optimize a parametric family
        # power[k] = exp(-a * k² / N²) * base_power[k]
        # where a controls the bandwidth
        
        if it % 10 == 0:
            # Try adjusting the spectral bandwidth
            for a in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]:
                k_norm = k_indices / N
                spectral_filter = np.exp(-a * k_norm**2)
                # Keep DC component fixed
                spectral_filter[0] = 1.0
                
                power_test = power * spectral_filter
                F_test = np.sqrt(np.maximum(power_test, 0)) * np.exp(1j * phase)
                h_test = np.fft.irfft(F_test, _NFFT)[:N]
                h_test = np.maximum(h_test, 0)
                s_test = np.sum(h_test) * DELTA
                if s_test < 1e-15:
                    continue
                h_test = h_test / s_test
                
                r_test = compute_ratio_fast(h_test)
                if r_test < best_ratio - 1e-12:
                    best_ratio = r_test
                    best_h = h_test.copy()
                    power = power_test.copy()
        
        # Also try random spectral perturbations
        if it % 5 == 0:
            for _ in range(3):
                noise = np.random.randn(len(power)) * 0.1
                noise[0] = 0  # keep DC
                power_test = power * np.exp(noise)
                F_test = np.sqrt(np.maximum(power_test, 0)) * np.exp(1j * phase)
                h_test = np.fft.irfft(F_test, _NFFT)[:N]
                h_test = np.maximum(h_test, 0)
                s_test = np.sum(h_test) * DELTA
                if s_test < 1e-15:
                    continue
                h_test = h_test / s_test
                
                r_test = compute_ratio_fast(h_test)
                if r_test < best_ratio - 1e-12:
                    best_ratio = r_test
                    best_h = h_test.copy()
                    power = power_test.copy()
    
    return best_h, best_ratio

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    x = np.linspace(-0.25, 0.25, N+1)[:-1] + DELTA/2  # centers of intervals
    
    candidates = []
    
    # Uniform
    candidates.append(np.ones(N))
    
    # Inverted V: concentrated at center
    candidates.append(0.25 - np.abs(x))
    
    # Cosine modulations
    for a in [0.3, 0.5, 0.7, 0.9, 1.0]:
        for k in [1, 2, 4, 8]:
            candidates.append(1.0 - a * np.cos(2 * np.pi * k * x))
    
    # Power shapes concentrated at center
    for p in [2, 4, 6, 8, 12]:
        candidates.append((0.25 - np.abs(x)) ** p)
    
    # Gaussian centered at 0
    for sigma in [0.02, 0.03, 0.05, 0.07, 0.1]:
        candidates.append(np.exp(-x**2 / (2*sigma**2)))
    
    # Two-bump
    for sigma in [0.02, 0.03, 0.05, 0.08, 0.1]:
        candidates.append(np.exp(-((x - 0.125)**2) / (2*sigma**2)) + np.exp(-((x + 0.125)**2) / (2*sigma**2)))
    
    # Spectral-based candidates: design in frequency domain
    # Flat spectrum with random phases → uniform-like
    # Smooth spectrum → smooth time function
    
    # Gaussian in frequency domain → sinc-like in time
    for bw in [0.5, 1.0, 2.0, 3.0, 5.0]:
        k_freq = np.arange(N)
        spec = np.exp(-k_freq**2 / (2 * bw**2 * N))
        # Make it symmetric for real signal
        h_spec = np.fft.irfft(spec, _NFFT)[:N]
        h_spec = np.maximum(h_spec, 0)
        if np.sum(h_spec) > 1e-10:
            candidates.append(h_spec)
    
    # Cosine in frequency → raised cosine in time
    for bw in [0.5, 1.0, 2.0, 3.0]:
        k_freq = np.arange(N)
        spec = np.cos(np.pi * k_freq / (2 * bw * N))
        spec = np.maximum(spec, 0)
        h_spec = np.fft.irfft(spec, _NFFT)[:N]
        h_spec = np.maximum(h_spec, 0)
        if np.sum(h_spec) > 1e-10:
            candidates.append(h_spec)
    
    # Evaluate all candidates
    best_ratio = 1e15
    best_h = None
    for h in candidates:
        h = np.maximum(h, 0)
        if np.sum(h) < 1e-10:
            continue
        r = compute_ratio_fast(h)
        if r < best_ratio:
            best_ratio = r
            best_h = h.copy()
    
    # Spectral optimization from best candidate
    if best_h is not None:
        h_opt, r_opt = spectral_optimize(best_h, max_iters=300, deadline=deadline - 5)
        if r_opt < best_ratio:
            best_ratio = r_opt
            best_h = h_opt
    
    # Additional local refinement: small random perturbations
    if best_h is not None:
        for _ in range(50):
            if time.time() > deadline - 3:
                break
            h_test = best_h.copy()
            idx = np.random.choice(N, size=20, replace=False)
            h_test[idx] *= np.random.uniform(0.7, 1.3, size=20)
            h_test = np.maximum(h_test, 0)
            s_test = np.sum(h_test) * DELTA
            if s_test < 1e-15:
                continue
            h_test = h_test / s_test
            r_test = compute_ratio_fast(h_test)
            if r_test < best_ratio - 1e-12:
                best_ratio = r_test
                best_h = h_test
    
    # Final
    h_final = np.maximum(best_h, 0.0)
    s_final = np.sum(h_final) * DELTA
    if s_final > 1e-15:
        h_final = h_final / s_final
    
    return [float(x) for x in h_final]
