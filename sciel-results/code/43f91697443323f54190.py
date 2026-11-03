"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Spectral design: represent h as 0.5 + x where x is zero-mean.
    Use iterative phase optimization to minimize peak sidelobe of autocorrelation.
    """
    np.random.seed(42)
    
    # We want to minimize max_k C_k where C_k = N/2 - autocorr_h(k)
    # autocorr_h(k) = sum_i h_i * h_{i+k mod N}
    # For h = 0.5 + x, autocorr_h(k) = N/4 + autocorr_x(k)
    # So C_k = N/4 - autocorr_x(k)
    # Minimizing max C_k is equivalent to maximizing min autocorr_x(k) for k≠0
    
    best_h = np.full(N, 0.5)
    best_score = np.inf
    
    # Try multiple initializations with different amplitude profiles
    for trial in range(30):
        # Choose amplitude profile
        # Option 1: Flat spectrum
        # Option 2: Tapered spectrum to reduce sidelobes
        if trial < 10:
            # Flat
            amps = np.ones(N)
        elif trial < 20:
            # Hanning taper
            amps = 0.5 * (1 + np.cos(2 * np.pi * np.arange(N) / N))
        else:
            # Blackman taper
            n = np.arange(N)
            amps = 0.42 + 0.5 * np.cos(2 * np.pi * n / N) + 0.08 * np.cos(4 * np.pi * n / N)
        
        amps[0] = 0  # Zero mean
        
        # Random initial phases
        phases = np.random.uniform(0, 2 * np.pi, N)
        
        # Iterative phase optimization (Weber-like algorithm)
        for it in range(200):
            # Compute current time-domain signal
            X = amps * np.exp(1j * phases)
            x = np.fft.ifft(X).real
            
            # Compute autocorrelation of x
            X_full = np.fft.fft(x)
            autocorr_x = np.fft.ifft(np.abs(X_full)**2).real
            
            # Compute C_k = N/4 - autocorr_x(k)
            C_vals = N/4 - autocorr_x
            
            # Find the lag with maximum C_k (worst sidelobe)
            k_max = np.argmax(C_vals)
            
            if k_max == 0:
                k_max = 1  # Skip DC component
            
            # Update phases to reduce the peak at k_max
            # The gradient of C_k_max w.r.t. phase_j is related to 
            # the cross-correlation between x and shifted x by k_max
            
            # Compute gradient: d(C_k)/d(phase_j) = -2 * Re[x * conj(shifted_x) * i * exp(1j*phase_j) * conj(X_j)]
            # Simplified: adjust phases to reduce autocorrelation at k_max
            
            # Shift x by k_max
            x_shifted = np.roll(x, k_max)
            
            # Compute complex gradient
            # The contribution of each frequency bin j to the autocorrelation at lag k
            # is |X_j|^2 * exp(-2j*pi*j*k/N)
            # To reduce C_k, we want to make autocorr_x(k) larger
            # which means making Re[sum_j |X_j|^2 exp(-2j*pi*j*k/N)] larger
            
            # Phase update rule: shift phase_j by -arg(exp(-2j*pi*j*k/N)) = 2*pi*j*k/N
            # But this is just a rotation, so we need a more careful approach
            
            # Use the fact that changing phase_j by delta changes autocorr_x(k) by
            # approximately 2*|X_j|^2 * sin(delta) * sin(2*pi*j*k/N)
            # To increase autocorr_x(k), we want sin(delta)*sin(2*pi*j*k/N) > 0
            # So delta should have the same sign as sin(2*pi*j*k/N)
            
            # Simple approach: add a small phase shift proportional to sin(2*pi*j*k/N)
            phase_shift = 0.1 * np.sin(2 * np.pi * np.arange(N) * k_max / N)
            phases = (phases + phase_shift) % (2 * np.pi)
            
            # Also try the opposite direction sometimes
            if it % 2 == 1:
                phases = (phases - 2 * phase_shift) % (2 * np.pi)
        
        # After optimization, reconstruct x
        X = amps * np.exp(1j * phases)
        x = np.fft.ifft(X).real
        
        # Scale x to fit in [-0.5, 0.5]
        max_abs = np.max(np.abs(x))
        if max_abs > 1e-10:
            # Scale so that max |x| is slightly less than 0.5 to allow for clipping
            scale = 0.45 / max_abs
            x = x * scale
        
        # Adjust to ensure sum(x) = 0
        x = x - np.mean(x)
        
        # Clip to [-0.5, 0.5]
        x = np.clip(x, -0.5, 0.5)
        
        # Re-adjust sum to exactly 0
        x = x - np.mean(x)
        
        # Final clip
        x = np.clip(x, -0.5, 0.5)
        
        # Correct sum exactly
        sum_error = np.sum(x)
        if abs(sum_error) > 1e-10:
            x = x - sum_error / N
            x = np.clip(x, -0.5, 0.5)
        
        h = 0.5 + x
        
        # Verify constraints
        h = np.clip(h, 0, 1)
        
        # Compute score: max_k C_k
        H = np.fft.fft(h)
        autocorr = np.fft.ifft(np.abs(H)**2).real
        
        C_vals = N/2 - autocorr
        score = np.max(C_vals)
        
        if score < best_score:
            best_score = score
            best_h = h.copy()
    
    # Final correction to ensure sum is exactly N/2
    sum_h = np.sum(best_h)
    if abs(sum_h - N/2) > 1e-10:
        best_h = best_h - (sum_h - N/2) / N
        best_h = np.clip(best_h, 0, 1)
    
    return best_h
# EVOLVE-BLOCK-END
