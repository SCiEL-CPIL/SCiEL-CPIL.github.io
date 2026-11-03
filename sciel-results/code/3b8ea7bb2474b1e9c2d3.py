"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np


def construct():
    """
    Spectral design: represent h as 0.5 + x where x is zero-mean.
    Use direct gradient-based phase optimization to minimize peak sidelobe.
    """
    np.random.seed(42)
    
    best_h = np.full(N, 0.5)
    best_score = np.inf
    
    # Try multiple initializations
    for trial in range(50):
        # Random initial phases
        phases = np.random.uniform(0, 2 * np.pi, N)
        
        # Iterative phase optimization with proper gradient
        for it in range(500):
            # Compute current time-domain signal
            X = np.exp(1j * phases)
            x = np.fft.ifft(X).real
            
            # Ensure zero mean
            x = x - np.mean(x)
            
            # Compute autocorrelation of x
            X_full = np.fft.fft(x)
            autocorr_x = np.fft.ifft(np.abs(X_full)**2).real
            
            # Compute C_k = N/4 - autocorr_x(k) for k != 0
            C_vals = N/4 - autocorr_x
            C_vals[0] = -np.inf  # Ignore DC
            
            # Find the lag with maximum C_k
            k_max = np.argmax(C_vals)
            
            if k_max == 0:
                k_max = 1
            
            # Compute gradient of C_k_max w.r.t. each phase
            # C_k = N/4 - Re[sum_j |X_j|^2 * exp(-2j*pi*j*k/N)]
            # d(C_k)/d(phase_j) = -2 * |X_j|^2 * Re[i * exp(-2j*pi*j*k/N)]
            #                    = 2 * |X_j|^2 * Im[exp(-2j*pi*j*k/N)]
            #                    = -2 * |X_j|^2 * sin(2*pi*j*k/N)
            
            j_idx = np.arange(N)
            gradient = -2 * np.ones(N) * np.sin(2 * np.pi * j_idx * k_max / N)
            
            # Gradient ascent on autocorr_x(k_max) = gradient descent on C_k_max
            # Update phases: phase_j += step * gradient_j
            step = 0.01
            phases = (phases + step * gradient) % (2 * np.pi)
            
            # Adaptive step size: reduce if score increases
            if it > 0 and it % 10 == 0:
                # Recompute to check improvement
                X_check = np.exp(1j * phases)
                x_check = np.fft.ifft(X_check).real
                x_check = x_check - np.mean(x_check)
                X_check_full = np.fft.fft(x_check)
                autocorr_check = np.fft.ifft(np.abs(X_check_full)**2).real
                C_check = N/4 - autocorr_check
                C_check[0] = -np.inf
                if np.max(C_check) > C_vals[k_max] + 0.01:
                    step *= 0.5
        
        # After optimization, reconstruct x
        X = np.exp(1j * phases)
        x = np.fft.ifft(X).real
        
        # Scale x to fit in [-0.5, 0.5]
        max_abs = np.max(np.abs(x))
        if max_abs > 1e-10:
            scale = 0.48 / max_abs
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
        C_vals[0] = -np.inf  # Ignore DC
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
