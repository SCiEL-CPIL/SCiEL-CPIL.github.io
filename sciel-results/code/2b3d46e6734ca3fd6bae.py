import numpy as np
import time
from scipy.optimize import minimize

N = 600

def compute_score(h):
    """Compute the normalized maximum of autoconvolution."""
    h = np.maximum(h, 0)
    area = np.sum(h)
    max_val = np.max(h)
    if area == 0 or max_val == 0:
        return float('inf')
    conv = np.convolve(h, h, mode='full')
    return np.max(conv) / (area * max_val)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill
    
    # Fix a random noise seed for reproducibility
    np.random.seed(42)
    
    # Generate a fixed Gaussian noise sequence
    # We will filter this and square it
    noise = np.random.randn(N)
    
    # Precompute FFT of noise for fast filtering
    noise_fft = np.fft.fft(noise)
    
    # Define the score function in terms of bandwidth parameter w (log-scale for optimization)
    # w is the log of the Gaussian filter sigma in frequency domain
    # h = (ifft(fft(noise) * exp(-w^2 * freq^2 / 2)))^2
    # Actually, let's use a Gaussian kernel in time or frequency.
    # Let's use a Gaussian low-pass filter in frequency domain.
    # freqs = np.fft.fftfreq(N)
    # kernel = np.exp(-w**2 * freqs**2 / 2)
    # h_raw = np.fft.ifft(noise_fft * kernel).real
    # h = h_raw**2
    
    # To ensure h is non-negative and we can control the scale, we can also add a small offset or just rely on squaring.
    # Squaring ensures non-negativity.
    # The score is scale-invariant, so we don't need to optimize the scale of h.
    # However, the "max_val" in the denominator changes with the shape.
    
    # Let's define the objective function for the optimizer
    def objective(w_log):
        w = np.exp(w_log[0])
        # w is the bandwidth parameter. Larger w -> wider filter -> smoother h.
        # Frequency vector
        freqs = np.fft.fftfreq(N)
        # Gaussian low-pass filter
        # sigma_freq = 1/w ? Or w is sigma?
        # Let's say w is the standard deviation of the Gaussian in frequency space.
        # Smaller w -> sharper cutoff? No, exp(-f^2 / 2w^2).
        # If w is small, it decays fast -> low pass.
        # If w is large, it decays slow -> high pass / white noise.
        # We want to explore different smoothness levels.
        # Let's define kernel = exp(-freqs^2 / (2 * w^2))
        kernel = np.exp(-freqs**2 / (2 * w**2))
        
        # Filter noise
        filtered_fft = noise_fft * kernel
        h_raw = np.fft.ifft(filtered_fft).real
        
        # Square to get non-negative step function
        h = h_raw**2
        
        # Ensure no numerical negatives (though squaring should handle it)
        h = np.maximum(h, 0)
        
        # Compute score
        area = np.sum(h)
        max_val = np.max(h)
        if area == 0 or max_val == 0:
            return 1e10
            
        # Use FFT for convolution for speed
        conv_fft = np.fft.fft(h, n=2*N)
        conv = np.fft.ifft(conv_fft * conv_fft).real
        max_conv = np.max(conv)
        
        score = max_conv / (area * max_val)
        return score

    # Optimization
    # Start with a few different initial bandwidths
    best_score = float('inf')
    best_h = None
    
    # Initial guesses for w (bandwidth). 
    # w in frequency domain.
    # If w is very small, h is very smooth (almost constant).
    # If w is large, h is noisy (close to noise^2).
    # Let's sample log(w) from -2 to 2.
    
    initial_logs = np.linspace(-1.5, 1.5, 10)
    
    for log_w_init in initial_logs:
        if time.time() > deadline:
            break
            
        res = minimize(objective, [log_w_init], method='Nelder-Mead', 
                       options={'xatol': 1e-4, 'fatol': 1e-5, 'maxiter': 500})
        
        if time.time() > deadline:
            break
            
        if res.fun < best_score:
            best_score = res.fun
            # Reconstruct best_h
            w = np.exp(res.x[0])
            freqs = np.fft.fftfreq(N)
            kernel = np.exp(-freqs**2 / (2 * w**2))
            filtered_fft = noise_fft * kernel
            h_raw = np.fft.ifft(filtered_fft).real
            best_h = h_raw**2

    # If the optimization didn't find anything better or failed, fall back to the analytical solution
    # Analytical solution: one spike, rest flat
    M = 1.0 / np.sqrt(N)
    h_analytical = np.full(N, (1.0 - M) / (N - 1))
    h_analytical[0] = M
    score_analytical = compute_score(h_analytical)
    
    if best_h is None or best_score > score_analytical:
        return list(h_analytical)
    
    return list(best_h)
