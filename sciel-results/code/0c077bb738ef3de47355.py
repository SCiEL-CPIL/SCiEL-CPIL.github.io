import numpy as np
import time

N = 400

def _compute_peak_sidelobe(x):
    """
    Computes the peak sidelobe of the autocorrelation of x.
    x is a numpy array of length N with values in {-1, 1}.
    Returns the maximum absolute value of the autocorrelation for lags 1 to N-1.
    """
    # Pad to 2N to avoid circular convolution effects
    n_fft = 2 * N
    X = np.fft.fft(x, n=n_fft)
    R = np.fft.ifft(X * np.conj(X)).real
    # R[0] is the main lobe (sum of squares = N)
    # R[1:N] are the lags 1 to N-1
    # R[N:2N] are the negative lags, which are symmetric
    sidelobes = np.abs(R[1:N])
    return np.max(sidelobes)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second margin from 120s limit
    
    best_x = np.ones(N)
    best_val = _compute_peak_sidelobe(best_x)
    
    # Try a few random restarts
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Random initialization
        x = np.random.choice([-1, 1], size=N)
        current_val = _compute_peak_sidelobe(x)
        
        if current_val < best_val:
            best_x = x.copy()
            best_val = current_val
            
        # Local search
        improved = True
        while improved and time.time() < deadline:
            improved = False
            current_val = _compute_peak_sidelobe(x)
            
            # Find the best flip
            best_flip_idx = -1
            best_flip_val = current_val
            
            # We can optimize by only checking flips that might help, 
            # but with N=400, checking all 400 is fast.
            for i in range(N):
                # Flip x[i]
                x[i] *= -1
                new_val = _compute_peak_sidelobe(x)
                
                if new_val < best_flip_val:
                    best_flip_val = new_val
                    best_flip_idx = i
                # Unflip for next iteration
                x[i] *= -1
                
            if best_flip_idx != -1:
                # Apply the best flip
                x[best_flip_idx] *= -1
                current_val = best_flip_val
                improved = True
                
                if current_val < best_val:
                    best_val = current_val
                    best_x = x.copy()
                    
        # If we hit a local minimum, try a few random perturbations to escape?
        # Or just move to next restart.
        
    return best_x
