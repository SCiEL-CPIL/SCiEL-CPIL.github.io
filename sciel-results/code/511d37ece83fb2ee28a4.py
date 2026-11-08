"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_autocorr_max(x):
    """Compute max |R(k)| for k=1..N-1 for a ±1 sequence x of length N."""
    N = len(x)
    # Use FFT-based autocorrelation
    # Zero-pad to avoid circular convolution
    size = 2 * N
    x_pad = np.zeros(size)
    x_pad[:N] = x
    X = np.fft.fft(x_pad)
    autocorr = np.fft.ifft(X * np.conj(X)).real[:N]
    # autocorr[k] = sum_{i=0}^{N-1-k} x[i]*x[i+k]
    return np.max(np.abs(autocorr[1:]))

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin before 120s limit
    
    # Initialize with random ±1 sequence
    rng = np.random.RandomState(42)
    x = rng.choice([-1, 1], size=N)
    best_x = x.copy()
    best_score = _compute_autocorr_max(x)
    
    # Simulated annealing
    T = 1.0
    T_min = 0.01
    cooling = 0.9995
    iterations = 0
    
    while time.time() < deadline:
        iterations += 1
        
        # Pick a random position to flip
        idx = rng.randint(N)
        x[idx] *= -1
        
        new_score = _compute_autocorr_max(x)
        delta = new_score - best_score
        
        if delta < 0 or rng.random() < np.exp(-delta / T):
            best_x = x.copy()
            best_score = new_score
        
        T = max(T * cooling, T_min)
        
        # If T is very small, do more aggressive local search
        if T <= T_min * 1.01:
            # Try flipping pairs
            if time.time() > deadline - 5.0:
                break
            i = rng.randint(N)
            j = rng.randint(N)
            if i == j:
                continue
            x[i] *= -1
            x[j] *= -1
            new_score = _compute_autocorr_max(x)
            if new_score < best_score:
                best_x = x.copy()
                best_score = new_score
            else:
                x[i] *= -1
                x[j] *= -1
    
    return best_x
# EVOLVE-BLOCK-END
