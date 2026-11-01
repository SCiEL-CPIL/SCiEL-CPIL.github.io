import numpy as np
import time

N = 400

def compute_metric(h):
    # h is array of +/- 1
    # c0 = sum(h^2) = N
    # We want to minimize max_{k != 0} |c_k|
    # Use FFT for speed
    n_fft = 2 * N
    H = np.fft.fft(h, n=n_fft)
    C = np.fft.ifft(H * np.conj(H)).real
    # C[0] is c_0 = N
    # C[1:N] are c_1 ... c_{N-1}
    # C[N+1:2N] are negative lags (symmetric for real)
    # We check lags 1 to N-1
    max_side = np.max(np.abs(C[1:N]))
    return max_side / N

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s guard
    
    # Initialize with a Barker-like or random sequence
    # Barker sequences of length 13, 31 exist, but N=400.
    # Random is a good start.
    best_h = np.random.choice([-1, 1], size=N)
    best_val = compute_metric(best_h)
    
    # Local Search
    # Flip one bit at a time
    # To speed up, we can precompute contributions or just do full FFT per step if N is small enough
    # N=400, FFT is fast. 100s allows many iterations.
    
    # Optimization: 
    # Flipping h[i] changes c_k by -2 * h[i] * sum_{j} h[j] h[j+k+i] ... complex.
    # Just recompute FFT.
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try flipping each bit
        for i in range(N):
            if time.time() > deadline:
                break
            
            old_val = best_h[i]
            best_h[i] = -old_val
            new_val = compute_metric(best_h)
            
            if new_val < best_val:
                best_val = new_val
                improved = True
            else:
                best_h[i] = old_val
        
        # If no improvement in a full pass, try random restarts or perturbations
        if not improved:
            # Perturb a few random bits to escape local minima
            if time.time() < deadline - 5:
                indices = np.random.choice(N, size=5, replace=False)
                best_h[indices] *= -1
                best_val = compute_metric(best_h)
                improved = True # Force another pass

    return best_h
