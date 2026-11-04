import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Initialize with a random sequence
    # Using a chirp or m-sequence might be better, but random with optimization works
    x = np.random.randn(N)
    # Ensure sum is positive
    if np.sum(x) < 0:
        x = -x
    x /= np.sum(x)
    
    best_x = x.copy()
    best_score = np.inf
    
    def calc_score(x):
        # Calculate max absolute autoconvolution for k != 0
        # Use FFT for speed
        size = 2 * N
        X = np.fft.rfft(x, n=size)
        A = np.fft.irfft(X * X, n=size)
        # A[1:N] are the sidelobes
        return np.max(np.abs(A[1:N]))
    
    current_score = calc_score(x)
    if current_score < best_score:
        best_score = current_score
        best_x = x.copy()
        
    eta = 0.05
    current_score = best_score
    
    while time.time() < deadline:
        # Calculate gradient of sum of squared sidelobes
        A_full = np.convolve(x, x, 'full')
        A_side = A_full[1:N]
        
        # G1[j] = sum_{k=1}^{N-1} A(k) x[j-k]
        G1 = np.convolve(A_side, x, 'full')[:N]
        
        # G2[j] = sum_{k=1}^{N-1} A(k) x[j+k]
        G2 = np.zeros(N)
        for k in range(1, N):
            G2[:N-k] += A_side[k-1] * x[k:]
            
        grad = 2 * (G1 + G2)
        
        # Update
        x_new = x - eta * grad
        sum_x = np.sum(x_new)
        if sum_x < 0:
            x_new = -x_new
            sum_x = -sum_x
            
        if sum_x > 1e-6:
            x_new /= sum_x
            new_score = calc_score(x_new)
            
            if new_score < best_score:
                best_score = new_score
                best_x = x_new.copy()
                x = x_new
                current_score = new_score
                # Increase eta slightly if improving
                eta = min(eta * 1.01, 0.1)
            else:
                # Decrease eta if not improving
                eta = max(eta * 0.9, 0.001)
        else:
            # If sum is too small, restart or perturb
            x_new = x + 0.01 * np.random.randn(N)
            if np.sum(x_new) < 0:
                x_new = -x_new
            x_new /= np.sum(x_new)
            x = x_new
            eta = 0.01

    return best_x
