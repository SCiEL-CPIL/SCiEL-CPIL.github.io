import numpy as np
import time

N = 400

def construct():
    # Deadline guard: 120s limit, use 110s for safety
    deadline = time.time() + 110
    
    # Initialize with random binary sequence
    x = np.random.choice([-1.0, 1.0], size=N)
    
    # Compute initial autocorrelation R(k) for k=1..N-1
    # R(k) = sum_{i=0}^{N-1-k} x[i] * x[i+k]
    R = np.zeros(N-1)
    for k in range(1, N):
        R[k-1] = np.dot(x[:N-k], x[k:])
        
    current_peak = np.max(np.abs(R))
    best_peak = current_peak
    best_x = x.copy()
    
    # Simulated Annealing
    T = 10.0  # Initial temperature
    cooling = 0.9995  # Slow cooling
    
    # Precompute indices for faster delta calculation?
    # The vectorized delta calculation is fast enough.
    
    while time.time() < deadline - 10:  # Reserve 10s for polish
        j = np.random.randint(N)
        
        # Calculate delta_R for flipping x[j]
        delta_R = np.zeros(N-1)
        
        # Contribution from x[j-k] for k=1..j
        if j > 0:
            # x[j-1::-1] is x[j-1], x[j-2], ..., x[0]
            # These correspond to lags k=1, 2, ..., j
            delta_R[:j] -= 2 * x[j] * x[j-1::-1]
            
        # Contribution from x[j+k] for k=1..N-1-j
        if j < N-1:
            # x[j+1:] is x[j+1], x[j+2], ..., x[N-1]
            # These correspond to lags k=1, 2, ..., N-1-j
            delta_R[:N-1-j] -= 2 * x[j] * x[j+1:]
            
        new_R = R + delta_R
        new_peak = np.max(np.abs(new_R))
        
        # Acceptance criterion
        if new_peak < current_peak:
            x[j] *= -1
            R += delta_R
            current_peak = new_peak
            if current_peak < best_peak:
                best_peak = current_peak
                best_x = x.copy()
        else:
            # Metropolis
            if T > 1e-6:
                prob = np.exp(-(new_peak - current_peak) / T)
                if np.random.rand() < prob:
                    x[j] *= -1
                    R += delta_R
                    current_peak = new_peak
            
        T *= cooling
        
    # Polish Phase: Steepest Descent
    # Try to find a flip that reduces the peak
    improved = True
    while improved and time.time() < deadline - 1:
        improved = False
        best_delta_peak = 0
        best_j = -1
        best_new_R = None
        
        for j in range(N):
            # Calculate delta_R for flipping x[j]
            delta_R = np.zeros(N-1)
            
            if j > 0:
                delta_R[:j] -= 2 * x[j] * x[j-1::-1]
            if j < N-1:
                delta_R[:N-1-j] -= 2 * x[j] * x[j+1:]
                
            new_R = R + delta_R
            new_peak = np.max(np.abs(new_R))
            
            # We want to minimize peak, so we look for the smallest new_peak
            # Or largest reduction
            if new_peak < current_peak:
                if new_peak < best_delta_peak or best_j == -1:
                    best_delta_peak = new_peak
                    best_j = j
                    best_new_R = new_R
                    best_delta_R = delta_R
        
        if best_j != -1:
            x[best_j] *= -1
            R = best_new_R
            current_peak = best_delta_peak
            improved = True
            if current_peak < best_peak:
                best_peak = current_peak
                best_x = x.copy()

    return best_x
