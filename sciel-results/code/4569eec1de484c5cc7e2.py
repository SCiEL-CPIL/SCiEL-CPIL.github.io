import numpy as np

N = 400

def construct():
    # Initialize with random real vector
    np.random.seed(42)
    x = np.random.randn(N)
    
    # Normalize
    x = x / np.linalg.norm(x)
    
    M = 2 * N  # Zero-padding for linear autocorrelation
    
    best_x = x.copy()
    best_score = np.inf
    
    # ADMM-like parameters
    mu = 1.0          # Quadratic penalty parameter
    t = 1.0           # Auxiliary variable for max sidelobe
    
    # Spectral flattening exponent
    # P_new = P^alpha * P_mean^(1-alpha)
    # alpha=0.5 is geometric mean
    alpha = 0.5
    
    for iteration in range(1000):
        # 1. Compute Autocorrelation
        x_pad = np.zeros(M)
        x_pad[:N] = x
        X = np.fft.fft(x_pad)
        P = np.abs(X)**2
        R = np.fft.ifft(P).real
        
        # Extract linear autocorrelation
        R_lag0 = R[0]
        R_sidelobes = np.concatenate([R[1:N], R[M-N+1:M]])
        
        # Calculate current score (max abs sidelobe / sum^2)
        sum_x = np.sum(x)
        if abs(sum_x) < 1e-9:
            sum_x = 1e-9
        score = np.max(np.abs(R_sidelobes)) / (sum_x**2)
        
        if score < best_score:
            best_score = score
            best_x = x.copy()
        
        # 2. Update x via spectral flattening
        P_mean = np.mean(P)
        if P_mean == 0:
            break
        P_new = (P ** alpha) * (P_mean ** (1 - alpha))
        
        # Recover X
        # Keep the phase of X
        phase = np.angle(X)
        X_new = np.sqrt(P_new) * np.exp(1j * phase)
        x_new = np.fft.ifft(X_new).real[:N]
        
        # Normalize
        norm = np.linalg.norm(x_new)
        if norm == 0:
            break
        x_new = x_new / norm
        
        # Optional: Add a small perturbation to escape local minima
        # x_new += 0.01 * np.random.randn(N)
        # x_new = x_new / np.linalg.norm(x_new)
        
        x = x_new
        
    return best_x
