import numpy as np

N = 400

def construct():
    # Initialize with a sparse random vector
    np.random.seed(42)
    x = np.random.randn(N)
    # Enforce initial sparsity
    threshold = 0.8
    x[np.abs(x) < threshold] = 0
    norm = np.linalg.norm(x)
    if norm < 1e-9:
        x = np.ones(N)
        norm = np.linalg.norm(x)
    x = x / norm
    
    M = 2 * N
    
    best_x = x.copy()
    best_psl = np.inf
    
    # ADMM-like parameters
    step_size = 0.05
    l1_lambda = 0.005
    
    for iteration in range(10000):
        # Zero-pad
        x_pad = np.zeros(M)
        x_pad[:N] = x
        # DFT
        X = np.fft.fft(x_pad)
        # Power spectrum
        P = np.abs(X)**2
        # Linear autocorrelation
        R = np.fft.ifft(P).real
        
        # Extract linear part
        R_lag0 = R[0]
        if R_lag0 < 1e-9:
            break
        R_sidelobes = np.concatenate([R[1:N], R[M-N+1:M]])
        
        # Find peak sidelobe
        idx = np.argmax(np.abs(R_sidelobes))
        
        # Determine which lag this corresponds to
        if idx < N - 1:
            lag = idx + 1
        else:
            lag = idx - 2 * N + 2
            
        # Compute gradient of R(lag) with respect to x
        g = np.zeros(N)
        if lag > 0:
            g[:N-lag] += x[lag:]
            g[lag:] += x[:N-lag]
        elif lag < 0:
            lag_abs = -lag
            g[lag_abs:] += x[:N-lag_abs]
            g[:N-lag_abs] += x[lag_abs:]
        else:
            g = 2 * x
            
        # Update x
        current_val = R_sidelobes[idx]
        x = x - step_size * np.sign(current_val) * g
        
        # Apply L1 soft-thresholding
        x = np.sign(x) * np.maximum(np.abs(x) - l1_lambda, 0)
        
        # Normalize
        norm = np.linalg.norm(x)
        if norm < 1e-9:
            x = np.random.randn(N)
            x[np.abs(x) < 0.8] = 0
            norm = np.linalg.norm(x)
            if norm < 1e-9:
                x = np.ones(N)
                norm = np.linalg.norm(x)
        x = x / norm
        
        # Update best
        psl = np.max(np.abs(R_sidelobes)) / R_lag0
        if psl < best_psl:
            best_psl = psl
            best_x = x.copy()
            
    return best_x
