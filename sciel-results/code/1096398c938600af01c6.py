"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Initial guess: constant vector
    h_best = np.ones(N)
    val_best = 2.0
    
    # Parameterize h using Fourier coefficients to ensure smoothness and structure
    # h[i] = sum_{k=0}^{M} (a_k cos(2pi k i/N) + b_k sin(2pi k i/N))
    # We optimize the coefficients.
    # To keep sum(h) non-zero and stable, we fix the DC term a_0 to be 1.0 
    # and optimize the AC terms.
    
    M = 20 # Number of harmonics
    # Params: a_1..a_M, b_1..b_M
    # Total params: 2*M
    x0 = np.zeros(2*M)
    
    # Generate grid
    i_grid = np.arange(N)
    angles = 2 * np.pi * np.outer(i_grid, np.arange(1, M+1)) / N
    cos_basis = np.cos(angles) # N x M
    sin_basis = np.sin(angles) # N x M
    
    def objective(params):
        a = params[:M]
        b = params[M:]
        h = np.ones(N) + cos_basis.dot(a) + sin_basis.dot(b)
        
        # Check sum
        s = np.sum(h)
        if np.abs(s) < 1e-6:
            return 1e10
        if not np.all(np.isfinite(h)):
            return 1e10
            
        # Compute autoconvolution
        # Using FFT for speed
        H = np.fft.fft(h)
        C = np.fft.ifft(H * H)
        c = C.real
        # Full convolve length is 2N-1. 
        # np.convolve(h,h) gives length 2N-1.
        # FFT method:
        # We need the full convolution.
        # Zero pad to 2N
        H_pad = np.fft.fft(h, n=2*N)
        C_pad = np.fft.ifft(H_pad * H_pad).real
        # The first N elements of C_pad correspond to lags 0..N-1?
        # Actually, linear convolution via FFT:
        # c[k] for k=0..2N-2
        # C_pad[0..N-1] are lags 0..N-1?
        # Let's verify.
        # h = [1, 0]. H=[1,0]. H^2=[1,0]. ifft=[1,0].
        # conv([1,0],[1,0]) = [1,0,0].
        # C_pad = [1, 0, 0, 0] (size 4).
        # We need max over all signed entries.
        max_c = np.max(C_pad)
        
        val = abs(2 * N * max_c / (s ** 2))
        return val

    # Optimize
    try:
        # Use L-BFGS-B
        res = minimize(objective, x0, method='L-BFGS-B', 
                       options={'maxiter': 100, 'maxfun': 1000})
        
        if res.success or res.fun < val_best:
            a = res.x[:M]
            b = res.x[M:]
            h_candidate = np.ones(N) + cos_basis.dot(a) + sin_basis.dot(b)
            s = np.sum(h_candidate)
            if np.abs(s) > 1e-6:
                H_pad = np.fft.fft(h_candidate, n=2*N)
                C_pad = np.fft.ifft(H_pad * H_pad).real
                max_c = np.max(C_pad)
                val_candidate = abs(2 * N * max_c / (s ** 2))
                if val_candidate < val_best:
                    h_best = h_candidate
                    val_best = val_candidate
    except Exception:
        pass

    # Check deadline
    if time.time() > deadline:
        return h_best

    # Try a few random restarts if time permits
    for _ in range(5):
        if time.time() > deadline:
            break
        x0 = np.random.normal(0, 0.1, 2*M)
        try:
            res = minimize(objective, x0, method='L-BFGS-B', 
                           options={'maxiter': 50, 'maxfun': 500})
            if res.fun < val_best:
                a = res.x[:M]
                b = res.x[M:]
                h_candidate = np.ones(N) + cos_basis.dot(a) + sin_basis.dot(b)
                s = np.sum(h_candidate)
                if np.abs(s) > 1e-6:
                    H_pad = np.fft.fft(h_candidate, n=2*N)
                    C_pad = np.fft.ifft(H_pad * H_pad).real
                    max_c = np.max(C_pad)
                    val_candidate = abs(2 * N * max_c / (s ** 2))
                    if val_candidate < val_best:
                        h_best = h_candidate
                        val_best = val_candidate
        except Exception:
            pass

    return h_best
# EVOLVE-BLOCK-END
