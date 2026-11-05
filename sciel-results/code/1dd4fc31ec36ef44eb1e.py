import numpy as np
import scipy.optimize as so
import time

N = 400

def _get_h_from_H(H):
    """Reconstruct h from DFT coefficients H."""
    # H is complex vector of length N
    # H[0] is fixed to 1.0 (since sum(h)=1)
    h = np.fft.irfft(H, n=N)
    return h

def _autoconv_max(h):
    """Compute the max of the full autoconvolution of h using FFT."""
    n = len(h)
    size = 2 * n - 1
    fsize = 1 << (size - 1).bit_length()
    H = np.fft.rfft(h, n=fsize)
    conv = np.fft.irfft(H * H, n=fsize)[:size]
    return np.max(conv)

def _exact_value(h):
    """Compute the exact task value."""
    S = np.sum(h)
    if abs(S) < 1e-12:
        return 1e10
    M = _autoconv_max(h)
    return abs(2.0 * N * M) / (S * S)

def _objective(params):
    """
    Objective function for L-BFGS-B.
    params: flattened real/imag parts of H[1:N/2]
    """
    try:
        # Reconstruct H
        H = np.zeros(N, dtype=complex)
        H[0] = 1.0
        
        # Unpack params
        # H[1] to H[N//2]
        # H[N - m] = conj(H[m])
        # If N is even, H[N//2] is real.
        
        half = N // 2
        # params contains Re(H[1]), Im(H[1]), ..., Re(H[half]), Im(H[half])?
        # Actually, for N even, H[N//2] is real.
        # Let's just say params has length 2 * (N//2 - 1) + 1 ?
        # Simpler: just optimize H[1] to H[N-1] with symmetry constraint?
        # No, let's optimize the first half.
        
        # Let's use a simpler parameterization:
        # x = [Re(H1), Im(H1), ..., Re(H_{N/2-1}), Im(H_{N/2-1}), Re(H_{N/2})]
        
        idx = 0
        for m in range(1, half):
            H[m] = params[idx] + 1j * params[idx+1]
            idx += 2
        # H[half] is real
        H[half] = params[idx]
        
        # Fill the rest
        for m in range(1, half):
            H[N-m] = np.conj(H[m])
            
        h = np.fft.irfft(H, n=N)
        
        # Check sum
        S = np.sum(h)
        if abs(S) < 1e-10:
            return 1e10
            
        # Normalize to sum=1 for objective calculation stability, 
        # but the value formula is scale invariant.
        # However, H[0]=1 implies sum(h)=1.
        # Due to numerical errors, sum(h) might not be exactly 1.
        # We should probably re-normalize h or just use the value formula.
        
        # To ensure we are minimizing the correct metric:
        M = _autoconv_max(h)
        val = abs(2.0 * N * M) / (S * S)
        
        # We want to minimize val.
        # But val is non-smooth due to max.
        # Let's try to minimize the variance of the convolution instead?
        # Or just use val with a smooth approximation?
        # L-BFGS-B can handle non-smooth functions but might be slow.
        # Let's stick to val for now, but maybe add a small penalty for variance?
        
        # Alternative: Minimize sum((c - mean_c)^2)
        # This promotes flatness.
        # Let's try a combination.
        
        c = np.fft.irfft(np.fft.rfft(h, n=2*N-1)**2, n=2*N-1)[:2*N-1]
        mean_c = np.mean(c)
        var_c = np.var(c)
        
        # The peak is bounded by mean + sqrt(var * something)?
        # Minimizing variance is a good proxy for minimizing peak if the distribution is somewhat Gaussian.
        
        return var_c + 0.1 * val
        
    except Exception:
        return 1e10

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_h = np.ones(N)
    best_val = _exact_value(best_h)
    
    # Initial H: H[0]=1, others 0. This gives h = 1/N.
    # Let's start with random phases to escape local minima.
    
    # Number of parameters: 2 * (N//2 - 1) + 1
    half = N // 2
    n_params = 2 * (half - 1) + 1
    
    # Bounds: H components can be anything?
    # If H is too large, h will have large values, but value is scale invariant.
    # However, large H might cause numerical issues.
    # Let's bound them to [-10, 10].
    bounds = [(-10.0, 10.0)] * n_params
    
    # Multiple restarts
    rng = np.random.RandomState(42)
    
    max_restarts = 5
    
    for i in range(max_restarts):
        if time.time() > deadline - 20:
            break
            
        # Random initial guess
        x0 = rng.uniform(-1.0, 1.0, size=n_params)
        
        try:
            res = so.minimize(
                _objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.fun < 1e10: # Valid
                # Reconstruct h
                H = np.zeros(N, dtype=complex)
                H[0] = 1.0
                idx = 0
                for m in range(1, half):
                    H[m] = res.x[idx] + 1j * res.x[idx+1]
                    idx += 2
                H[half] = res.x[idx]
                for m in range(1, half):
                    H[N-m] = np.conj(H[m])
                
                h = np.fft.irfft(H, n=N)
                
                # Ensure sum is not zero
                S = np.sum(h)
                if abs(S) > 1e-10:
                    val = _exact_value(h)
                    if val < best_val:
                        best_val = val
                        best_h = h
        except Exception:
            pass

    # Final polish on best_h?
    # We can do a few steps of coordinate descent on h directly?
    # Or just return best_h.
    
    return best_h
