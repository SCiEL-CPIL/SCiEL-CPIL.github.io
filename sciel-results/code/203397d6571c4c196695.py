import numpy as np
from scipy.optimize import minimize
import time

N = 400
START_TIME = time.time()
TIME_LIMIT = 115  # Safety margin before 120s/150s kill

def fast_autoconv_max(h):
    """
    Compute max of linear auto-convolution of h using FFT.
    h: 1D array of length N
    """
    # Pad to at least 2*N - 1 for linear convolution
    # Next power of 2 for FFT efficiency
    n_fft = 1
    while n_fft < 2 * N - 1:
        n_fft *= 2
    
    # FFT of h
    H = np.fft.rfft(h, n=n_fft)
    
    # Power spectrum
    S = H * np.conj(H)
    
    # Inverse FFT to get auto-convolution
    # Note: irfft returns the real part of the inverse FFT
    # The result is the circular convolution of h with itself, 
    # but since we padded to >= 2N-1, the first 2N-1 elements 
    # correspond to the linear convolution.
    conv = np.fft.irfft(S, n=n_fft)
    
    # The linear convolution of length N with itself has length 2N-1.
    # In the FFT domain with zero padding, the linear convolution 
    # is stored in the first 2N-1 samples if we arrange it correctly?
    # Actually, np.convolve(h, h) gives [c_0, ..., c_{2N-2}].
    # The FFT method gives the circular convolution of the padded h.
    # Because h is zero-padded to n_fft, the circular convolution 
    # equals the linear convolution for indices 0 to n_fft-1.
    # So conv[0 : 2*N-1] is the linear auto-convolution.
    
    max_val = np.max(conv[:2*N-1])
    return max_val

def objective(x):
    """
    x: array of length N-1
    Returns the value to minimize.
    """
    # Reconstruct h
    # h[0:N-1] = x
    # h[N-1] = 1 - sum(x)
    s_x = np.sum(x)
    h = np.empty(N)
    h[:N-1] = x
    h[N-1] = 1.0 - s_x
    
    # Check for numerical stability
    if not np.all(np.isfinite(h)):
        return 1e10
        
    # Calculate max auto-conv
    max_conv = fast_autoconv_max(h)
    
    # The target value is abs(2 * N * max_conv / sum(h)^2)
    # Since sum(h) = 1, it is abs(2 * N * max_conv)
    # max_conv is always positive (sum of squares is max_conv[0] >= 0)
    return 2.0 * N * max_conv

def construct():
    global START_TIME, TIME_LIMIT
    
    # Fallback result
    best_h = np.ones(N) / N  # Sum is 1, value is 2*N*(1/N^2) = 2/N ? No.
    # For h = ones/N, sum=1. 
    # conv[0] = sum(h^2) = N * (1/N^2) = 1/N.
    # max(conv) is likely conv[0] or conv[1]?
    # For constant signal, conv is triangular. Peak is at 0.
    # Value = 2 * N * (1/N) = 2.
    # This matches the seed score of 2.0.
    
    best_val = 2.0
    
    rng = np.random.default_rng(42)
    
    # Try several initializations
    # 1. Flat (Seed)
    # 2. Gaussian window
    # 3. Raised Cosine window
    # 4. Random smooth
    
    inits = []
    
    # Init 1: Gaussian
    t = np.linspace(-3, 3, N)
    g = np.exp(-t**2 / 2)
    g = g / np.sum(g)
    inits.append(g)
    
    # Init 2: Raised Cosine
    rc = 0.5 * (1 + np.cos(2 * np.pi * (np.arange(N) - N/2) / N))
    rc = rc / np.sum(rc)
    inits.append(rc)
    
    # Init 3: Random smooth (low freq noise)
    for i in range(5):
        spec = np.zeros(N//2 + 1, dtype=complex)
        # Low frequency emphasis
        weights = 1.0 / (1.0 + np.arange(N//2 + 1))
        spec_mag = rng.random(N//2 + 1) * weights
        spec_phase = rng.uniform(-np.pi, np.pi, N//2 + 1)
        spec = spec_mag * np.exp(1j * spec_phase)
        spec[0] = 0 # Zero mean initially? No, we need sum=1.
        h_rand = np.fft.irfft(spec, n=N)
        h_rand = h_rand - np.mean(h_rand)
        # Make it positive? Not strictly necessary, but helps.
        # Just normalize to sum 1
        if np.sum(h_rand) != 0:
            h_rand = h_rand / np.sum(h_rand)
        else:
            h_rand = np.ones(N) / N
        inits.append(h_rand)

    # Evaluate initializations
    for h_init in inits:
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
        val = objective(h_init[:N-1])
        if val < best_val:
            best_val = val
            best_h = h_init.copy()
            
    # Optimize from the best initialization and a few others
    # We will run L-BFGS-B on the top few initializations
    
    # Sort inits by initial value
    scored_inits = []
    for h_init in inits:
        val = objective(h_init[:N-1])
        scored_inits.append((val, h_init))
    scored_inits.sort(key=lambda x: x[0])
    
    # Optimize top 3
    for i, (init_val, h_init) in enumerate(scored_inits[:3]):
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
            
        x0 = h_init[:N-1]
        
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'maxfun': 20000, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.fun < best_val:
                best_val = result.fun
                # Reconstruct h
                x_opt = result.x
                h_opt = np.empty(N)
                h_opt[:N-1] = x_opt
                h_opt[N-1] = 1.0 - np.sum(x_opt)
                best_h = h_opt
        except Exception as e:
            pass
            
    # Final check
    if not np.all(np.isfinite(best_h)) or abs(np.sum(best_h)) < 1e-10:
        return np.ones(N) / N
        
    return best_h

# Test run to ensure it works
if __name__ == "__main__":
    h = construct()
    s = np.sum(h)
    conv = np.convolve(h, h)
    val = abs(2 * N * np.max(conv) / s**2)
    print(f"Score: {val}")
