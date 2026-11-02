"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import random

def _fast_autocorr_max_abs(x):
    """
    Compute the maximum absolute value of the linear autocorrelation of x.
    Uses FFT for O(N log N) complexity.
    """
    M = 1
    while M < 2 * N - 1:
        M *= 2
        
    X = np.fft.fft(x, M)
    P = X * np.conj(X)
    ac = np.fft.ifft(P).real
    
    # The first N values of ac correspond to lags 0 to N-1.
    # Due to symmetry, ac[k] for k >= 0 is the correlation.
    # We want max absolute value.
    return np.max(np.abs(ac[:N]))

def _softmax_autocorr(x, beta=10.0):
    """
    Smooth approximation to max(|ac|).
    Uses log-sum-exp trick for stability.
    """
    M = 1
    while M < 2 * N - 1:
        M *= 2
        
    X = np.fft.fft(x, M)
    P = X * np.conj(X)
    ac = np.fft.ifft(P).real[:N]
    
    # We want to approximate max(|ac|).
    # Let y = |ac|. max(y) = lim_{beta->inf} (1/beta) * log(sum(exp(beta*y)))
    # To avoid overflow, we can scale y or use a simpler penalty.
    # A common alternative is sum(y^p) for large p, or just sum(y^2) which is L2 norm.
    # L2 norm minimization is smooth and convex (in x).
    # Minimizing L2 norm of autocorrelation is a well-known problem (Golay sequences, etc.).
    # It doesn't directly minimize the max, but it tends to flatten the autocorrelation.
    # Let's try minimizing the sum of squares of the sidelobes (excluding lag 0).
    # Lag 0 is always sum(x^2). For +/-1, sum(x^2) = N.
    # Sidelobe energy = sum_{k=1}^{N-1} ac[k]^2.
    # Total energy in autocorrelation domain is N * sum(x^2) = N^2.
    # So minimizing sidelobe energy is equivalent to maximizing the "peakiness" at lag 0 relative to others,
    # which is good for low peak sidelobe ratio.
    
    # Let's use the sum of squares of the autocorrelation for lags 1 to N-1.
    # This is a smooth, differentiable function.
    # Note: ac is real.
    return np.sum(ac[1:] ** 2)

def _spectral_init(rng):
    """
    Generate an initial signal using spectral phase retrieval principles.
    We fix the magnitude spectrum to a raised-cosine (Hann-like) window to
    minimize the peak-to-sidelobe ratio in the time domain autoconvolution,
    and assign random phases.
    """
    M = 1
    while M < 2 * N - 1:
        M *= 2
        
    freqs = np.fft.fftfreq(M)
    freqs_norm = freqs * M
    
    sigma = M / 8.0
    mag = np.exp(-0.5 * (freqs_norm / sigma) ** 2)
    
    half_M = M // 2
    phases = rng.uniform(0, 2 * np.pi, half_M)
    
    spec = np.zeros(M, dtype=np.complex128)
    spec[0] = mag[0]
    if M % 2 == 0:
        spec[half_M] = mag[half_M]
        
    for k in range(1, half_M):
        spec[k] = mag[k] * np.exp(1j * phases[k-1])
        spec[M-k] = np.conj(spec[k])
        
    x = np.fft.ifft(spec).real
    
    norm = np.linalg.norm(x)
    if norm > 0:
        x = x / norm
    return x

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # 10s safety margin
    
    rng = np.random.default_rng(42)
    
    best_val = float('inf')
    best_x = np.zeros(N, dtype=np.float64)
    
    # Strategy:
    # 1. Use multiple spectral initializations.
    # 2. Optimize the continuous signal to minimize the L2 norm of the sidelobes (smooth objective).
    #    This is a different optimization landscape than the max absolute value (non-smooth).
    #    Minimizing L2 energy often leads to flatter autocorrelation, which helps with the max.
    # 3. Threshold to +/-1.
    # 4. Local search refinement on the +/-1 sequence using the actual max absolute value metric.
    
    num_inits = 5
    inits = []
    for _ in range(num_inits):
        x_init = _spectral_init(rng)
        inits.append(x_init)
        
    for x0 in inits:
        if time.time() > deadline - 5.0:
            break
            
        # Objective: Minimize sum of squares of sidelobes
        def objective(x):
            return _softmax_autocorr(x)
            
        try:
            res = minimize(objective, x0, method='L-BFGS-B', 
                           options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-5})
            
            x_opt = res.x
            # The continuous optimization result is not directly scored.
            # We must quantize.
            x_quant = np.sign(x_opt)
            x_quant[x_quant == 0] = 1
            
            val_quant = _fast_autocorr_max_abs(x_quant)
            
            if val_quant < best_val:
                best_val = val_quant
                best_x = x_quant.copy()
                
        except Exception:
            x_quant = np.sign(x0)
            x_quant[x_quant == 0] = 1
            val_quant = _fast_autocorr_max_abs(x_quant)
            if val_quant < best_val:
                best_val = val_quant
                best_x = x_quant.copy()

    # Final local search refinement on the best quantized solution
    current_x = best_x.copy()
    current_val = best_val
    
    time_left = deadline - time.time()
    if time_left > 5.0:
        # More aggressive local search: try flipping pairs of bits?
        # Or just single flips but with a larger budget if time permits.
        # The key is that the L2 optimization might land in a different basin of attraction
        # for the max metric than the direct max optimization.
        
        steps = int(time_left * 100)  # Faster loop, more steps
        for _ in range(steps):
            if time.time() > deadline - 1.0:
                break
                
            i = rng.integers(0, N)
            current_x[i] *= -1
            new_val = _fast_autocorr_max_abs(current_x)
            
            if new_val < current_val:
                current_val = new_val
                if current_val < best_val:
                    best_val = current_val
                    best_x = current_x.copy()
            else:
                current_x[i] *= -1  # Revert

    return best_x.astype(np.int8)
# EVOLVE-BLOCK-END
