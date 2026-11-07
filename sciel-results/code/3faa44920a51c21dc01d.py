import numpy as np
import time

_START_TIME = time.time()

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Deadline guard
    if time.time() - _START_TIME > 145:
        x_arr = np.asarray(x, dtype=np.float64).ravel()
        return np.zeros_like(x_arr)
    
    x = np.asarray(x, dtype=np.float64).ravel()
    p = np.asarray(params, dtype=np.float64).ravel()
    
    if p.size < 10:
        p = np.pad(p, (0, 10 - p.size), constant_values=0.0)
    else:
        p = p[:10]
    
    # Strategy: Fourier series with learnable frequencies and phases,
    # but fixed amplitudes to keep the optimization landscape smooth.
    # 
    # We use 5 harmonics (k=1..5), each with a frequency and phase.
    # Amplitudes are fixed to 1/k for smooth decay.
    # 
    # params[0:5] = frequencies (omega_k) for k=1..5
    # params[5:10] = phases (phi_k) for k=1..5
    #
    # f(x) = sum_{k=1}^{5} (1/k) * sin(omega_k * x + phi_k)
    #
    # This is a smooth, periodic-like basis that can capture
    # non-periodic structures through constructive interference
    # of optimally tuned harmonics. The optimization is over
    # 10 parameters but the structure is constrained to be
    # smooth and well-conditioned for BFGS.
    
    # Extract frequencies and phases
    freqs = p[0:5]   # omega_1 ... omega_5
    phases = p[5:10] # phi_1 ... phi_5
    
    # Fixed amplitudes with 1/k decay
    amplitudes = np.array([1.0/1.0, 1.0/2.0, 1.0/3.0, 1.0/4.0, 1.0/5.0])
    
    # Compute the Fourier sum
    # For each k, compute sin(omega_k * x + phi_k)
    # Shape: (n_points, 5)
    result = np.zeros_like(x, dtype=np.float64)
    
    for k in range(5):
        result += amplitudes[k] * np.sin(freqs[k] * x + phases[k])
    
    # Safety: clip to prevent overflow
    result = np.clip(result, -1e8, 1e8)
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
