import numpy as np
import time

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: Chebyshev polynomials (linear part) + bounded sinusoidal terms (non-linear part).
    
    The Chebyshev part provides a stable, well-conditioned linear approximation base.
    The sinusoidal part adds non-linear expressiveness with bounded parameters,
    allowing BFGS to fit complex smooth functions by separating stable linear
    coefficients from robustly bounded sinusoidal parameters.
    
    Model:
        y(x) = sum_{k=0}^{5} c_k * T_k(t) + sum_{j=1}^{3} a_j * sin(f_j * t + p_j)
    
    where t = x / 3 maps [-3, 3] to [-1, 1].
    
    Parameters (10 total):
        params[0:6]  = c_0 ... c_5  (Chebyshev coefficients)
        params[6:9]  = a_1, a_2, a_3 (amplitudes, bounded via tanh)
        params[9]    = shared frequency scale (bounded via tanh, mapped to [0.5, 5.0])
    
    All non-linear parameters are bounded via tanh to keep BFGS stable.
    """
    x = np.asarray(x, dtype=np.float64)
    t = x / 3.0  # Map to [-1, 1]
    
    # --- Chebyshev polynomial part (linear in params) ---
    # Evaluate T_0 through T_5 on t
    n_cheb = 6
    P = np.zeros((len(t), n_cheb), dtype=np.float64)
    P[:, 0] = 1.0
    if n_cheb > 1:
        P[:, 1] = t
    for k in range(2, n_cheb):
        P[:, k] = 2.0 * t * P[:, k-1] - P[:, k-2]
    
    # Normalize for orthonormality on [-1, 1]
    # ||T_0||^2 = 1, ||T_k||^2 = 1/2 for k >= 1
    # So: T_0 stays, T_k * sqrt(2) for k >= 1
    P[:, 1:] *= np.sqrt(2.0)
    
    # Linear part
    y_cheb = P @ params[0:n_cheb]
    
    # --- Sinusoidal part (non-linear but bounded) ---
    # Amplitudes: bounded via tanh, scaled to [-2, 2]
    a = 2.0 * np.tanh(params[n_cheb:n_cheb+3])  # 3 amplitudes
    
    # Frequency: single shared frequency, bounded via tanh, mapped to [0.5, 5.0]
    f_raw = np.tanh(params[n_cheb+3])  # in [-1, 1]
    f = 0.5 + 2.25 * (f_raw + 1.0)  # map to [0.5, 5.0]
    
    # Three sinusoidal terms with fixed phase offsets for diversity
    phases = np.array([0.0, np.pi/3.0, 2.0*np.pi/3.0])
    
    # Compute sin(f * t + phase_j) for each j
    # t is (N,), f is scalar, phases is (3,)
    # We want (N, 3) output
    y_sin = np.zeros_like(t)
    for j in range(3):
        y_sin += a[j] * np.sin(f * t + phases[j])
    
    result = y_cheb + y_sin
    
    # Ensure finite
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
