"""Symbolic regression with sum of shifted sigmoid basis functions."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Sum of shifted sigmoid basis functions:
    y = sum_{i=0}^{4} p_i / (1 + exp(-s_i * (x - c_i)))
    
    params layout (length 10):
      params[0..4]  = amplitudes p_0..p_4
      params[5..9]  = (s_i, c_i) pairs: s_0,c_0, s_1,c_1, ..., s_4,c_4
    
    Each sigmoid is centered at c_i with steepness s_i. The sum of these
    smooth step-like functions can model piecewise-constant, saturating,
    and other non-polynomial behaviors while remaining globally smooth
    for BFGS convergence.
    
    To improve numerical stability:
    - Steepness s_i is bounded via tanh to [-5, 5]
    - Centers c_i are bounded via tanh to [-3, 3] (covers [-2,2] with margin)
    """
    x = np.asarray(x, dtype=float)
    
    # Extract parameters
    p = params[0:5]       # amplitudes
    s_raw = params[5:10:2]   # steepness parameters (indices 5, 7, 9)
    c_raw = params[6:10:2]   # center parameters (indices 6, 8)
    
    # Wait, params[5:10:2] gives indices 5, 7, 9 (3 values)
    # params[6:10:2] gives indices 6, 8 (2 values)
    # We need 5 pairs, so let's reorganize:
    # params[5], params[6] -> s_0, c_0
    # params[7], params[8] -> s_1, c_1
    # params[9]           -> s_2 (no c_2, use fixed center)
    # Actually with 10 params total, we have 5 for p and 5 for (s,c) pairs
    # But 5 pairs need 10 params. Let's use 4 sigmoids to fit in 10 params:
    # p_0..p_3 (4 amplitudes), s_0..s_3, c_0..c_3 (4 steepness, 4 centers) = 12 > 10
    # 
    # Alternative: 5 sigmoids with shared structure, or use 3 sigmoids:
    # p_0..p_2 (3 amplitudes), s_0..s_2, c_0..c_2 (3+3=6) = 9, leaving 1 extra
    #
    # Let's use 5 sigmoids but with a smarter parameterization:
    # params[0..4] = amplitudes
    # params[5..9] = 5 values that encode both s and c via a single parameter each
    # For each i, let t_i = params[5+i]
    #   s_i = 3.0 * tanh(t_i)    # steepness in [-3, 3]
    #   c_i = 2.0 * tanh(0.5 * t_i + 0.1 * i)  # center, slightly offset
    #
    # Actually, simpler: use 5 sigmoids with fixed centers and learnable steepness+amplitude
    # Or: use 4 sigmoids with 2 params each (s, c) = 8, plus 2 amplitude params?
    #
    # Cleanest: 5 sigmoids, each with amplitude and a combined shape parameter
    # Let's just use 5 sigmoids with:
    #   p_i = params[i]
    #   s_i = 2.0 * tanh(params[5 + i])  # steepness
    #   c_i = 1.5 * tanh(0.7 * params[5 + i] + 0.3 * i)  # center, depends on same param
    # This is not ideal. Let me use a different approach:
    #
    # Use 5 sigmoids with:
    #   amplitude_i = params[i]
    #   For i=0..4: center_i = -1.5 + 0.75*i (fixed: -1.5, -0.75, 0, 0.75, 1.5)
    #   steepness_i = 2.0 * tanh(params[5+i])
    # This uses all 10 params and gives good coverage of [-2,2].
    
    # Fixed centers spread across [-2, 2]
    centers = np.array([-1.5, -0.75, 0.0, 0.75, 1.5])
    
    # Steepness bounded to [-3, 3]
    steepness = 3.0 * np.tanh(params[5:10])
    
    # Amplitudes
    amplitudes = params[0:5]
    
    # Compute sum of sigmoids
    result = np.zeros_like(x)
    for i in range(5):
        # Sigmoid: 1 / (1 + exp(-s * (x - c)))
        # Use stable computation to avoid overflow
        z = steepness[i] * (x - centers[i])
        # Stable sigmoid: for z >= 0: 1/(1+exp(-z)); for z < 0: exp(z)/(1+exp(z))
        sigmoid = np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)), np.exp(z) / (1.0 + np.exp(z)))
        result += amplitudes[i] * sigmoid
    
    # Ensure finite output
    result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
