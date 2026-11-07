"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Symbolic regression equation using 5 Gaussian basis functions.
    The 10 parameters define 5 (center, width) pairs for Gaussian bumps,
    allowing adaptive local fitting of the hidden function on [-3, 3].
    
    params[0::2] = centers c_0..c_4
    params[1::2] = widths  w_0..w_4 (amplitudes are fixed at 1.0 for stability)
    
    Actually, to give BFGS more flexibility, we use:
    params[0::2] = centers
    params[1::2] = inverse widths (1/width)
    And we add a small constant offset via the sum of Gaussians.
    
    Better approach: use 5 Gaussians with learnable amplitude, center, and width?
    But we only have 10 params. So: 5 Gaussians, each with center and width,
    and amplitudes fixed at 1.0. This might be too restrictive.
    
    Alternative: 4 Gaussians (8 params) + 1 linear term + 1 constant = 10 params.
    This gives local bumps + global trend.
    
    Let's try: 
    params[0] = constant offset
    params[1] = linear coefficient
    params[2], params[3] = center, inv_width for Gaussian 1
    params[4], params[5] = center, inv_width for Gaussian 2
    params[6], params[7] = center, inv_width for Gaussian 3
    params[8], params[9] = center, inv_width for Gaussian 4
    
    Each Gaussian has amplitude 1.0.
    """
    x = np.asarray(x, dtype=np.float64)
    
    # Clip parameters to avoid numerical issues
    p = np.clip(params, -1e3, 1e3)
    
    # Linear part
    result = p[0] + p[1] * x
    
    # Gaussian bumps
    # For each of 4 Gaussians: center = p[2+2*i], inv_width = p[3+2*i]
    for i in range(4):
        c = p[2 + 2 * i]
        inv_w = np.abs(p[3 + 2 * i]) + 0.1  # ensure positive, min width 0.1
        # Gaussian: exp(-(x - c)^2 * inv_w^2)
        # Actually, let's use: exp(-(x - c)^2 / (2 * width^2)) where width = 1/inv_w
        # So: exp(-(x - c)^2 * inv_w^2 / 2)
        gauss = np.exp(-0.5 * ((x - c) * inv_w) ** 2)
        result += gauss
    
    # Ensure finite output
    if not np.all(np.isfinite(result)):
        result = np.clip(p[0], -1e3, 1e3) + np.clip(p[1], -1e3, 1e3) * x
    
    return result
# EVOLVE-BLOCK-END
