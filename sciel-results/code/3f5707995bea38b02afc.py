import numpy as np
import time

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid model: Global Linear Trend + 3 Localized Quadratic Features.
    
    Parameters (10 total):
    p[0..3]: Global linear coefficients (intercept, x, y, z)
    p[4..6]: Centers for x, y, z localized features (cx, cy, cz)
    p[7..9]: Widths (scales) for x, y, z localized features (sx, sy, sz)
    
    The localized features are quadratic in the deviation from their centers, 
    weighted by a Gaussian-like decay to localize them.
    """
    p = params
    
    # Safety check for params length
    if len(p) < 10:
        p = np.pad(p, (0, 10 - len(p)))
        
    # Extract parameters
    b0, bx, by, bz = p[0], p[1], p[2], p[3]
    cx, cy, cz = p[4], p[5], p[6]
    sx, sy, sz = p[7], p[8], p[9]
    
    # Ensure widths are positive and not too small to avoid division by zero or extreme gradients
    # Using a softplus-like transformation or simple clipping for stability
    # Since BFGS starts at 1, and inputs are in [0.5, 2], widths around 0.5-1.0 are reasonable.
    # We clip widths to [0.1, 2.0] to prevent numerical instability.
    sx = np.clip(sx, 0.1, 2.0)
    sy = np.clip(sy, 0.1, 2.0)
    sz = np.clip(sz, 0.1, 2.0)
    
    # Centers are unconstrained, but we can clip them to a reasonable range [0.0, 2.5]
    # to keep them near the data domain.
    cx = np.clip(cx, 0.0, 2.5)
    cy = np.clip(cy, 0.0, 2.5)
    cz = np.clip(cz, 0.0, 2.5)
    
    # Global Linear Part
    pred = b0 + bx * x + by * y + bz * z
    
    # Localized Quadratic Features
    # Feature 1: Centered on cx, active mainly around x=cx
    # We use a Gaussian window to localize the quadratic term.
    # Window: exp(-((x - cx) / sx)^2)
    # Term: Window * ( (x - cx)^2 )
    # This creates a "bump" or "dip" in the function shape around cx.
    
    dx = (x - cx) / sx
    dy = (y - cy) / sy
    dz = (z - cz) / sz
    
    # Gaussian windows (smooth and localized)
    wx = np.exp(-dx * dx)
    wy = np.exp(-dy * dy)
    wz = np.exp(-dz * dz)
    
    # Quadratic deviations
    qx = (x - cx) * (x - cx)
    qy = (y - cy) * (y - cy)
    qz = (z - cz) * (z - cz)
    
    # We add these localized terms with fixed coefficients (or we could learn them, 
    # but we have only 10 params. Using 3 for centers, 3 for widths, 4 for linear leaves 0 for 
    # amplitudes. We can assume unit amplitude or share them. 
    # To make it more powerful, let's use the linear part to also capture some local slope 
    # by making the "quadratic" term actually a full local quadratic: a*(x-c)^2 + b*(x-c) + c.
    # But we don't have enough params for full local quadratics for all 3 dims.
    # 
    # Alternative: Use the 3 "width" params also as amplitudes? No, they are scales.
    # Let's stick to the form: Linear + Sum( Amplitude_i * exp(-((var - center_i)/width_i)^2) * (var - center_i)^2 )
    # Since we don't have amplitude params, we assume amplitude is 1.0.
    # This might be too restrictive.
    
    # Let's re-evaluate parameter usage.
    # 10 params.
    # Maybe: 1 Intercept, 3 Linear Coeffs (x,y,z), 3 Centers, 3 Widths.
    # The localized term adds non-linearity.
    
    pred += wx * qx
    pred += wy * qy
    pred += wz * qz
    
    return pred
