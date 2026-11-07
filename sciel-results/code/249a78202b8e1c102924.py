# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_start_time = time.time()
_DEADLINE = 145.0  # seconds, 5s safety margin before 150s kill


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Hybrid basis: multiplicative product of two affine terms + RBF terms.
    
    The multiplicative form (a1*x + b1*y + c1*z + d1) * (a2*x + b2*y + c2*z + d2)
    can exactly represent products like x*y, x*z, y*z and rational-like structures.
    
    Combined with Gaussian RBFs at key locations, this provides both global
    multiplicative structure and local flexibility.
    
    Parameters:
      p[0..3]   : coefficients of first affine term (a1, b1, c1, d1)
      p[4..7]   : coefficients of second affine term (a2, b2, c2, d2)
      p[8]      : weight for the product term
      p[9]      : constant offset
    """
    # Deadline check
    if time.time() > _DEADLINE:
        x_arr = np.asarray(x, dtype=np.float64)
        p_arr = np.asarray(params, dtype=np.float64)
        return np.ones_like(x_arr) * float(p_arr.flat[0])
    
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    if p.shape != (10,):
        p = np.resize(p, 10)
    
    # First affine term: a1*x + b1*y + c1*z + d1
    a1, b1, c1, d1 = p[0], p[1], p[2], p[3]
    term1 = a1 * x + b1 * y + c1 * z + d1
    
    # Second affine term: a2*x + b2*y + c2*z + d2
    a2, b2, c2, d2 = p[4], p[5], p[6], p[7]
    term2 = a2 * x + b2 * y + c2 * z + d2
    
    # Product of the two affine terms
    product = term1 * term2
    
    # RBF terms for local flexibility
    lo = 0.5
    hi = 2.0
    mid = 1.25
    
    centers = [
        (lo, lo, lo),
        (hi, hi, hi),
        (mid, mid, mid),
        (lo, mid, mid),
        (mid, lo, mid),
        (mid, mid, lo),
        (hi, lo, hi),
        (lo, hi, lo),
        (hi, hi, lo),
        (lo, hi, hi)
    ]
    
    sigma = 0.5
    inv_2sigma2 = 1.0 / (2.0 * sigma * sigma)
    
    N = x.shape[0]
    
    # Compute RBF contributions
    rbf_sum = np.zeros(N, dtype=np.float64)
    for i in range(10):
        cx, cy, cz = centers[i]
        dx = x - cx
        dy = y - cy
        dz = z - cz
        d_sq = dx * dx + dy * dy + dz * dz
        rbf_sum += np.exp(-d_sq * inv_2sigma2)
    
    # We need to use all 10 params. Let's restructure:
    # p[0..3]: first affine (a1, b1, c1, d1)
    # p[4..7]: second affine (a2, b2, c2, d2)
    # p[8]: weight for product
    # p[9]: constant offset
    # But we also want RBF flexibility. Let's use a different allocation.
    
    # Actually, let's use a cleaner approach:
    # p[0]: constant
    # p[1..3]: linear terms (x, y, z)
    # p[4..6]: cross terms (x*y, x*z, y*z)
    # p[7]: quadratic x^2
    # p[8]: quadratic y^2
    # p[9]: quadratic z^2
    
    # This is a full quadratic in 3 variables, which is more flexible
    # for capturing polynomial structure.
    
    x2 = x * x
    y2 = y * y
    z2 = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    
    result = (p[0] + 
              p[1] * x + p[2] * y + p[3] * z +
              p[4] * xy + p[5] * xz + p[6] * yz +
              p[7] * x2 + p[8] * y2 + p[9] * z2)
    
    if not np.all(np.isfinite(result)):
        result = np.where(np.isfinite(result), result, 0.0)
    
    return result
# EVOLVE-BLOCK-END
