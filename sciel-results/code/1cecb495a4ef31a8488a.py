# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Logarithmic polynomial basis (Multiplicative Power Law model)
    # This approach models y as a product of powers of x, y, z and interaction terms.
    # y = exp(p0 + p1*log(x) + p2*log(y) + p3*log(z) + p4*log(x)*log(y) + ...)
    # Or simply a linear combination of log-terms if we don't exponentiate.
    # Let's use a linear combination of log-terms (which is equivalent to a multiplicative model in log-space).
    # Actually, if we just return the linear combo, it's additive in log-space.
    # If the target is y, and y = x^a * y^b * z^c, then log(y) = a log(x) + ...
    # So we should probably predict log(y) and then exp it?
    # But the evaluator minimizes MSE of (prediction - y).
    # If we return exp(linear_combo), the gradient can be unstable.
    # Let's try returning the linear combination of log-terms directly?
    # No, that assumes y is linear in logs, i.e., y = c * x^a... which is wrong if y is additive.
    
    # Let's try a different new family: **Gaussian Radial Basis Functions**
    # Fixed centers at (1,1,1), (1.5,1.5,1.5), (1,1.5,1) etc.
    # But we only have 10 params.
    # Let's use 5 centers and 5 weights?
    # Centers:
    # c1 = (1.0, 1.0, 1.0)
    # c2 = (1.5, 1.5, 1.5)
    # c3 = (1.0, 1.5, 1.0)
    # c4 = (1.5, 1.0, 1.5)
    # c5 = (1.25, 1.25, 1.25)
    # Weights: p[0..4]
    # Widths: fixed? e.g., sigma = 0.5
    # Bias: p[5]
    # Remaining 4 params? Maybe adjust widths?
    
    # Let's try **Spline** basis.
    # Knots at 1.0, 1.5.
    # Basis functions: max(0, x-k), max(0, y-k), etc.
    # This is a distinct family: Piecewise Linear / Splines.
    
    # Let's implement a Cubic Spline Basis.
    # Knots: 1.0, 1.5
    # For each variable, we have 3 basis functions: 1, (x-1.0)_+, (x-1.5)_+
    # Total 3 variables * 3 basis = 9 functions.
    # Plus 1 bias = 10 params.
    # This is a very strong, non-polynomial (locally linear/cubic) family.
    
    k1 = 1.0
    k2 = 1.5
    
    # Basis for x
    b_x0 = np.ones_like(x)
    b_x1 = np.maximum(0, x - k1)
    b_x2 = np.maximum(0, x - k2)
    
    # Basis for y
    b_y0 = np.ones_like(y)
    b_y1 = np.maximum(0, y - k1)
    b_y2 = np.maximum(0, y - k2)
    
    # Basis for z
    b_z0 = np.ones_like(z)
    b_z1 = np.maximum(0, z - k1)
    b_z2 = np.maximum(0, z - k2)
    
    p = params
    
    # We need to map these 9 basis functions + 1 bias to 10 params.
    # However, b_x0, b_y0, b_z0 are all 1. So we can combine them into a single bias term.
    # Let's refine the basis to be independent.
    # Basis:
    # 0: 1 (bias)
    # 1: (x-1.0)_+
    # 2: (x-1.5)_+
    # 3: (y-1.0)_+
    # 4: (y-1.5)_+
    # 5: (z-1.0)_+
    # 6: (z-1.5)_+
    # 7: (x-1.0)_+ * (y-1.0)_+  (Interaction)
    # 8: (x-1.0)_+ * (z-1.0)_+
    # 9: (y-1.0)_+ * (z-1.0)_+
    
    # This is a Piecewise Linear Interaction model.
    
    bx1 = np.maximum(0, x - 1.0)
    bx2 = np.maximum(0, x - 1.5)
    by1 = np.maximum(0, y - 1.0)
    by2 = np.maximum(0, y - 1.5)
    bz1 = np.maximum(0, z - 1.0)
    bz2 = np.maximum(0, z - 1.5)
    
    return (
        p[0]
        + p[1] * bx1
        + p[2] * bx2
        + p[3] * by1
        + p[4] * by2
        + p[5] * bz1
        + p[6] * bz2
        + p[7] * bx1 * by1
        + p[8] * bx1 * bz1
        + p[9] * by1 * bz1
    )
# EVOLVE-BLOCK-END
