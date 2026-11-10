"""Symbolic regression with 1 input: local B-spline basis for flexible fitting."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Local piecewise cubic B-spline basis with 10 parameters.
    
    Uses 10 cubic B-spline basis functions over [-2, 2] with evenly spaced
    knots. This provides local flexibility (each basis function has compact
    support) while maintaining C2 smoothness. The model is linear in params,
    making the optimization convex and well-conditioned for BFGS.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    
    # Define knot vector: 10 internal knots evenly spaced in [-2, 2]
    # For cubic B-splines, we need enough knots. We'll use a simple approach:
    # 10 basis functions, each centered at a knot, with local support.
    
    # Knot positions (centers of basis functions)
    # 10 evenly spaced points in [-2, 2]
    knots = np.linspace(-2.0, 2.0, 10)
    
    # Width of each basis function - use spacing between knots
    # Spacing is 4/9 ≈ 0.444
    # Use a width that gives good overlap: about 1.5 * spacing
    spacing = 4.0 / 9.0
    width = 1.5 * spacing
    
    # Compute B-spline-like basis using normalized cubic polynomials
    # For each basis i, compute the value at each x
    # Using a compact cubic kernel: (1 - (x-c)^2/w^2)^3 for |x-c| < w, else 0
    # This is a C2 smooth compactly supported basis
    
    x_expanded = x[:, np.newaxis]  # shape (n, 1)
    knots_expanded = knots[np.newaxis, :]  # shape (1, 10)
    
    # Compute normalized distance
    dist = np.abs(x_expanded - knots_expanded) / width  # shape (n, 10)
    
    # Cubic B-spline-like kernel: (1 - t^2)^3 for t in [0, 1], 0 otherwise
    # This is actually (1-t^2)^3 which is C2 at t=1
    # For better approximation to true B-splines, use this compact form
    with np.errstate(invalid='ignore'):
        t = np.clip(dist, 0.0, 1.0)
        basis = (1.0 - t**2)**3
        basis[dist > 1.0] = 0.0
    
    # Normalize each basis function so that the sum of basis functions is ~1
    # This helps with conditioning
    basis_sums = basis.sum(axis=0, keepdims=True)
    basis_sums = np.where(basis_sums > 1e-10, basis_sums, 1.0)
    basis = basis / basis_sums
    
    # Linear combination
    result = basis @ params
    
    return result
# EVOLVE-BLOCK-END
