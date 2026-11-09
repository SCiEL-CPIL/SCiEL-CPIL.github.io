"""
Symbolic regression using a hybrid basis: low-order polynomial + fixed sigmoid functions.
This structural change moves from a periodic Fourier representation to one that
can model local step-like or saturating behaviors, providing a different functional
geometry suitable for non-periodic, potentially non-smooth targets.
"""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Model: y(x) = sum_{i=0}^{4} p_i x^i + sum_{j=0}^{4} p_{5+j} * sigmoid(2.5 * (x - c_j))
    
    Polynomial part: p0 + p1*x + p2*x^2 + p3*x^3 + p4*x^4
    Sigmoid part: 5 fixed sigmoids with centers c_j in [-1.5, 1.5] and steepness 2.5.
    
    The model is linear in the 10 parameters, ensuring convex optimization
    and reliable convergence with BFGS starting from ones.
    """
    x = np.asarray(x, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    if p.ndim != 1:
        p = p.flatten()

    # 1. Polynomial basis (degree 4)
    result = p[0] + p[1]*x + p[2]*x**2 + p[3]*x**3 + p[4]*x**4
    
    # 2. Sigmoid basis (5 fixed centers)
    # Centers spread across the domain [-2, 2]
    centers = np.array([-1.5, -0.75, 0.0, 0.75, 1.5])
    steepness = 2.5
    
    # Compute sigmoids for all x and centers
    # Shape of x: (N,), Shape of centers: (5,)
    # We want to compute sigmoid(steepness * (x[:, None] - centers[None, :]))
    # But x might be 1D.
    x_expanded = x[:, np.newaxis]
    c_expanded = centers[np.newaxis, :]
    
    # Calculate arguments
    args = steepness * (x_expanded - c_expanded)
    
    # Calculate sigmoid values: shape (N, 5)
    sigs = 1.0 / (1.0 + np.exp(-args))
    
    # Linear combination with p[5:10]
    # p[5:10] shape (5,), sigs shape (N, 5)
    result += np.sum(p[5:10] * sigs, axis=1)
    
    return result
# EVOLVE-BLOCK-END
