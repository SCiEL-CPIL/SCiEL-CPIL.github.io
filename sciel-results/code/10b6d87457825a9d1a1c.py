"""Chebyshev polynomial basis for nonlinear oscillator symbolic regression."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, v: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Acceleration of a nonlinear oscillator modeled with a Chebyshev polynomial basis
    in scaled position and velocity.
    
    Inputs:
        x: position array
        v: velocity array
        params: array of 10 fitted constants
    
    Returns:
        Acceleration array (one prediction per row)
    """
    # Scale inputs to [-1, 1] range for Chebyshev basis
    # Training ranges: x in [-0.5728, 0.5275], v in [-0.1754, 0.2821]
    # Use center and half-range for scaling
    x_center = (0.5275 + (-0.5728)) / 2.0  # ≈ -0.02265
    x_half = (0.5275 - (-0.5728)) / 2.0    # ≈ 0.55015
    
    v_center = (0.2821 + (-0.1754)) / 2.0  # ≈ 0.05335
    v_half = (0.2821 - (-0.1754)) / 2.0    # ≈ 0.22875
    
    # Add small epsilon to avoid division by zero
    x_scaled = (x - x_center) / (x_half + 1e-10)
    v_scaled = (v - v_center) / (v_half + 1e-10)
    
    # Clip to [-1, 1] to stay in Chebyshev domain (for numerical stability)
    x_scaled = np.clip(x_scaled, -1.0, 1.0)
    v_scaled = np.clip(v_scaled, -1.0, 1.0)
    
    # Chebyshev polynomials of the first kind: T_0=1, T_1=x, T_{n+1}=2x*T_n - T_{n-1}
    def chebyshev_basis(t, degree=5):
        """Compute Chebyshev polynomials T_0(t) through T_{degree-1}(t)."""
        basis = np.zeros((t.shape[0], degree))
        basis[:, 0] = 1.0
        if degree > 1:
            basis[:, 1] = t
        for k in range(2, degree):
            basis[:, k] = 2.0 * t * basis[:, k-1] - basis[:, k-2]
        return basis
    
    # Use 5 Chebyshev polynomials in x and 5 in v
    # This gives us a 2D basis: T_i(x) * T_j(v) for i,j in range(5)
    # But we only have 10 parameters, so we use a mixed approach:
    # 5 terms in x alone + 5 terms in v alone (plus their interaction if needed)
    
    # Actually, let's use a full 2D Chebyshev basis but limited to 10 terms:
    # T_0(x)T_0(v), T_1(x)T_0(v), T_2(x)T_0(v), T_0(x)T_1(v), T_1(x)T_1(v),
    # T_2(x)T_1(v), T_0(x)T_2(v), T_1(x)T_2(v), T_2(x)T_2(v), T_3(x)T_0(v)
    
    Tx = chebyshev_basis(x_scaled, degree=4)  # T_0 to T_3
    Tv = chebyshev_basis(v_scaled, degree=3)  # T_0 to T_2
    
    # Construct the 10 basis functions
    # p0: T_0(x) * T_0(v)
    # p1: T_1(x) * T_0(v)
    # p2: T_2(x) * T_0(v)
    # p3: T_3(x) * T_0(v)
    # p4: T_0(x) * T_1(v)
    # p5: T_1(x) * T_1(v)
    # p6: T_2(x) * T_1(v)
    # p7: T_0(x) * T_2(v)
    # p8: T_1(x) * T_2(v)
    # p9: T_2(x) * T_2(v)
    
    result = (
        params[0] * Tx[:, 0] * Tv[:, 0] +
        params[1] * Tx[:, 1] * Tv[:, 0] +
        params[2] * Tx[:, 2] * Tv[:, 0] +
        params[3] * Tx[:, 3] * Tv[:, 0] +
        params[4] * Tx[:, 0] * Tv[:, 1] +
        params[5] * Tx[:, 1] * Tv[:, 1] +
        params[6] * Tx[:, 2] * Tv[:, 1] +
        params[7] * Tx[:, 0] * Tv[:, 2] +
        params[8] * Tx[:, 1] * Tv[:, 2] +
        params[9] * Tx[:, 2] * Tv[:, 2]
    )
    
    return result
# EVOLVE-BLOCK-END
