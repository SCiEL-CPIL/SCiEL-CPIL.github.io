"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(x: np.ndarray, y: np.ndarray, z: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Polynomial basis model using normalized inputs and shifted Legendre polynomials.
    
    This model uses a linear combination of 10 basis functions constructed from
    normalized inputs (mapped from [0.5, 2] to [-1, 1]) and Legendre polynomials
    up to degree 3. This provides an orthogonal, well-conditioned representation
    that captures higher-order curvature and interactions efficiently.
    """
    # Ensure inputs are float arrays
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    p = np.asarray(params, dtype=np.float64)
    
    # Normalize inputs from [0.5, 2.0] to [-1, 1]
    # u = (val - 1.25) / 0.75
    u = (x - 1.25) / 0.75
    v = (y - 1.25) / 0.75
    w = (z - 1.25) / 0.75
    
    # Compute Legendre polynomials P0 to P3 for each normalized variable
    # P0(t) = 1
    # P1(t) = t
    # P2(t) = (3t^2 - 1)/2
    # P3(t) = (5t^3 - 3t)/2
    
    u0 = np.ones_like(u)
    u1 = u
    u2 = 0.5 * (3.0 * u**2 - 1.0)
    u3 = 0.5 * (5.0 * u**3 - 3.0 * u)
    
    v0 = np.ones_like(v)
    v1 = v
    v2 = 0.5 * (3.0 * v**2 - 1.0)
    v3 = 0.5 * (5.0 * v**3 - 3.0 * v)
    
    w0 = np.ones_like(w)
    w1 = w
    w2 = 0.5 * (3.0 * w**2 - 1.0)
    w3 = 0.5 * (5.0 * w**3 - 3.0 * w)
    
    # Construct 10 basis functions:
    # 1. Constant: 1
    # 2. Linear x: u1
    # 3. Linear y: v1
    # 4. Linear z: w1
    # 5. Quadratic x: u2
    # 6. Quadratic y: v2
    # 7. Quadratic z: w2
    # 8. Cubic x: u3
    # 9. Cubic y: v3
    # 10. Cubic z: w3
    
    # Note: This choice focuses on individual variable nonlinearities.
    # If interactions are needed, we could replace some with cross terms,
    # but pure polynomials often work well for smooth functions.
    # Let's stick to individual terms for simplicity and conditioning.
    
    basis = np.column_stack((
        u0,
        u1,
        v1,
        w1,
        u2,
        v2,
        w2,
        u3,
        v3,
        w3
    ))
    
    # Prediction is linear combination
    pred = basis @ p
    
    return pred
# EVOLVE-BLOCK-END
