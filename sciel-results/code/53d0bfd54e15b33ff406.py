"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(x: np.ndarray, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Recovers a function using a hybrid basis of polynomials and Ridgelets (Gaussians).
    
    The model is linear in the 10 parameters:
    f(x, y) = p0*1 + p1*x + p2*y + p3*(x^2-0.5) + p4*(y^2-0.5) + p5*xy +
              p6*G(x,y; 0.5,0.5) + p7*G(x,y; -0.5,-0.5) +
              p8*G(x,y; 0.5,-0.5) + p9*G(x,y; -0.5,0.5)
    
    where G(x,y; cx, cy) = exp(-2.0*((x-cx)^2 + (y-cy)^2)).
    
    This basis combines global polynomial trends with local Gaussian bumps,
    allowing it to fit a wider variety of smooth functions than a pure polynomial.
    """
    # Ensure inputs are 1D arrays for consistent broadcasting
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    
    # Polynomial basis terms
    phi0 = np.ones_like(x)
    phi1 = x
    phi2 = y
    phi3 = x**2 - 0.5
    phi4 = y**2 - 0.5
    phi5 = x * y
    
    # Ridgelet (Gaussian) basis terms
    # Width parameter alpha = 2.0
    alpha = 2.0
    
    # Center (0.5, 0.5)
    dx = x - 0.5
    dy = y - 0.5
    phi6 = np.exp(-alpha * (dx**2 + dy**2))
    
    # Center (-0.5, -0.5)
    dx = x + 0.5
    dy = y + 0.5
    phi7 = np.exp(-alpha * (dx**2 + dy**2))
    
    # Center (0.5, -0.5)
    dx = x - 0.5
    dy = y + 0.5
    phi8 = np.exp(-alpha * (dx**2 + dy**2))
    
    # Center (-0.5, 0.5)
    dx = x + 0.5
    dy = y - 0.5
    phi9 = np.exp(-alpha * (dx**2 + dy**2))
    
    # Stack features into a matrix (N, 10)
    Phi = np.stack((phi0, phi1, phi2, phi3, phi4, phi5, phi6, phi7, phi8, phi9), axis=1)
    
    # Compute predictions: Phi (N, 10) @ params (10,) -> (N,)
    pred = Phi @ params
    
    # Ensure finite values (safety)
    if not np.all(np.isfinite(pred)):
        # Fallback to simple linear if something went wrong
        return params[0] * x + params[1] * y + params[2]
        
    return pred
# EVOLVE-BLOCK-END
