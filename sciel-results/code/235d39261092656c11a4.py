"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using an Additive Polynomial Basis (GAM-like) approach.
    
    This model assumes the growth rate is a linear combination of basis functions
    of the inputs. This makes the problem linear in parameters, which provides a
    smooth, convex-like optimization landscape for BFGS, avoiding local minima
    often found in multiplicative mechanistic models.
    
    Inputs are scaled to zero mean and unit variance (approx) to improve
    numerical stability during optimization.
    
    Parameters (10 total):
    p[0]: Intercept
    p[1]: Coeff for biomass (scaled)
    p[2]: Coeff for substrate (scaled)
    p[3]: Coeff for temp (scaled)
    p[4]: Coeff for pH (scaled)
    p[5]: Coeff for biomass^2
    p[6]: Coeff for substrate^2
    p[7]: Coeff for temp^2
    p[8]: Coeff for pH^2
    p[9]: Coeff for temp*pH interaction
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Scale inputs to improve conditioning
    # b range ~ [1, 5], center 3, scale 2
    x_b = (b - 3.0) / 2.0
    # s range ~ [1, 5], center 3, scale 2
    x_s = (s - 3.0) / 2.0
    # temp range ~ [1, 40], center 20, scale 10
    x_t = (temp - 20.0) / 10.0
    # pH range ~ [2, 12], center 7, scale 5
    x_p = (pH - 7.0) / 5.0
    
    # Basis functions
    # Linear terms
    term_b = x_b
    term_s = x_s
    term_t = x_t
    term_p = x_p
    
    # Quadratic terms
    term_b2 = x_b * x_b
    term_s2 = x_s * x_s
    term_t2 = x_t * x_t
    term_p2 = x_p * x_p
    
    # Interaction term (Temp and pH often interact in biological systems)
    term_tp = x_t * x_p
    
    # Combine
    # Note: This is a linear model in parameters, so BFGS will converge to the
    # global minimum for these parameters given the fixed basis functions.
    result = (p0 + 
              p1 * term_b + 
              p2 * term_s + 
              p3 * term_t + 
              p4 * term_p + 
              p5 * term_b2 + 
              p6 * term_s2 + 
              p7 * term_t2 + 
              p8 * term_p2 + 
              p9 * term_tp)
              
    return result
# EVOLVE-BLOCK-END
