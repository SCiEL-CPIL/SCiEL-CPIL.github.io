"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Uses a linear-in-parameters model based on a polynomial expansion
    of strain with temperature interactions. This allows BFGS to find
    the global optimum efficiently by solving a linear least-squares problem.
    
    params (10 constants):
    p0: Intercept
    p1: Linear strain coefficient
    p2: Linear strain * temp coefficient
    p3: Quadratic strain coefficient
    p4: Quadratic strain * temp coefficient
    p5: Cubic strain coefficient
    p6: Cubic strain * temp coefficient
    p7: Quartic strain coefficient
    p8: Quartic strain * temp coefficient
    p9: Quintic strain coefficient
    """
    p0 = params[0]
    p1 = params[1]
    p2 = params[2]
    p3 = params[3]
    p4 = params[4]
    p5 = params[5]
    p6 = params[6]
    p7 = params[7]
    p8 = params[8]
    p9 = params[9]
    
    e = strain
    t = temp
    
    # Pre-compute powers of strain and temperature interactions
    e2 = e * e
    e3 = e2 * e
    e4 = e2 * e2
    e5 = e4 * e
    
    # Basis functions
    phi0 = np.ones_like(e)
    phi1 = e
    phi2 = e * t
    phi3 = e2
    phi4 = e2 * t
    phi5 = e3
    phi6 = e3 * t
    phi7 = e4
    phi8 = e4 * t
    phi9 = e5
    
    # Linear combination
    stress = (p0 * phi0 + p1 * phi1 + p2 * phi2 + p3 * phi3 + 
              p4 * phi4 + p5 * phi5 + p6 * phi6 + p7 * phi7 + 
              p8 * phi8 + p9 * phi9)
              
    return stress
# EVOLVE-BLOCK-END
