"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def equation(strain: np.ndarray, temp: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    Stress of an aluminium rod from strain and temperature.
    
    Strategy: Piecewise-Linear (Spline) Basis Model.
    This approach uses a set of "hinge" functions (max(0, x - knot)) to create a 
    piecewise-linear approximation of the stress-strain curve. This is highly 
    effective for materials that have distinct elastic and plastic regions, 
    as it can perfectly fit a piecewise-linear curve with the right number of knots.
    
    The model is linear in the coefficients (slopes), which helps BFGS converge 
    reliably, while the knots (positions) are nonlinear parameters.
    
    Inputs:
    strain: [-0.001911, 0.9636]
    temp:   [0.06667, 1]
    
    Parameters (10 total):
    p[0]: Base slope (elastic modulus approximation)
    p[1]: Temperature-dependent slope shift
    p[2]: Intercept (offset)
    p[3]: Temperature intercept shift
    p[4]: Knot 1 position (strain)
    p[5]: Slope change at Knot 1
    p[6]: Knot 2 position (strain)
    p[7]: Slope change at Knot 2
    p[8]: Knot 3 position (strain)
    p[9]: Slope change at Knot 3
    
    We also include a simple linear temperature effect on the overall stress to 
    account for thermal expansion/contraction or temperature-dependent yield, 
    but the primary complexity is in the strain response via the spline.
    """
    # Extract parameters
    base_slope = params[0]
    temp_slope = params[1]
    intercept = params[2]
    temp_intercept = params[3]
    
    k1 = params[4]
    s1 = params[5]
    k2 = params[6]
    s2 = params[7]
    k3 = params[8]
    s3 = params[9]
    
    # Center temperature for numerical stability (optional, but good practice)
    T_c = temp - 0.5
    
    # Linear part: Stress = (E0 + E_T * T) * Strain + (C0 + C_T * T)
    # This captures the general linear trend and temperature dependence
    linear_stress = (base_slope + temp_slope * T_c) * strain + (intercept + temp_intercept * T_c)
    
    # Spline part: Sum of hinge functions
    # h(x, k) = max(0, x - k)
    # This adds a "kink" at k with slope change s
    # We apply this to strain to model the nonlinearity in the stress-strain curve
    
    # Using np.maximum for vectorized hinge function
    h1 = np.maximum(0.0, strain - k1)
    h2 = np.maximum(0.0, strain - k2)
    h3 = np.maximum(0.0, strain - k3)
    
    # Add the spline contributions
    spline_stress = s1 * h1 + s2 * h2 + s3 * h3
    
    # Total stress
    stress = linear_stress + spline_stress
    
    return stress
# EVOLVE-BLOCK-END
