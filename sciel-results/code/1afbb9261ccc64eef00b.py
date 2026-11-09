# EVOLVE-BLOCK-START
import numpy as np
import time


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model combining Monod kinetics with a flexible
    Fourier (sum-of-sines) basis for temperature and pH, plus a quadratic
    polynomial component to capture curvature and interactions.

    This hybrid approach uses the biologically-motivated Monod term for
    substrate saturation. It combines:
    1. A linear combination of sinusoidal features (Fourier basis) for 
       temperature and pH to capture periodic/optimal ranges.
    2. A quadratic polynomial basis for temperature, pH, and their interaction
       to capture smooth curvature (bell-shaped responses) that might be 
       better represented by polynomials than low-order Fourier terms in 
       specific regimes, or to provide redundancy for robust fitting.

    The structure is:
    mu = p1 * monod + p2 * (quadratic_terms) + p3 * (fourier_terms) + p4

    To keep parameter count at 10 and allow BFGS to converge:
    - p0: Ks (Monod half-saturation, used as exp(p0))
    - p1: Scale for Monod term
    - p2: Scale for Quadratic terms
    - p3: Scale for Fourier terms
    - p4: Bias
    - p5, p6: Quadratic coefficients for temp, pH (temp^2, pH^2)
    - p7: Quadratic interaction coefficient (temp*pH)
    - p8, p9: Fourier amplitudes for 1st harmonic of temp (sin, cos)
    
    Note: We sacrifice some Fourier flexibility (only 1st harmonic for temp, 
    none for pH in this specific allocation) to include the quadratic terms 
    which are often very effective for single-peaked biological responses 
    within a bounded domain. Alternatively, we can mix them more directly.

    Let's refine the parameter allocation to be more balanced:
    p0: Ks (exp)
    p1: Monod scale
    p2: Bias
    p3, p4: Fourier temp (sin, cos) 1st harmonic
    p5, p6: Fourier pH (sin, cos) 1st harmonic
    p7, p8: Quadratic temp, pH (normalized temp^2, pH^2)
    p9: Quadratic interaction (temp*pH)
    
    This allows the model to choose between periodic and polynomial curvature.
    """
    p = params

    # Normalize inputs to [0, 1] range for stability
    # b: [1, 5]
    b_norm = (b - 1.0) / 4.0
    # s: [1, 40]
    s_norm = (s - 1.0) / 39.0
    # temp: [1, 40]
    temp_norm = (temp - 1.0) / 39.0
    # pH: [1, 14]
    ph_norm = (pH - 1.0) / 13.0

    # Monod term for substrate
    Ks = np.exp(p[0])
    monod = s / (Ks + s)

    # Fourier basis for temperature (1st harmonic)
    sin_temp_1 = np.sin(2.0 * np.pi * temp_norm)
    cos_temp_1 = np.cos(2.0 * np.pi * temp_norm)

    # Fourier basis for pH (1st harmonic)
    sin_ph_1 = np.sin(2.0 * np.pi * ph_norm)
    cos_ph_1 = np.cos(2.0 * np.pi * ph_norm)

    # Quadratic basis (using normalized inputs to keep magnitudes comparable)
    # Centering helps with numerical stability and conditioning
    # temp_centered = temp_norm - 0.5
    # ph_centered = ph_norm - 0.5
    # For simplicity and direct mapping to "curvature", we use raw normalized squares
    # but scaled to be O(1).
    quad_temp = temp_norm * temp_norm
    quad_ph = ph_norm * ph_norm
    quad_inter = temp_norm * ph_norm

    # Linear combination
    # The model is linear in p1..p9 given p0.
    # This convexity in the linear part helps BFGS.
    
    # Term 1: Substrate-limited growth
    base_growth = p[1] * monod
    
    # Term 2: Environmental effects (Fourier + Quadratic)
    env_fourier = p[3] * sin_temp_1 + p[4] * cos_temp_1 + p[5] * sin_ph_1 + p[6] * cos_ph_1
    env_quad = p[7] * quad_temp + p[8] * quad_ph + p[9] * quad_inter
    
    # Total prediction
    result = base_growth + env_fourier + env_quad + p[2]

    # Ensure the output is finite
    return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
# EVOLVE-BLOCK-END
