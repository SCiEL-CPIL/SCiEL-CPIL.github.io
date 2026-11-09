"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model.
    
    Structural change: Use a linear-in-parameters basis function expansion 
    instead of a fixed multiplicative structure. This allows the optimizer 
    (BFGS) to find the best linear combination of biologically-inspired 
    features. The features include:
    - Monod substrate terms (saturating kinetics)
    - Gaussian temperature and pH responses
    - Interaction terms between substrate and environment
    - Linear density inhibition
    
    This representation is more flexible than the previous multiplicative model 
    because it can approximate additive corrections and non-multiplicative 
    interactions more easily, which may be necessary if the true underlying 
    function is not strictly multiplicative.
    
    Parameters (10 used):
    p0: Coefficient for base Monod * Temp * pH
    p1: Coefficient for Monod * Temp (without pH)
    p2: Coefficient for Monod * pH (without Temp)
    p3: Coefficient for Monod alone
    p4: Coefficient for linear substrate * Temp
    p5: Coefficient for linear substrate * pH
    p6: Coefficient for density inhibition (linear in b)
    p7: Coefficient for quadratic density inhibition
    p8: Bias term
    p9: Coefficient for high-order interaction (Monod * Temp * pH * b)
    """
    # Unpack params
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params

    # Pre-compute basis functions
    # 1. Monod kinetics: s / (K + s). We use a fixed K=1 for stability, 
    #    or we can let the optimizer adjust it? 
    #    The task says "up to 10 fitted constants". 
    #    If we use a fixed K, we lose one parameter. 
    #    However, BFGS struggles with non-linear parameter fitting if the 
    #    landscape is too complex. 
    #    Let's stick to linear-in-parameters where possible.
    #    But Monod requires a K. 
    #    Alternative: Use a simple saturating function like 1/(1+exp(-s)) or just s/(1+s).
    #    Let's use s/(1+s) as a robust Monod-like term with K=1.
    #    This frees up parameters for coefficients.
    
    monod = s / (1.0 + s)
    
    # 2. Temperature response: Gaussian centered at 37C (typical for E. coli) 
    #    or let the center be a parameter? 
    #    If we fix the center, we lose flexibility. 
    #    However, fitting a Gaussian center with BFGS can be tricky.
    #    Let's use a polynomial in temp or a fixed Gaussian.
    #    Given the input range [1, 40], a Gaussian centered around 30-40 is good.
    #    Let's use two Gaussians to allow for asymmetric response or just one.
    #    To keep it linear in parameters, we fix the centers and widths.
    #    Center T=37, Width=10
    temp_g1 = np.exp(-((temp - 37.0) ** 2) / (2.0 * 10.0 ** 2))
    #    Maybe a second broader one?
    temp_g2 = np.exp(-((temp - 30.0) ** 2) / (2.0 * 15.0 ** 2))
    
    # 3. pH response: Gaussian centered at 7.0, Width=2.0
    ph_g1 = np.exp(-((pH - 7.0) ** 2) / (2.0 * 2.0 ** 2))
    #    Broader pH term?
    ph_g2 = np.exp(-((pH - 7.0) ** 2) / (2.0 * 4.0 ** 2))
    
    # 4. Density terms
    dens_l = b
    dens_q = b ** 2
    
    # Construct features
    # F0: Main multiplicative core
    F0 = monod * temp_g1 * ph_g1
    
    # F1: Substrate * Temp (ignoring pH)
    F1 = monod * temp_g1
    
    # F2: Substrate * pH (ignoring Temp)
    F2 = monod * ph_g1
    
    # F3: Substrate only
    F3 = monod
    
    # F4: Linear substrate * Temp (for non-saturating part?)
    #    Actually, monod is already saturating. 
    #    Let's add interaction with raw temp/pH to allow for linear drift.
    F4 = s * temp_g1
    F5 = s * ph_g1
    
    # F6: Density inhibition (linear)
    #    Usually growth rate decreases with density.
    F6 = F0 * dens_l
    
    # F7: Density inhibition (quadratic)
    F7 = F0 * dens_q
    
    # F8: Bias
    F8 = np.ones_like(b)
    
    # F9: High-order interaction
    F9 = F0 * dens_l * temp_g2 * ph_g2

    # Combine linearly
    result = (p0 * F0 + 
              p1 * F1 + 
              p2 * F2 + 
              p3 * F3 + 
              p4 * F4 + 
              p5 * F5 + 
              p6 * F6 + 
              p7 * F7 + 
              p8 * F8 + 
              p9 * F9)
    
    # Ensure finite output
    return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
# EVOLVE-BLOCK-END
