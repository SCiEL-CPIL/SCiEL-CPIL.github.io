import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate using a linear-in-parameters quadratic polynomial model.
    
    Strategy:
    - The previous approach used non-linear parameters (centers of Gaussians), 
      creating a non-convex optimization landscape for BFGS.
    - This approach uses a fixed basis of monomials (main effects and interactions).
    - The model is linear in the parameters (coefficients), making the optimization 
      problem convex. BFGS will reliably find the global minimum for the coefficients.
    - Inputs are normalized to [0, 1] to improve numerical conditioning.
    
    Parameters (10 total):
    - params[0]: Intercept (bias)
    - params[1:5]: Linear coefficients for b, s, temp, pH
    - params[5:10]: Quadratic/Interaction coefficients for the most relevant pairs:
      - params[5]: temp^2 (non-linearity in temperature is crucial)
      - params[6]: pH^2 (non-linearity in pH)
      - params[7]: temp * pH (interaction)
      - params[8]: s * temp (nutrient-temperature interaction)
      - params[9]: b * temp (biomass-temperature interaction, or just b^2 if b is less critical)
      
    Note: We prioritize temperature interactions as it is the strongest driver.
    We drop b^2 and s^2 and b*s to make room for more critical temp/pH terms.
    Actually, let's include the most standard quadratic terms for the 4 variables 
    and the most critical interactions.
    
    Basis functions:
    1. 1
    2. b_norm
    3. s_norm
    4. t_norm
    5. p_norm
    6. t_norm^2
    7. p_norm^2
    8. t_norm * p_norm
    9. s_norm * t_norm
    10. b_norm * t_norm
    
    This covers all main effects, the two most important non-linearities (T, pH), 
    and three key interactions.
    """
    # Normalize inputs to [0, 1] based on known training ranges
    # b: [1, 4.998] -> approx [0, 1]
    # s: [1, 4.999] -> approx [0, 1]
    # temp: [1, 39.98] -> approx [0, 1]
    # pH: [2, 12] -> approx [0, 1]
    
    b_n = (b - 1.0) / 4.0
    s_n = (s - 1.0) / 4.0
    t_n = (temp - 1.0) / 39.0
    p_n = (pH - 2.0) / 10.0
    
    # Clip to [0, 1] to handle slight out-of-range values gracefully during optimization
    # though training is in-range, this prevents numerical issues if optimizer pushes boundaries
    b_n = np.clip(b_n, 0.0, 1.0)
    s_n = np.clip(s_n, 0.0, 1.0)
    t_n = np.clip(t_n, 0.0, 1.0)
    p_n = np.clip(p_n, 0.0, 1.0)
    
    # Extract parameters
    c0 = params[0]
    c1 = params[1]
    c2 = params[2]
    c3 = params[3]
    c4 = params[4]
    c5 = params[5]
    c6 = params[6]
    c7 = params[7]
    c8 = params[8]
    c9 = params[9]
    
    # Compute linear combination
    y = (c0 + 
         c1 * b_n + 
         c2 * s_n + 
         c3 * t_n + 
         c4 * p_n + 
         c5 * (t_n * t_n) + 
         c6 * (p_n * p_n) + 
         c7 * (t_n * p_n) + 
         c8 * (s_n * t_n) + 
         c9 * (b_n * t_n))
         
    return y
