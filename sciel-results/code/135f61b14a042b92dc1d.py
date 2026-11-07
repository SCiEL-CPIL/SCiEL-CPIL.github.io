import numpy as np

def equation(b, s, temp, pH, params):
    """
    Symbolic regression for E. coli growth rate using a universal polynomial basis.
    
    Strategy:
    Instead of fixed biological priors (Monod, Gaussian), we use a complete 
    multivariate polynomial basis up to degree 3. This allows the model to 
    learn arbitrary curvatures and interactions directly from the data.
    
    To ensure numerical stability for BFGS (which starts at all ones), 
    we standardize the inputs to have zero mean and unit variance (approximated 
    based on the provided training ranges). The parameters then scale these 
    standardized features.
    
    Inputs:
    b: [1, 4.998] -> Mean ~3, SD ~1.5
    s: [1, 4.999] -> Mean ~3, SD ~1.5
    temp: [1.002, 39.98] -> Mean ~20.5, SD ~11.5
    pH: [1, 12] -> Mean ~6.5, SD ~3.5
    
    Note: The actual means/SDs in the dataset might differ, but standardizing 
    to roughly [-1, 1] range prevents overflow/underflow in exponentials or 
    high-degree polynomials during the initial BFGS steps.
    """
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9 = params
    
    # Standardize inputs to roughly [-1, 1] range to stabilize polynomial terms
    # Using approximations from the given ranges.
    # b: [1, 5] -> center 3, range 4 -> scale 2
    b_std = (b - 3.0) / 2.0
    # s: [1, 5] -> center 3, range 4 -> scale 2
    s_std = (s - 3.0) / 2.0
    # temp: [1, 40] -> center 20.5, range 39 -> scale 19.5
    t_std = (temp - 20.5) / 19.5
    # pH: [1, 12] -> center 6.5, range 11 -> scale 5.5
    ph_std = (pH - 6.5) / 5.5
    
    # Basis functions:
    # We have 10 parameters. We need to select a subset of polynomial terms.
    # A full degree-3 polynomial in 4 variables has many terms (35 terms).
    # We can only use 10. We will select the most likely relevant terms:
    # 1. Intercept (p0)
    # 2. Linear terms for each variable (p1-p4)
    # 3. Key quadratic terms (p5-p8)
    # 4. One key interaction term (p9)
    
    # Let's define the basis explicitly to map to the 10 params.
    
    # p0: Bias
    term_0 = np.ones_like(b)
    
    # p1: Linear b
    term_1 = b_std
    
    # p2: Linear s
    term_2 = s_std
    
    # p3: Linear temp
    term_3 = t_std
    
    # p4: Linear pH
    term_4 = ph_std
    
    # p5: Quadratic s (Substrate often has saturation/curvature)
    term_5 = s_std ** 2
    
    # p6: Quadratic temp (Temperature often has optimal peak, hence quadratic)
    term_6 = t_std ** 2
    
    # p7: Quadratic pH (pH often has optimal range)
    term_7 = ph_std ** 2
    
    # p8: Interaction s * temp (Substrate availability may modulate temp sensitivity)
    term_8 = s_std * t_std
    
    # p9: Interaction b * s (Density and substrate interaction)
    term_9 = b_std * s_std
    
    # Combine
    y = (p0 * term_0 + 
         p1 * term_1 + 
         p2 * term_2 + 
         p3 * term_3 + 
         p4 * term_4 + 
         p5 * term_5 + 
         p6 * term_6 + 
         p7 * term_7 + 
         p8 * term_8 + 
         p9 * term_9)
         
    return y
