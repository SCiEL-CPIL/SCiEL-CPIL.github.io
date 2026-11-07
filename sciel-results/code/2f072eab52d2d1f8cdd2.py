import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize inputs to [0, 1] based on training ranges
    # b: [1, 5] -> (b-1)/4
    # s: [1, 5] -> (s-1)/4
    # temp: [1, 40] -> (temp-1)/39
    # pH: [2, 12] -> (pH-2)/10
    
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 4.0
    t_norm = (temp - 1.0) / 39.0
    ph_norm = (pH - 2.0) / 10.0
    
    p = params
    
    # Structural Change: Unconstrained 4th-order multivariate polynomial
    # With 10 parameters, we select a subset of monomials to capture key interactions.
    # The full 4th order polynomial in 4 variables has many terms. 
    # We select 10 specific monomials that are likely to be important for biological growth:
    # - Constant
    # - Linear terms for all 4 inputs
    # - Key quadratic and interaction terms
    
    # Let's pick 10 monomials:
    # 1. Constant: 1
    # 2. Linear b: b_norm
    # 3. Linear s: s_norm
    # 4. Linear t: t_norm
    # 5. Linear ph: ph_norm
    # 6. Quadratic t: t_norm^2 (temperature often has optimal range)
    # 7. Quadratic ph: ph_norm^2 (pH often has optimal range)
    # 8. Interaction t*ph: t_norm * ph_norm
    # 9. Interaction t*s: t_norm * s_norm
    # 10. Interaction ph*s: ph_norm * s_norm
    
    # This is a sparse polynomial model.
    # y = p[0] + p[1]*b_norm + p[2]*s_norm + p[3]*t_norm + p[4]*ph_norm + 
    #     p[5]*t_norm^2 + p[6]*ph_norm^2 + p[7]*t_norm*ph_norm + 
    #     p[8]*t_norm*s_norm + p[9]*ph_norm*s_norm
    
    y = (p[0] + 
         p[1] * b_norm + 
         p[2] * s_norm + 
         p[3] * t_norm + 
         p[4] * ph_norm + 
         p[5] * t_norm**2 + 
         p[6] * ph_norm**2 + 
         p[7] * s_norm**2 + 
         p[8] * t_norm * s_norm + 
         p[9] * ph_norm * s_norm)
    
    # Ensure finite output
    if not np.all(np.isfinite(y)):
        return np.zeros_like(y)
        
    return y
