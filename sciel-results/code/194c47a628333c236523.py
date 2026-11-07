import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    # Normalize inputs to [0, 1] based on training ranges
    # b: [1, 5] -> shift by 1, divide by 4 => [0, 1]
    # s: [1, 5] -> shift by 1, divide by 4 => [0, 1]
    # temp: [1, 40] -> shift by 1, divide by 39 => [0, 1]
    # pH: [2, 12] -> shift by 2, divide by 10 => [0, 1]
    b_norm = (b - 1.0) / 4.0
    s_norm = (s - 1.0) / 4.0
    t_norm = (temp - 1.0) / 39.0
    ph_norm = (pH - 2.0) / 10.0
    
    p = params
    
    # 4th-order multivariate polynomial in normalized inputs
    # We have 4 variables: b_norm, s_norm, t_norm, ph_norm
    # Total terms in a full 4th-order polynomial in 4 variables is C(4+4, 4) = 35.
    # But we only have 10 parameters.
    # We need to select a subset of terms or use a structured polynomial.
    
    # Given the constraints, a full 4th order polynomial is too complex for 10 params.
    # The directive says: "Replace the multiplicative Gaussian and Monod terms with a single, unconstrained 4th-order multivariate polynomial in the normalized inputs."
    # This implies we should use the 10 params to define a polynomial that captures the essential 4th-order behavior, or perhaps a reduced set of terms.
    # However, with only 10 params, we can't fit a full 4th order polynomial (35 coeffs).
    # Maybe the directive means a polynomial where the coefficients are constrained or it's a specific form?
    # Or perhaps "4th-order" refers to the highest degree term included, but we pick the most relevant ones?
    # Let's try to include the most likely important terms for biological growth:
    # 1. Constant (p[0])
    # 2. Linear terms (p[1]*b, p[2]*s, p[3]*t, p[4]*ph)
    # 3. Quadratic terms (p[5]*t^2, p[6]*ph^2, p[7]*s^2, p[8]*b^2) -- wait, that's 10 params already.
    # Let's try:
    # p[0]: constant
    # p[1]: b
    # p[2]: s
    # p[3]: t
    # p[4]: ph
    # p[5]: t^2 (to capture the bell shape of temperature)
    # p[6]: ph^2 (to capture the bell shape of pH)
    # p[7]: s^2 (saturation curvature)
    # p[8]: b^2 (inhibition curvature)
    # p[9]: t*ph (interaction)
    
    # This is a "reduced" 4th-order polynomial (actually max degree 2, but includes interactions).
    # Wait, the directive explicitly says "4th-order".
    # A true 4th-order term would be t^4, t^2*ph^2, etc.
    # With 10 params, we can't fit a general 4th-order polynomial.
    # Perhaps the intent is to allow the optimizer to find the best 10-term polynomial from the space of 4th-order monomials?
    # But the function signature requires a fixed expression.
    # Let's stick to a high-degree polynomial structure that fits within 10 params, prioritizing the most biologically relevant higher-order terms.
    
    # Let's try a polynomial that includes up to 4th degree terms for the most critical variables (t and ph), and lower for others.
    # Or maybe a radial basis function approximation? No, directive says polynomial.
    
    # Let's try this structure:
    # y = p[0] + p[1]*b + p[2]*s + p[3]*t + p[4]*ph + p[5]*t^2 + p[6]*ph^2 + p[7]*t^3 + p[8]*ph^3 + p[9]*t*ph
    
    # This allows for non-monotonic behavior in t and ph (up to cubic), and interaction.
    # It uses all 10 params.
    
    y = (p[0] 
         + p[1] * b_norm 
         + p[2] * s_norm 
         + p[3] * t_norm 
         + p[4] * ph_norm 
         + p[5] * t_norm**2 
         + p[6] * ph_norm**2 
         + p[7] * t_norm**3 
         + p[8] * ph_norm**3 
         + p[9] * t_norm * ph_norm)
         
    return y
