import numpy as np

def equation(b, s, temp, pH, params):
    """
    E. coli growth rate model using a Biologically-Inspired Multiplicative Model.
    
    Strategy:
    Instead of a generic additive basis expansion, we use a structured model
    that combines:
    1. Monod kinetics for substrate (s): mu_s = s / (Ks + s)
    2. Arrhenius/Quadratic term for temperature (temp): Gaussian peak around optimal temp
    3. Hill function for density (b): Inhibition or saturation depending on exponent
    4. Logistic/Gaussian term for pH: Gaussian peak around optimal pH
    
    This structure reduces the hypothesis space by enforcing known physical/biological
    constraints (e.g., growth should decrease at extreme temps/pH, saturate at high substrate).
    
    Parameters (10 total):
    p[0]: Base growth rate multiplier (mu_max)
    p[1]: Substrate saturation constant (Ks)
    p[2]: Temperature optimal center (T_opt)
    p[3]: Temperature width (sigma_T)
    p[4]: pH optimal center (pH_opt)
    p[5]: pH width (sigma_pH)
    p[6]: Density inhibition constant (Kb)
    p[7]: Density inhibition exponent (n_b)
    p[8]: Substrate activation exponent (n_s) - usually 1, but flexible
    p[9]: Global bias/offset (additive)
    
    Model:
    mu = p[0] * (s^n_s / (p[1] + s^n_s)) * 
         exp(-((temp - p[2])^2) / (2 * p[3]^2)) * 
         exp(-((pH - p[4])^2) / (2 * p[5]^2)) * 
         (b^n_b / (p[6] + b^n_b)) + p[9]
         
    Note: We use softplus or exp to ensure positive parameters where needed, 
    or rely on BFGS to stay in valid regions if initialized well. 
    However, since BFGS starts at 1, we must ensure the function is smooth 
    and defined for all real inputs.
    
    To handle potential negative parameters from BFGS (which can happen if 
    landscape is flat or if it explores negative space), we use:
    - Softplus for widths and constants that must be positive: sigma = log(1+exp(x))
    - Direct use for centers (T_opt, pH_opt) as they can be any real number.
    - For exponents, we use softplus to keep them positive (typically > 0).
    """
    
    p = params
    
    # Unpack parameters with softplus for positive constraints
    # p[0] mu_max: should be positive. Softplus.
    mu_max = np.log1p(np.exp(p[0]))
    
    # p[1] Ks: should be positive. Softplus.
    Ks = np.log1p(np.exp(p[1]))
    
    # p[2] T_opt: center, can be any real. Direct.
    T_opt = p[2]
    
    # p[3] sigma_T: width, must be positive. Softplus.
    sigma_T = np.log1p(np.exp(p[3]))
    
    # p[4] pH_opt: center, can be any real. Direct.
    pH_opt = p[4]
    
    # p[5] sigma_pH: width, must be positive. Softplus.
    sigma_pH = np.log1p(np.exp(p[5]))
    
    # p[6] Kb: inhibition constant, must be positive. Softplus.
    Kb = np.log1p(np.exp(p[6]))
    
    # p[7] n_b: density exponent, must be positive. Softplus.
    n_b = np.log1p(np.exp(p[7]))
    
    # p[8] n_s: substrate exponent, must be positive. Softplus.
    n_s = np.log1p(np.exp(p[8]))
    
    # p[9] bias: additive offset. Direct.
    bias = p[9]
    
    # Calculate components
    
    # 1. Substrate term (Monod-like)
    # s is in [1, 5]. Ks is positive.
    # Avoid division by zero or negative base if n_s is fractional and s is negative (s is positive here).
    # s^n_s is safe for s>0.
    term_s = (s ** n_s) / (Ks + (s ** n_s))
    
    # 2. Temperature term (Gaussian)
    # temp is in [1, 40].
    # Ensure sigma_T is not too small to avoid numerical overflow in exp.
    # Softplus(1) ~ 1.3, so sigma_T >= ~0.69 if p[3] is very negative? No, softplus(0)=0.69.
    # If p[3] -> -inf, softplus -> 0. We should clamp or add a small epsilon.
    # Let's add a small epsilon to sigma to prevent division by zero in exp.
    sigma_T_safe = sigma_T + 1e-6
    term_t = np.exp(-((temp - T_opt) ** 2) / (2 * sigma_T_safe ** 2))
    
    # 3. pH term (Gaussian)
    # pH is in [2, 12].
    sigma_pH_safe = sigma_pH + 1e-6
    term_ph = np.exp(-((pH - pH_opt) ** 2) / (2 * sigma_pH_safe ** 2))
    
    # 4. Density term (Hill function for inhibition/saturation)
    # b is in [1, 5].
    # If n_b is large, this becomes a step function.
    term_b = (b ** n_b) / (Kb + (b ** n_b))
    
    # Combine
    # Multiplicative structure for the main growth drivers
    growth = mu_max * term_s * term_t * term_ph * term_b
    
    # Add bias
    result = growth + bias
    
    # Ensure output is finite
    # The inputs are bounded, and softplus keeps denominators positive.
    # The exponents are positive.
    # The only risk is overflow in exp if sigma is tiny and temp far from T_opt.
    # exp(-large) -> 0, which is safe.
    # exp(-small) -> 1, safe.
    
    return result
