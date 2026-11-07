import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model using Monod kinetics, Gaussian temperature/pH responses.
    
    Params:
    p[0]: Amplitude
    p[1]: log(Ks) -> Ks = exp(p1)
    p[2]: Optimal Temp
    p[3]: log(sigma_T) -> sigma_T = exp(p3)
    p[4]: Optimal pH
    p[5]: log(sigma_pH) -> sigma_pH = exp(p5)
    p[6]: Biomass coeff
    p[7]: Offset
    p[8]: Temp skew
    p[9]: pH skew
    """
    # Extract parameters
    A = params[0]
    log_Ks = params[1]
    T_opt = params[2]
    log_sig_T = params[3]
    pH_opt = params[4]
    log_sig_pH = params[5]
    C_b = params[6]
    Offset = params[7]
    skew_T = params[8]
    skew_pH = params[9]
    
    # Ensure stability: if params are extreme, exp might overflow/underflow
    # We use np.exp directly, but clip the final exponent in gaussians
    
    Ks = np.exp(log_Ks)
    sig_T = np.exp(log_sig_T)
    sig_pH = np.exp(log_sig_pH)
    
    # Avoid division by zero if sig is extremely small (exp(-inf) -> 0)
    # If sig is 0, the gaussian becomes a delta function or 0.
    # To be safe, add a tiny epsilon to denominators
    eps = 1e-12
    Ks_safe = Ks + eps
    sig_T_safe = sig_T + eps
    sig_pH_safe = sig_pH + eps
    
    # Monod term
    monod = s / (Ks_safe + s)
    
    # Temperature term
    z_T = (temp - T_opt) / sig_T_safe
    exp_T_arg = -0.5 * z_T**2 + skew_T * z_T
    exp_T_arg = np.clip(exp_T_arg, -500, 500)
    temp_term = np.exp(exp_T_arg)
    
    # pH term
    z_pH = (pH - pH_opt) / sig_pH_safe
    exp_pH_arg = -0.5 * z_pH**2 + skew_pH * z_pH
    exp_pH_arg = np.clip(exp_pH_arg, -500, 500)
    pH_term = np.exp(exp_pH_arg)
    
    # Combine
    pred = A * monod * temp_term * pH_term + C_b * b + Offset
    
    # Ensure finite output
    pred = np.where(np.isfinite(pred), pred, 0.0)
    
    return pred
