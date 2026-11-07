import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate modeled with a multiplicative mechanistic model:
    - Monod saturation kinetics for substrate
    - Gaussian (bell-shaped) response for temperature
    - Gaussian (bell-shaped) response for pH
    - Density-dependent inhibition (logistic-like)
    
    params: [mu_max, Ks, T_opt, sigma_T, pH_opt, sigma_pH, alpha_b, beta_b, gamma, delta]
    """
    mu_max = params[0]
    Ks = params[1]
    T_opt = params[2]
    sigma_T = params[3]
    pH_opt = params[4]
    sigma_pH = params[5]
    alpha_b = params[6]
    beta_b = params[7]
    gamma = params[8]
    delta = params[9]
    
    # Ensure numerical stability for denominators
    Ks_safe = np.maximum(np.abs(Ks), 1e-4)
    sigma_T_safe = np.maximum(np.abs(sigma_T), 1e-4)
    sigma_pH_safe = np.maximum(np.abs(sigma_pH), 1e-4)
    beta_b_safe = np.maximum(np.abs(beta_b), 1e-4)
    
    # Monod saturation: s / (Ks + s)
    monod_s = s / (Ks_safe + s)
    
    # Gaussian temperature response: exp(-((T - T_opt)^2) / (2 * sigma_T^2))
    T_resid = temp - T_opt
    gauss_T = np.exp(-(T_resid ** 2) / (2.0 * sigma_T_safe ** 2))
    
    # Gaussian pH response: exp(-((pH - pH_opt)^2) / (2 * sigma_pH^2))
    pH_resid = pH - pH_opt
    gauss_pH = np.exp(-(pH_resid ** 2) / (2.0 * sigma_pH_safe ** 2))
    
    # Density-dependent inhibition: 1 / (1 + alpha_b * b / beta_b)
    # This gives a saturating decrease in growth rate with increasing density
    density_factor = 1.0 / (1.0 + alpha_b * b / beta_b_safe)
    
    # Cross-interaction term between substrate and temperature
    cross_term = gamma * monod_s * gauss_T
    
    # Combine multiplicatively
    y = mu_max * monod_s * gauss_T * gauss_pH * density_factor + cross_term + delta
    
    return y
