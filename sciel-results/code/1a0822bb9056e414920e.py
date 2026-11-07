import numpy as np

def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate model.
    
    Structure:
    Multiplicative terms for Substrate (Monod), Temperature (Gaussian), pH (Gaussian).
    Additive quadratic terms for Density (b).
    Offset.
    
    Parameters:
    p[0]: Scale factor
    p[1]: Log(Ks) for Monod
    p[2]: Optimal Temperature
    p[3]: Log(Temperature Width)
    p[4]: Optimal pH
    p[5]: Log(pH Width)
    p[6]: Linear coefficient for b
    p[7]: Quadratic coefficient for b
    p[8]: Interaction coefficient (b * s normalized?)
    p[9]: Offset
    """
    p = params
    
    # Safe exponentials for positive parameters
    Ks = np.exp(p[1])
    sigma_T = np.exp(p[3])
    sigma_pH = np.exp(p[5])
    
    # Monod kinetics for substrate
    monod_s = s / (s + Ks)
    
    # Gaussian for temperature
    gauss_T = np.exp(-((temp - p[2]) ** 2) / (2 * sigma_T ** 2))
    
    # Gaussian for pH
    gauss_pH = np.exp(-((pH - p[4]) ** 2) / (2 * sigma_pH ** 2))
    
    # Density effects (linear and quadratic)
    # b is in [1, 5]
    density_term = p[6] * b + p[7] * b**2
    
    # Interaction: maybe density effect is stronger at high substrate?
    # Let's add a simple interaction term scaled by monod_s
    interaction = p[8] * b * monod_s
    
    # Combine
    # Main growth driven by S, T, pH
    base_growth = p[0] * monod_s * gauss_T * gauss_pH
    
    # Add density corrections
    pred = base_growth + density_term + interaction + p[9]
    
    return pred
