import numpy as np


def equation(b: np.ndarray, s: np.ndarray, temp: np.ndarray, pH: np.ndarray, params: np.ndarray) -> np.ndarray:
    """
    E. coli growth rate using a smooth additive model with softplus activations.
    This allows for more flexible interactions and offsets compared to strict multiplicative forms.
    """
    # Unpack parameters
    c0, Ks, T_opt, sigma_T, pH_opt, sigma_pH, alpha_b, beta_b, gamma, delta = params
    
    # Safety margins to prevent overflow or division by zero
    Ks_safe = np.maximum(np.abs(Ks), 1e-6)
    sigma_T_safe = np.maximum(np.abs(sigma_T), 0.1)
    sigma_pH_safe = np.maximum(np.abs(sigma_pH), 0.1)
    beta_b_safe = np.maximum(np.abs(beta_b), 1e-6)
    
    # 1. Substrate: Smooth saturation (Monod-like but smooth everywhere)
    # Using softplus to ensure smoothness and positive contribution
    # monod = s / (Ks + s) is standard, but we can make it slightly more flexible with a power or just keep it.
    # Let's stick to a robust Monod but ensure it's well-behaved.
    monod = s / (Ks_safe + s)
    
    # 2. Temperature: Gaussian peak
    temp_factor = np.exp(-((temp - T_opt) ** 2) / (2.0 * sigma_T_safe ** 2))
    
    # 3. pH: Gaussian peak
    ph_factor = np.exp(-((pH - pH_opt) ** 2) / (2.0 * sigma_pH_safe ** 2))
    
    # 4. Density: Smooth inhibition
    # Instead of 1/(1+(b/beta)^alpha), use a softplus-based decay or similar
    # This allows for a smoother transition and potentially better fitting of the inhibition curve.
    density_inhibition = np.logaddexp(0.0, (b / beta_b_safe) ** alpha_b)
    # Invert to get a factor that decreases with b
    # To keep it bounded and positive, we can use exp(-density_inhibition) or similar
    # Let's use a simple exponential decay of the softplus term to ensure it stays positive and smooth
    density_factor = np.exp(-density_inhibition)
    
    # Combine additively with softplus to allow for non-linear interactions and offsets
    # The idea is that the total growth is not just a product, but a smooth function of the sum of contributions.
    # This can capture cases where one factor might compensate for another.
    
    # Linear combination of the factors
    linear_combo = (
        c0 * monod * temp_factor * ph_factor * density_factor + 
        gamma * monod + 
        delta
    )
    
    # Apply a final softplus to ensure the output is smooth and potentially bounded away from negative infinity
    # However, growth rate can be negative (death rate), so a simple softplus might not be ideal if it forces positivity.
    # The current model allows negative growth via the delta term if the product is small.
    # Let's stick to the additive structure but ensure smoothness.
    
    return linear_combo
