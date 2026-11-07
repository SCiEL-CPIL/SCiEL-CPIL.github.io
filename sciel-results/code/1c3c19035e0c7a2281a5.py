import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
        # Ensure sigma is positive and not zero
        sigma_safe = np.maximum(sigma, 1e-9)
        
        # z-score for Expected Improvement / KG
        z = (f_best - mu) / sigma_safe
        
        # Phi(z) and phi(z)
        phi_z = norm.pdf(z)
        Phi_z = norm.cdf(z)
        
        # Expected Minimum of f_best and f(x)
        # E[min(f_best, f(x))] = mu * Phi(z) + sigma * phi(z)
        # Note: This formula holds for the expectation of the minimum.
        
        expected_min = mu * Phi_z + sigma_safe * phi_z
        
        # Knowledge Gradient
        kg = f_best - expected_min
        
        # To avoid numerical issues and encourage exploration in flat regions,
        # we can add a small multiple of sigma.
        # However, pure KG is a strong baseline.
        # Let's stick to pure KG first, or a slight blend.
        
        # The prompt mentions "hybrid EI + Exploration bonus" worked well.
        # KG is structurally different from EI.
        
        return kg
