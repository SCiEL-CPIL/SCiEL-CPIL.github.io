import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Knowledge Gradient (KG) acquisition function for minimization.
    
    KG approximates the expected reduction in the minimum value (f_best)
    after evaluating at each candidate point. It provides a more theoretically
    optimal balance between exploration and exploitation than simple EI or UCB.
    
    For a GP with posterior N(mu, sigma^2), the KG at point x is:
    E[max(0, f_best - f(x))] where f(x) ~ N(mu(x), sigma(x)^2)
    
    This equals:
    (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    where Phi is the standard normal CDF and phi is the standard normal PDF.
    """
    # Ensure sigma is non-negative and handle numerical stability
    sigma = np.maximum(sigma, 1e-8)
    
    # Compute z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Clip z to avoid extreme values in CDF/PDF calculations
    z = np.clip(z, -10, 10)
    
    # Standard normal CDF and PDF
    from scipy.stats import norm
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # KG value: expected improvement over f_best
    # E[max(0, f_best - f(x))] = (f_best - mu) * Phi(z) + sigma * phi(z)
    kg_values = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure non-negative values (numerical stability)
    kg_values = np.maximum(kg_values, 0.0)
    
    # Higher KG means more expected reduction in f_best, so we maximize this
    return kg_values
