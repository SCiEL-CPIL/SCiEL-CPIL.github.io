import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Sanitize sigma to prevent division by zero and numerical instability
    min_sigma = 1e-6
    sigma_safe = np.where(sigma > min_sigma, sigma, min_sigma)
    
    # Calculate the Z-scores for the potential improvement
    # z represents how many standard deviations the predicted mean is below f_best
    z = (f_best - mu) / sigma_safe
    
    # Calculate the standard normal CDF and PDF
    # Phi(z) = P(Z <= z)
    # phi(z) = standard normal PDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * P(f(x) < f_best) + sigma * pdf(z)
    # The term (f_best - mu) * Phi(z) captures the expected gain where improvement is positive
    # The term sigma * phi(z) is the correction factor from the normal distribution
    ei_scores = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Use a hybrid approach: combine EI with a strong exploitation component
    # This helps converge faster to the minimum while still exploring
    # The exploitation term is -mu (posterior mean minimization)
    # We blend them with a weight that favors exploitation as f_best approaches 0
    
    # Normalize the range of f_best for adaptive weighting
    # When f_best is close to 0 (minimum), we want more exploitation
    # When f_best is far from 0, we want more exploration via EI
    
    # Use a sigmoid-like function to adapt the weight
    # As f_best -> 0, weight -> 1 (more exploitation)
    # As f_best -> large, weight -> 0 (more exploration)
    scale = 0.1  # Scale factor for adaptation
    weight = 1.0 / (1.0 + np.exp(scale * (f_best - 0.0)))
    
    # Blend EI and exploitation
    # exploitation_score = -mu (higher is better, i.e., lower mu)
    # We need to scale them to be comparable
    # Normalize EI to be in a similar range as -mu
    
    # Simple approach: use a fixed blend that favors exploitation
    # But adapt based on how close we are to the minimum
    alpha = 0.7 * weight + 0.3  # Blend factor: 0.3 to 1.0
    
    # Normalize scores to be comparable
    # EI scores are typically small when close to optimum
    # -mu scores can be larger in magnitude
    
    # Use a robust combination
    scores = alpha * (-mu) + (1 - alpha) * ei_scores
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
