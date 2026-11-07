import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Calculate z-score for standard normal CDF/PDF
    z = (f_best - mu) / sigma_safe
    
    # Calculate Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Calculate UCB for exploration fallback
    # Using a modest beta to avoid excessive exploration
    beta = 1.5
    ucb = -mu + beta * sigma_safe
    
    # Combine EI and a small amount of UCB
    # The UCB term helps if EI is zero (e.g., GP is confident in a suboptimal region)
    # or to encourage exploration of high-uncertainty areas slightly more than pure EI
    combined = ei + 0.1 * ucb
    
    # Cost-awareness
    # The original cost is 1 + mean(x_scaled). Range [1, 2].
    # Dividing by cost directly biases against the center (cost 1.5) vs edges (cost 1.0).
    # We compress the cost range to reduce this bias while maintaining cost-awareness.
    # cost_factor = 1 + alpha * (cost - 1)
    # If alpha = 0.5: cost 1 -> 1, cost 2 -> 1.5. Ratio 1.5 instead of 2.
    # This makes the center less penalized relative to the edges.
    cost_factor = 1.0 + 0.5 * (cost - 1.0)
    cost_factor = np.maximum(cost_factor, 1e-8)
    
    score = combined / cost_factor
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -1e10)
    
    return score
