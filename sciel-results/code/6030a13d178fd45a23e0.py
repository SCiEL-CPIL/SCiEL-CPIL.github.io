import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization of Rosenbrock 2D.
    
    Uses a hybrid approach combining:
    - Expected Improvement (EI) for exploitation
    - Probability of Improvement (PI) for robustness
    - Cost-aware scaling that penalizes high-cost regions more in exploration
    
    The cost model is 1 + mean(x scaled to [0,1]), ranging from 1.0 to 2.0.
    """
    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Standard normal CDF and PDF
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Expected Improvement for minimization
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Probability of Improvement
    pi = Phi_z
    
    # Cost ranges from 1.0 to 2.0
    # Normalize cost to [0, 1] for easier weighting
    cost_norm = (cost - 1.0)  # ranges from 0 to 1
    
    # Cost-aware scaling:
    # - Exploitation (EI's first term) gets a mild cost penalty
    # - Exploration (EI's second term) gets a stronger cost penalty
    # - PI gets a moderate cost penalty
    
    # Decompose EI into exploitation and exploration
    exploitation = (f_best - mu) * Phi_z
    exploration = sigma_safe * phi_z
    
    # Apply cost penalties with different strengths
    # High-cost regions should be explored less
    exploit_weight = 1.0 / (1.0 + 0.2 * cost_norm)
    explore_weight = 1.0 / (1.0 + 0.8 * cost_norm)
    pi_weight = 1.0 / (1.0 + 0.5 * cost_norm)
    
    # Combine: primarily EI with cost-aware weights, plus a PI component
    # The PI component helps with robustness in short-budget scenarios
    ei_score = exploit_weight * exploitation + explore_weight * exploration
    pi_score = pi_weight * pi
    
    # Blend EI and PI with a fixed ratio
    # EI provides better gradient information, PI is more robust
    blend_ei = 0.7
    blend_pi = 0.3
    
    scores = blend_ei * ei_score + blend_pi * pi_score
    
    # Handle any potential non-finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
