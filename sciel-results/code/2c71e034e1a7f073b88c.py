"""Acquisition function for Bayesian optimization of Styblinski-Tang 1D.
Uses a hybrid of Probability of Improvement (PI) and Upper Confidence Bound (UCB)
to balance aggressive exploitation with necessary exploration for multi-modal landscapes.
"""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Structured Bayesian Optimization acquisition using Expected Improvement (EI)
    with adaptive exploration via knowledge gradient approximation.
    
    Uses a hybrid approach:
    - Expected Improvement for direct exploitation
    - Knowledge Gradient surrogate for efficient exploration in multi-modal landscapes
    - Adaptive weighting based on uncertainty concentration
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Posterior mean predictions.
    sigma : ndarray, shape (2000,)
        Posterior standard deviation predictions.
    f_best : float
        Best observed value so far.
        
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores; higher values are preferred.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Numerical stability
    eps = 1e-8
    sigma_safe = np.maximum(sigma, eps)
    
    # 1. Expected Improvement (EI) for minimization
    # EI(x) = (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)  # EI should be non-negative
    
    # 2. Knowledge Gradient (KG) approximation
    # KG approximates the expected reduction in regret from evaluating x
    # For minimization: KG(x) ≈ P(f(x) < f_best) * (f_best - mu(x))
    # This captures both probability and magnitude of improvement
    kg = norm.cdf(z) * np.maximum(f_best - mu, 0.0)
    
    # 3. Uncertainty-aware exploration bonus
    # Add a term that rewards high uncertainty in regions where improvement is plausible
    # This helps escape local minima in multi-modal functions
    # Use a softplus-like function to avoid zeroing out high-sigma regions
    uncertainty_bonus = sigma_safe * np.exp(-0.5 * np.maximum(z, 0.0))
    
    # 4. Adaptive weighting based on current search state
    # When sigma is concentrated (low variance), favor exploitation (EI)
    # When sigma is spread out (high variance), favor exploration (KG + uncertainty)
    sigma_mean = np.mean(sigma_safe)
    sigma_std = np.std(sigma_safe)
    
    # Normalize uncertainty concentration: high when sigma is uniform, low when concentrated
    if sigma_mean > eps:
        concentration = sigma_std / sigma_mean  # Coefficient of variation
    else:
        concentration = 0.0
    
    # Weight EI more when we have high confidence (low concentration)
    # Weight KG and uncertainty more when exploring (high concentration)
    w_ei = 0.7 - 0.3 * min(concentration, 1.0)  # 0.7 to 0.4
    w_kg = 0.2 + 0.2 * min(concentration, 1.0)  # 0.2 to 0.4
    w_unc = 0.1 * min(concentration, 1.0)       # 0.0 to 0.1
    
    # 5. Normalize each component to comparable scale
    def normalize(arr):
        if np.std(arr) < eps:
            return np.zeros_like(arr)
        return (arr - np.mean(arr)) / np.std(arr)
    
    ei_norm = normalize(ei)
    kg_norm = normalize(kg)
    unc_norm = normalize(uncertainty_bonus)
    
    # 6. Combine components
    scores = w_ei * ei_norm + w_kg * kg_norm + w_unc * unc_norm
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
