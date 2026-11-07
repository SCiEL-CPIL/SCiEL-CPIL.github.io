"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for minimizing Branin 2D via Bayesian optimization.
    Uses a Knowledge Gradient (KG) approximation: estimates the probability
    that each candidate is the global minimizer, which naturally balances
    exploration and exploitation for multimodal functions.
    
    mu: (2000,) array of GP mean predictions
    sigma: (2000,) array of GP standard deviation predictions
    f_best: scalar, best observed value so far
    Returns: (2000,) array of scores, higher is better
    """
    # Ensure sigma is safe for division
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Knowledge Gradient approximation:
    # For each candidate i, KG(i) ≈ P(f(x_i) is the global min)
    # This can be approximated as the probability that f(x_i) < min(all other candidates)
    
    # A practical approximation: 
    # KG(i) ≈ Φ((f_best - mu_i) / sigma_i) * [1 - β * (sigma_i / (f_best - mu_i + eps))]
    # where the second term penalizes high uncertainty relative to potential improvement
    
    # Better: Use the fact that KG ≈ E[min_j f(x_j) | x_i] - min_j E[f(x_j)]
    # Approximate as: P(f(x_i) < f_best) adjusted for exploration
    
    # Standard KG approximation for minimization:
    # For candidate i, the probability it beats the current best is:
    # p_i = Φ((f_best - mu_i) / sigma_i)
    
    # But KG also considers that choosing i reveals information.
    # A common approximation is:
    # KG(i) ≈ (f_best - mu_i) * Φ(z_i) + sigma_i * φ(z_i)  [this is just EI]
    
    # To get a truly different behavior, use a probability-weighted approach:
    # Score = P(candidate is minimizer) ≈ Φ((f_best - mu) / sigma) 
    # but weighted by how much it could improve
    
    # Let's use a hybrid: combine EI with a probability-of-improvement term
    # that has different scaling to encourage exploration of unexplored regions
    
    z = (f_best - mu) / sigma_safe
    
    # Probability of improvement (POI)
    poi = norm.cdf(z)
    
    # Expected improvement (EI)  
    ei = (f_best - mu) * poi + sigma_safe * norm.pdf(z)
    
    # Knowledge-gradient-inspired score:
    # Weight the EI by the probability that this point is in a promising region
    # Use a softmax-like normalization to create competition between candidates
    
    # Compute a "confidence" that this point is near a minimum
    # Points with low mu and moderate sigma are most promising
    
    # Alternative: Use the analytic KG approximation
    # KG(i) = E[min_j f(x_j)] - E[min_{j≠i} f(x_j)]
    # ≈ P(f(x_i) < min_{j≠i} f(x_j)) * E[f(x_i) | f(x_i) < min_{j≠i} f(x_j)]
    
    # Practical approximation:
    # 1. For each candidate, compute the probability it's better than current best
    # 2. Multiply by the expected value given it's better (truncated normal mean)
    # 3. This naturally explores regions with high uncertainty that could contain minima
    
    # Truncated normal mean: if X ~ N(mu, sigma^2) and X < f_best
    # E[X | X < f_best] = mu - sigma * φ(z) / Φ(z) where z = (f_best - mu)/sigma
    
    # For large z, this approaches mu
    # For small z, this is significantly lower than mu
    
    # KG score ≈ P(improvement) * (f_best - E[X | X < f_best])
    #           = Φ(z) * (f_best - mu + sigma * φ(z)/Φ(z))
    #           = Φ(z) * (f_best - mu) + sigma * φ(z)
    #           = EI  (same as before!)
    
    # So the standard EI IS the first-order KG approximation.
    # To get different behavior, use a second-order approximation or
    # a different heuristic that captures multimodality.
    
    # Strategy: Use a "diversity-aware" EI that penalizes candidates
    # that are close to already-explored regions (in prediction space).
    # Since we don't have coordinates, we use the GP predictions themselves.
    
    # Candidates with similar (mu, sigma) profiles to the current best
    # are likely in the same basin. Penalize them to encourage exploration.
    
    # Find the candidate with the best EI as a reference
    ei_ref = np.max(ei)
    
    # Compute a "diversity" score: how different is each candidate's
    # prediction profile from the top EI candidates?
    # Use the mu and sigma values directly
    
    # Top 10 candidates by EI
    top_idx = np.argsort(ei)[-10:]
    top_mu = mu[top_idx]
    top_sigma = sigma[top_idx]
    
    # For each candidate, compute distance to the top-10 set in (mu, sigma) space
    # Normalize to make the distance meaningful
    mu_range = np.ptp(mu) + 1e-12
    sigma_range = np.ptp(sigma) + 1e-12
    
    # Average distance to top-10 in normalized space
    dist_to_top = np.zeros(len(mu))
    for i in range(len(mu)):
        d_mu = np.abs(mu[i] - top_mu) / mu_range
        d_sigma = np.abs(sigma[i] - top_sigma) / sigma_range
        dist_to_top[i] = np.mean(np.sqrt(d_mu**2 + d_sigma**2))
    
    # Blend EI with a diversity bonus
    # Higher diversity = more likely to be in a new basin
    diversity_bonus = dist_to_top * np.std(ei)
    
    # Final score: EI + small diversity bonus
    # The diversity bonus is scaled to be small relative to EI
    scores = ei + 0.1 * diversity_bonus
    
    # Ensure finite and non-negative
    scores = np.maximum(scores, 0.0)
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
