import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Knowledge Gradient (KG) for Bayesian optimization.
    
    KG estimates the expected value of information from evaluating each candidate.
    For 1D problems with Gaussian posteriors, this can be computed analytically.
    """
    start_time = time.time()
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-9)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    
    # Ensure mu is finite
    mu = np.where(np.isfinite(mu), mu, 0.0)
    
    # Knowledge Gradient approximation:
    # For minimization, KG(x) ≈ E[max(f_best - Y, 0)] where Y ~ N(mu(x), sigma(x)^2)
    # This is equivalent to EI, but we use a different formulation that emphasizes
    # the probability of being the new best.
    
    # Standardize: z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Probability that this point will be the new best:
    # P(Y < f_best) = Phi(z) where z = (f_best - mu)/sigma
    prob_best = norm.cdf(z)
    
    # Expected improvement (for reference, but we use a different weighting)
    ei = (f_best - mu) * prob_best + sigma * norm.pdf(z)
    
    # KG-like score: weight by both the probability of being best AND the magnitude
    # of improvement, but with a different functional form
    # Use: score = prob_best * (f_best - mu) + sigma * norm.pdf(z) * (1 - prob_best)
    # This emphasizes points that are likely to be best AND have significant improvement
    
    # Alternative structural change: use a "probability-weighted improvement"
    # that accounts for the uncertainty in whether this is truly the best
    score = prob_best * (f_best - mu) + 0.5 * sigma * norm.pdf(z)
    
    # Add a small exploration bonus for points with high uncertainty
    # but only when they're not already very promising
    exploration_bonus = sigma * (1 - prob_best) * 0.1
    score += exploration_bonus
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Time guard
    if time.time() - start_time > 290:
        return -mu + 0.1 * sigma
    
    return score
