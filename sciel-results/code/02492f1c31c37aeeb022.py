import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Cost-Adjusted Expected Improvement with Probability of Success weighting
    # Based on Stochastic Shortest Path analogy:
    # Score = Expected Improvement / (Cost / Probability of Improvement)
    # Score = (Expected Improvement * Probability of Improvement) / Cost
    
    # Avoid division by zero or negative costs
    cost = np.maximum(cost, 1e-8)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-8)
    
    # Calculate z-score for Expected Improvement
    z = (f_best - mu) / sigma
    
    # Calculate CDF and PDF of the standard normal distribution
    # Phi(z) is the probability that f(x) < f_best
    # phi(z) is the density at the threshold
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Calculate Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * cdf_z + sigma * pdf_z
    
    # Ensure EI is non-negative
    ei = np.maximum(ei, 0.0)
    
    # Apply the "Stochastic Shortest Path" adjustment
    # Weight EI by the probability of improvement (cdf_z)
    # This effectively divides the cost by the probability of success,
    # increasing the effective cost-to-go for points with low probability of improvement.
    adjusted_ei = ei * cdf_z
    
    # The acquisition score is the adjusted EI divided by cost
    scores = adjusted_ei / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
