# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm
from scipy.optimize import minimize

# Internal deadline guard (seconds)
_DEADLINE = 325.0
_T0 = time.time()


def _check_deadline():
    return time.time() - _T0 > _DEADLINE


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition for Bayesian optimization of cosine8 8d.
    
    Strategy:
    1. Compute EI per unit cost as a baseline.
    2. Additionally, perform a local refinement around the best GP mean region
       using a lightweight local optimizer on the acquisition function itself,
       but since we don't have coordinates, we must rely on the mu/sigma structure.
       
    Actually, we don't have coordinates, so we can't do local optimization in x-space.
    We must work purely with mu, sigma, f_best, cost.
    
    Key insight: The current EI/cost is a strong baseline. Let's try a different
    acquisition that accounts for the cost structure more carefully.
    
    Alternative: Use a variant of EI that penalizes high-cost points more aggressively,
    or use a "probability of improvement" weighted by cost efficiency.
    
    Let's try: EI with an additional term that rewards points with low cost and high improvement.
    Or: Use a "cost-aware" EI that considers the expected improvement relative to the cost.
    
    Another idea: The cosine8 function has a known structure. The minimum is at the center.
    Since we don't have coordinates, we can't directly exploit this, but the GP should learn it.
    
    Let's try a different acquisition: Expected Improvement with a cost penalty that is
    adaptive based on the current f_best and the expected cost.
    
    Actually, let's try a simpler structural change: use a "thompson sampling" like approach
    by adding noise to the GP mean and selecting the point with the best (mu - noise)/cost.
    But we need to return scores for 2000 points, not sample.
    
    Let's try: EI with a different cost weighting. Instead of dividing by cost, use
    EI - lambda * cost, where lambda is adaptive.
    
    Or: Use a "cost-aware" EI where the improvement is scaled by the inverse of the cost
    in a more nuanced way.
    
    Let's try: Expected Improvement with a cost-aware threshold.
    """
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Standard EI
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Cost-aware EI: divide by cost
    # But let's try a different approach: use a "cost-efficiency" metric
    # that combines EI with a penalty for high cost
    
    # Idea: Use a "probability of improvement" weighted by cost
    # Or: Use a "cost-aware" EI that accounts for the uncertainty in cost
    
    # Let's try a different structural change: use a "multi-objective" approach
    # where we consider both the objective value and the cost as objectives.
    
    # For now, let's try a simple modification: use EI with a cost penalty
    # that is proportional to the square of the cost, to penalize high-cost points more.
    
    # Alternative: Use a "cost-aware" EI that scales the improvement by the inverse
    # of the cost, but with a floor to avoid division by near-zero costs.
    
    # Let's try: EI / (cost + epsilon) where epsilon is small
    scores = ei / (cost + 1e-6)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
