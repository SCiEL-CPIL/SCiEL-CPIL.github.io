# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time


def acquisition(mu, sigma, f_best, cost):
    # Cost-aware acquisition for minimizing a multimodal objective.
    # Structural Change: Use strict Cost-Adjusted Expected Improvement.
    # Remove exploration/UCB terms and constant cheapness bonuses.
    # This prioritizes evaluations that offer the highest expected improvement
    # per unit of cost, which is critical in tight budget regimes.

    # Guard: ensure finite inputs and positive sigma/cost.
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)

    # --- Expected Improvement (for minimization) ---
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    z = (f_best - mu) / sigma
    # Numerically stable calculation
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # --- Cost Adjustment ---
    # We want to maximize improvement per cost.
    # Using EI / cost directly can be very aggressive.
    # Using EI / sqrt(cost) is a common compromise, but the lesson suggests
    # strict alignment with "high-efficiency evaluations".
    # Let's try EI / cost first as the structural shift from the blended model.
    # However, if costs are very low (near 1), this behaves like EI.
    # If costs are high (near 2), it penalizes heavily.
    # This effectively biases the search towards the center of the domain
    # (where cost is 1) if the EI is comparable, but prioritizes high EI
    # regions even if they are slightly more expensive.
    
    score = ei / cost

    # Ensure finite output.
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
