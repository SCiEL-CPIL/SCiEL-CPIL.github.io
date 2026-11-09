import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    # Sanitize inputs
    sigma = np.asarray(sigma, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    sigma = np.maximum(sigma, 1e-8)
    sigma = np.nan_to_num(sigma, nan=1e-8, posinf=1e6, neginf=1e-8)
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    cost = np.maximum(cost, 1e-8)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1.0)

    # z-score for EI
    z = (f_best - mu) / sigma
    z = np.clip(z, -50, 50)

    # Expected Improvement for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

    # Cost-aware score: combine EI/cost with a small uncertainty bonus
    # The uncertainty bonus encourages exploration in high-variance regions
    # while the EI/cost term focuses on promising, cheap points
    scores = ei / cost + 0.05 * sigma / cost

    # Ensure finite output
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)

    return scores
