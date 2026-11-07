import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d on [-5, 5].
    Uses a Knowledge Gradient (KG) approximation combined with adaptive UCB.
    The KG component estimates the expected reduction in the global minimum
    by considering the probability that each candidate becomes the new best,
    weighted by the expected improvement over the current best.
    """
    start_time = time.time()
    deadline = 315.0  # Safety margin before 330s limit

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Safety checks: ensure finite values and avoid division by zero
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-12), sigma, 1e-12)
    mu = np.where(np.isfinite(mu), mu, f_best)

    # Compute z-score: (f_best - mu) / sigma
    # For minimization, we want f(x) < f_best
    z = (f_best - mu) / sigma
    z = np.clip(z, -10, 10)

    # Probability of Improvement
    pi = norm.cdf(z)
    pi = np.where(np.isfinite(pi), pi, 0.0)

    # Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where phi is the standard normal PDF
    ei = (f_best - mu) * pi + sigma * norm.pdf(z)
    ei = np.where(np.isfinite(ei), ei, 0.0)
    ei = np.maximum(ei, 0.0)

    # Knowledge Gradient approximation:
    # The KG value is the expected reduction in the global minimum.
    # A simple approximation is: KG ≈ EI * (1 - probability that another point is already better)
    # But since we don't have access to other points, we use a heuristic:
    # KG ≈ EI * pi, which emphasizes points that are both likely to improve AND have high expected improvement.
    # This creates a stronger preference for points that are "safe bets" for global improvement.
    kg_approx = ei * pi

    # Adaptive UCB component for exploration
    # f_best starts high (up to 25) and decreases to 0.
    # We use a beta that starts higher (more exploration) and decreases (more exploitation).
    max_f = 25.0
    ratio = np.clip(f_best / max_f, 0.0, 1.0)
    beta = 0.3 + 0.7 * np.sqrt(ratio)

    # UCB score for minimization: -mu + beta * sigma
    ucb = -mu + beta * sigma
    ucb = np.where(np.isfinite(ucb), ucb, -np.inf)

    # Normalize EI and UCB to comparable scales
    # EI is in objective units, UCB is in objective units
    # But their magnitudes can differ, so we blend them carefully
    # Scale EI by a factor to make it comparable to UCB
    # Use f_best as a reference scale
    scale = np.maximum(f_best, 1e-6)
    ei_scaled = ei / scale
    kg_scaled = kg_approx / scale

    # Combine KG and UCB
    # KG provides the main acquisition signal (global convergence), UCB adds exploration
    # Weight KG more heavily as it directly optimizes for global minimum reduction
    score = 1.0 * kg_scaled + 0.3 * (ucb / scale)

    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)

    # Deadline guard: if computation took too long, return a safe fallback
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Fallback: return EI only (fastest path)
        return ei

    return score
