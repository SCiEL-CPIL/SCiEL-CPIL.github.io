import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Pareto-Filtered UCB for Weierstrass 1D minimization.
    
    Structural Change: Instead of a normalized hybrid of EI and UCB, this method
    identifies the Pareto frontier of the posterior statistics (minimizing mean,
    maximizing uncertainty) and assigns high scores only to points on this frontier,
    using UCB to break ties. This forces exploration into regions that are either
    highly uncertain or have low predicted means, avoiding "compromise" points that
    are mediocre in both aspects.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Adaptive Beta:
    # If f_best is close to 0 (global min), we want to exploit (lower beta).
    # If f_best is far, we want to explore (higher beta).
    # Weierstrass min is 0.
    # Heuristic: beta decays as f_best approaches 0.
    # f_best is typically in [0, ~1].
    # Let's use a smooth decay.
    # beta = 1.0 + 2.0 * (f_best / (f_best + 0.01))
    # When f_best >> 0.01, beta ~ 3.0.
    # When f_best ~ 0, beta ~ 1.0.
    beta = 1.0 + 2.0 * (f_best / (f_best + 0.01))
    
    # UCB Score for minimization: -mu + beta * sigma
    # Higher is better.
    ucb_score = -mu + beta * sigma
    
    # Identify Pareto Frontier
    # We want points that are not dominated in the (mu, sigma) space.
    # Dominance: Point A dominates Point B if mu_A <= mu_B and sigma_A >= sigma_B, with at least one strict inequality.
    # We want the "upper-right" frontier of the (mu, sigma) plot if we plot mu on x (min) and sigma on y (max).
    # Actually, standard Pareto for minimization of mu and maximization of sigma:
    # Sort by mu ascending.
    # Keep points where sigma is strictly greater than the maximum sigma seen so far in the sorted order.
    
    # Sort indices by mu
    sorted_indices = np.argsort(mu)
    sorted_mu = mu[sorted_indices]
    sorted_sigma = sigma[sorted_indices]
    
    # Initialize mask
    pareto_mask = np.zeros(len(mu), dtype=bool)
    
    # Traverse sorted by mu
    max_sigma_seen = -np.inf
    for i in range(len(sorted_mu)):
        if sorted_sigma[i] > max_sigma_seen:
            pareto_mask[sorted_indices[i]] = True
            max_sigma_seen = sorted_sigma[i]
        # If sigma is not > max_sigma_seen, it is dominated by a point with lower mu and higher/equal sigma.
        
    # Assign scores
    # Points on the frontier get their UCB score.
    # Points not on the frontier get a very low score (e.g., -np.inf or min_score - epsilon).
    # Using -np.inf might cause issues if all are -inf, but at least one will be True.
    # To be safe, use a large negative number.
    
    scores = np.full(len(mu), -1e9, dtype=np.float64)
    scores[pareto_mask] = ucb_score[pareto_mask]
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, -1e9)
    
    return scores
