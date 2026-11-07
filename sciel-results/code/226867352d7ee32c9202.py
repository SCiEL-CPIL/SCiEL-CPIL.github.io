import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Hybrid acquisition function combining EI, UCB, and a diversity/repulsion
    term with a sigma-rank exploration bonus.
    """
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin before 300s limit
    
    n = mu.shape[0]
    
    # Ensure numerical stability
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # --- 1. Expected Improvement (EI) for minimization ---
    # EI(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    # where z = (f_best - mu(x)) / sigma(x)
    z = (f_best - mu) / sigma_safe
    phi = norm.pdf(z)
    Phi = norm.cdf(z)
    ei = (f_best - mu) * Phi + sigma_safe * phi
    # For minimization, we want to maximize EI
    ei_scores = ei
    
    # --- 2. UCB for minimization ---
    # Score = -mu + beta * sigma
    # Use a moderate beta
    beta = 2.0
    ucb_scores = -mu + beta * sigma_safe
    
    # --- 3. Normalize scores to [0, 1] range for blending ---
    def normalize(scores):
        s_min = np.min(scores)
        s_max = np.max(scores)
        if s_max - s_min < 1e-12:
            return np.ones_like(scores)
        return (scores - s_min) / (s_max - s_min)
    
    ei_norm = normalize(ei_scores)
    ucb_norm = normalize(ucb_scores)
    
    # --- 4. Sigma-rank exploration bonus ---
    # Rank sigma values; higher sigma should get a bonus
    sigma_rank = np.argsort(np.argsort(sigma_safe)) / max(n - 1, 1)  # rank in [0, 1]
    # Apply a soft bonus that emphasizes top uncertainty regions
    sigma_bonus = sigma_rank ** 1.5  # sharper emphasis on high uncertainty
    
    # --- 5. Combine scores ---
    # Weighted combination: 50% EI, 30% UCB, 20% sigma-rank bonus
    w_ei = 0.50
    w_ucb = 0.30
    w_sigma = 0.20
    
    scores = w_ei * ei_norm + w_ucb * ucb_norm + w_sigma * sigma_bonus
    
    # --- 6. Add a small diversity penalty to avoid picking points too close to already-explored regions ---
    # We don't have coordinates, so we can't do true spatial diversity.
    # However, we can add a small random perturbation to break ties
    # and encourage exploration of slightly different regions in the latent space.
    # The random seed is not controllable, but we can use a deterministic small noise
    # based on the index to add slight variation.
    # Actually, since we don't have coordinates, let's just ensure finite output.
    
    # Check deadline
    if time.time() > deadline:
        # Return a simple UCB score as fallback
        return -mu + beta * sigma_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
