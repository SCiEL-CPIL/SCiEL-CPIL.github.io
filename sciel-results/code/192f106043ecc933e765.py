import numpy as np
import time

_START_TIME = time.time()

def acquisition(mu, sigma, f_best):
    # Deadline guard
    if time.time() - _START_TIME > 320:
        return np.zeros_like(mu)
    
    # Pareto-frontier guided exploration
    # 1. Identify the exploitation anchor: candidate with lowest predicted mean
    anchor_idx = np.argmin(mu)
    anchor_mu = mu[anchor_idx]
    
    # 2. Compute normalized distance from anchor in prediction space
    #    This measures how far each candidate is from the current best prediction
    mu_range = np.ptp(mu)
    if mu_range > 1e-10:
        # Normalize mu values to [0, 1] range
        mu_norm = (mu - np.min(mu)) / mu_range
    else:
        mu_norm = np.zeros_like(mu)
    
    # Distance from anchor in normalized prediction space
    # Candidates far from the anchor in prediction space are in different regions
    dist_from_anchor = np.abs(mu_norm - mu_norm[anchor_idx])
    
    # 3. Combine uncertainty (sigma) with geometric distance
    #    High sigma AND far from anchor = high priority for exploration
    #    This creates a Pareto frontier: we want both high uncertainty AND diversity
    
    # Normalize sigma to [0, 1] for balanced combination
    sigma_range = np.ptp(sigma)
    if sigma_range > 1e-10:
        sigma_norm = (sigma - np.min(sigma)) / sigma_range
    else:
        sigma_norm = np.zeros_like(sigma)
    
    # Product of normalized uncertainty and normalized distance
    # This prioritizes regions that are both uncertain AND different from current best
    scores = sigma_norm * dist_from_anchor
    
    # Add a small exploitation component to ensure we still consider good predictions
    # Weight decreases as we get closer to the optimum
    progress = f_best / (f_best + 1.0)
    exploit_weight = 0.1 * progress
    scores += exploit_weight * (1 - mu_norm)
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
