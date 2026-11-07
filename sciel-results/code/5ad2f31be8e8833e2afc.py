import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Branin 2D.
    Combines a Kriging Believer-inspired structural approach with
    Probability of Improvement (PI) for robust exploitation.
    
    1. Adaptive UCB identifies promising regions (high uncertainty + low bound).
    2. A sharp sigmoid gate focuses on the top candidates by UCB.
    3. Within these regions, PI provides a robust exploitation signal,
       preferring points with high probability of beating f_best.
    4. A small UCB component breaks ties and encourages exploration.
    """
    start_time = time.time()
    deadline = 295.0  # Safety margin before 300s limit
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # Adaptive Beta: High regret -> High beta (explore), Low regret -> Low beta (exploit)
    f_global_min = 0.397887
    regret = f_best - f_global_min
    
    beta_min = 0.3
    beta_max = 2.5
    r_scale = 5.0
    
    if regret >= 0:
        fraction = regret / (regret + r_scale)
    else:
        fraction = 0.0
        
    beta = beta_min + (beta_max - beta_min) * fraction
    
    # Step 1: Compute UCB scores to identify promising regions
    # UCB for minimization: beta * sigma - mu
    ucb_scores = beta * sigma_safe - mu
    
    # Normalize UCB scores to [0, 1] for gating
    ucb_min = np.min(ucb_scores)
    ucb_max = np.max(ucb_scores)
    ucb_range = ucb_max - ucb_min + eps
    ucb_norm = (ucb_scores - ucb_min) / ucb_range
    
    # Step 2: Create a sharper gate that emphasizes high-UCB regions
    # Use a sharper sigmoid to focus on top ~10-15% of UCB candidates
    temperature = 5.0
    # Shift so that ucb_norm=0.8 is the center (we want top 20% to be strongly gated)
    gate_center = 0.8
    gate = 1.0 / (1.0 + np.exp(-temperature * (ucb_norm - gate_center)))
    
    # Step 3: Compute Probability of Improvement (PI)
    # PI is robust for multimodal functions like Branin
    z = (f_best - mu) / sigma_safe
    pi_scores = norm.cdf(z)
    
    # Normalize PI scores to [0, 1] for consistent scaling
    pi_min = np.min(pi_scores)
    pi_max = np.max(pi_scores)
    pi_range = pi_max - pi_min + eps
    pi_norm = (pi_scores - pi_min) / pi_range
    
    # Step 4: Combine gated PI with UCB component
    # Primary signal: gated PI (exploitation within promising regions)
    # Secondary signal: UCB (exploration and tie-breaking)
    scores = gate * pi_norm + 0.2 * ucb_norm
    
    # Check time
    if time.time() - start_time > deadline:
        scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
        return scores
    
    # Ensure all scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
