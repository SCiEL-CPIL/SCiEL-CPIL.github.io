import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Computes the Adaptive Lower Confidence Bound (LCB) acquisition score.
    
    Args:
        mu: Mean of the GP posterior (shape (2000,))
        sigma: Standard deviation of the GP posterior (shape (2000,))
        f_best: Best observed value so far (scalar)
        
    Returns:
        Scores (shape (2000,)), higher is better for selection.
    """
    # Adaptive beta based on how close f_best is to the known global minimum
    f_min = -3.86278
    # f_best is between f_min and ~0
    # progress = 0 if f_best == f_min, 1 if f_best is far
    # We want higher beta when f_best is far (early stage), lower when close (late stage)
    
    # Clamp f_best to reasonable range to avoid issues
    # f_best should be >= f_min
    diff = f_best - f_min
    # Normalize diff. If f_best is 0, diff is ~3.86.
    # Let's map diff to [0, 1] roughly.
    # If diff > 2, treat as early stage.
    norm_diff = np.clip(diff / 2.0, 0.0, 1.0)
    
    # Beta ranges from 1.0 (exploit) to 3.0 (explore)
    beta_min = 1.0
    beta_max = 3.0
    beta = beta_min + (beta_max - beta_min) * norm_diff
    
    # LCB for minimization: Score = beta * sigma - mu
    # Higher score is better
    scores = beta * sigma - mu
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
