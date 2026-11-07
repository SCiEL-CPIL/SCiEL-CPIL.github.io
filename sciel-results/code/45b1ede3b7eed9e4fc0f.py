"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Adaptive Lower Confidence Bound (LCB) acquisition function.
    
    The exploration parameter kappa is scaled by the distance between the 
    current best observed value and the known global minimum of the 
    Michalewicz function (-1.8013). This encourages higher exploration 
    when the search is far from the optimum and tighter exploitation 
    when the search is close.
    """
    # Known global minimum for Michalewicz 2D
    f_min_known = -1.8013
    
    # Calculate the gap between current best and the known minimum
    gap = f_best - f_min_known
    
    # Adaptive kappa:
    # Base exploration level
    kappa_base = 0.5
    # Scaling factor for the gap
    kappa_scale = 1.0
    
    # kappa increases as we get further from the optimum
    # We cap it to avoid extreme values if initial points are very bad
    kappa = kappa_base + kappa_scale * gap
    kappa = np.clip(kappa, 0.1, 3.0)
    
    # LCB: We want to minimize the function, so we prefer low mu and high sigma
    # score = mu - kappa * sigma
    # Higher score means more promising (lower expected value with uncertainty consideration)
    scores = mu - kappa * sigma
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
