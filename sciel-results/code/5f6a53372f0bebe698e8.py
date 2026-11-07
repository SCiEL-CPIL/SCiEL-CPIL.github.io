import numpy as np

def acquisition(mu, sigma, f_best):
    # Adaptive UCB
    # f_best is the current minimum observed value.
    # For sphere 1d on [-5, 5], max value is 25.
    # We want high exploration (high kappa) when f_best is large,
    # and low exploration (low kappa) when f_best is small.
    
    # Normalize f_best to [0, 1] roughly
    # f_best starts around 10-25 and goes to 0.
    # Let's use a decay function.
    # kappa = base + range * decay(f_best)
    
    # If f_best is large, decay is high -> kappa is high.
    # If f_best is small, decay is low -> kappa is low.
    
    # Example: decay = 1 / (1 + f_best)
    # If f_best = 20, decay ~ 0.05. kappa ~ base.
    # If f_best = 0, decay = 1. kappa ~ base + range.
    # This is opposite of what we want.
    
    # We want kappa HIGH when f_best is LARGE.
    # So kappa should increase with f_best.
    
    # kappa = 1.0 + 3.0 * (f_best / 10.0)
    # Cap it to avoid extreme values.
    
    scale = 10.0
    ratio = f_best / scale
    if ratio > 1.0:
        ratio = 1.0
        
    kappa = 1.0 + 3.0 * ratio
    
    # UCB for minimization: mu - kappa * sigma
    scores = mu - kappa * sigma
    
    # Ensure finite scores
    return scores
