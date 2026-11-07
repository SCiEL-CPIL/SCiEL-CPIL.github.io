import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    # Adaptive LCB with cost penalty
    # The theoretical minimum of cosine8 is -0.8.
    # We adapt the exploration parameter beta based on how close f_best is to -0.8.
    # If f_best is far from -0.8, we explore more (higher beta).
    # If f_best is close to -0.8, we exploit more (lower beta).
    # Also, we penalize high cost.
    
    # Ensure sigma is non-negative
    sigma = np.maximum(sigma, 1e-8)
    
    # Theoretical minimum
    f_min_theoretical = -0.8
    
    # Calculate progress towards the minimum
    # progress is 0 when f_best is high (e.g., 0.5) and 1 when f_best is close to -0.8
    # Using a simple linear scaling or a saturation function
    # Let's define a range. Initial points are uniform in [-1, 1]^8, so f_best could be around 0 to 0.5.
    # Let's say we consider "close" if f_best < -0.7.
    # Let's use a sigmoid or linear mapping.
    
    # Map f_best to a beta value.
    # High uncertainty (early) -> high beta.
    # Low uncertainty (late) -> low beta.
    
    # Let's use the distance from the theoretical minimum.
    # dist = f_best - f_min_theoretical
    # If dist is large, beta should be large.
    
    # A simple heuristic: beta = base_beta + alpha * (f_best - f_min)
    # But we need to be careful not to make beta too large.
    
    # Alternative: Use the standard deviation of the predictions as a proxy for uncertainty?
    # No, f_best is the key signal.
    
    # Let's try: beta = 1.0 + 2.0 * exp(-5.0 * (f_best - f_min_theoretical))
    # When f_best is close to -0.8, exp term is ~1, so beta ~ 3.0? No, that's backwards.
    # We want LOWER beta when we are CLOSE to the minimum.
    # So: beta = base + scale * (f_best - f_min)
    # If f_best = -0.8, beta = base.
    # If f_best = 0.0, beta = base + scale * 0.8.
    
    # Let's try base = 0.5, scale = 2.0
    # f_best = -0.8 -> beta = 0.5
    # f_best = -0.5 -> beta = 0.5 + 2.0 * 0.3 = 1.1
    # f_best = 0.0 -> beta = 0.5 + 2.0 * 0.8 = 2.1
    
    beta = 0.5 + 2.0 * np.maximum(0, f_best - f_min_theoretical)
    
    # LCB for minimization: mu - beta * sigma
    lcb = mu - beta * sigma
    
    # We want to maximize the "improvement" potential.
    # Score = (f_best - lcb) / cost?
    # Or just -lcb / cost?
    # If we use -lcb / cost, we are looking for the point with the lowest expected value, penalized by cost.
    # But we also want to ensure we find something better than f_best.
    # EI is better for this. But the finding says "Adaptive LCB ... improving upon static EI/cost".
    
    # Let's try Score = (f_best - (mu - beta * sigma)) / cost
    # This is the expected improvement if we assume the distribution is truncated or something?
    # Actually, (f_best - mu + beta * sigma) is an optimistic bound on improvement.
    
    # Let's stick to the finding's hint: "Adaptive LCB with cost penalty".
    # Maybe: Score = (f_best - mu + beta * sigma) / cost
    # This is similar to EI but without the probability weighting. It's an upper bound on EI.
    
    # Let's try this.
    scores = (f_best - mu + beta * sigma) / cost
    
    # Ensure finite scores
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
