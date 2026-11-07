import numpy as np
from scipy.stats import norm
import time

_DEADLINE = time.time() + 325.0

def acquisition(mu, sigma, f_best):
    sigma = np.maximum(sigma, 1e-9)
    
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # LCB for minimization: mu - beta * sigma
    # We want to maximize the score, so we use -LCB = -mu + beta * sigma
    beta = 2.0
    lcb_score = -mu + beta * sigma
    
    # Blend
    # The magnitudes of EI and LCB_score might differ.
    # EI is typically small when f_best is good.
    # LCB_score is in the same units as mu.
    # To make them comparable, we might need to scale one.
    # However, the inspiration suggests a direct blend works.
    # Let's try: Score = 0.8 * ei + 0.2 * lcb_score
    
    score = 0.8 * ei + 0.2 * lcb_score
    
    score = np.where(np.isfinite(score), score, 0.0)
    return score
