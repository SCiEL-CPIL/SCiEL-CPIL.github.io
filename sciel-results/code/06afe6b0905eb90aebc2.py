import numpy as np

def heuristics(prizes, weights):
    eps = 1e-9
    ratios = prizes / (weights + eps)  # shape (5, 100)
    # Geometric mean across constraints
    log_ratios = np.log(ratios)
    geo_mean = np.exp(np.mean(log_ratios, axis=0))  # shape (100,)
    return geo_mean
