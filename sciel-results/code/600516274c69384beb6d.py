"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def heuristics(prizes, weights):
    """
    Computes heuristic scores for items in a Multi-Constraint Knapsack Problem.
    
    The heuristic combines:
    1. Total Prize: Rewards high-value items.
    2. Geometric Mean of Inverse Weights: Penalizes items that consume significant 
       capacity in ANY constraint (bottleneck awareness).
    3. Minimum Prize-to-Weight Ratio: Penalizes items that are inefficient in 
       their most constraining dimension.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    
    # Ensure shapes are correct
    if prizes.shape != weights.shape:
        raise ValueError("Prizes and weights must have the same shape")
    
    # Add epsilon to avoid division by zero
    eps = 1e-9
    w_safe = weights + eps
    p_safe = prizes + eps
    
    # 1. Total prize per item
    total_prize = np.sum(prizes, axis=0)
    
    # 2. Geometric mean of inverse weights (penalizes bulk in any constraint)
    # log(1/w) = -log(w)
    log_inv_w = -np.log(w_safe)
    geo_mean_inv_w = np.exp(np.mean(log_inv_w, axis=0))
    
    # 3. Minimum prize-to-weight ratio across constraints
    ratios = p_safe / w_safe
    min_ratio = np.min(ratios, axis=0)
    
    # Combine components
    # Weights for the combination can be tuned, but equal weighting is a good start
    # Score = Total Prize * GeoMean(1/W) * Min(Ratio)
    scores = total_prize * geo_mean_inv_w * min_ratio
    
    # Ensure positivity and finiteness
    scores = np.where(np.isfinite(scores) & (scores > 0), scores, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
