"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    # prizes: (5, 100), weights: (5, 100), all in [0, 1]
    # Total prize per item
    total_prize = prizes.sum(axis=0)  # (100,)
    
    # Bottleneck weight: max weight across constraints per item
    bottleneck_weight = weights.max(axis=0)  # (100,)
    
    # Score: prize per bottleneck weight
    scores = total_prize / (bottleneck_weight + 1e-9)
    
    # Normalize so mean is ~1 to keep heuristic in same scale as pheromone
    mean_score = scores.mean()
    if mean_score > 0:
        scores = scores / mean_score
    
    return scores
# EVOLVE-BLOCK-END
