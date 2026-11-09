"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best, cost):
    # UCB-style acquisition for minimization, normalized by cost.
    # We want to find points with low mu and high sigma (uncertainty).
    # Score = (Potential Improvement) / Cost
    # Potential Improvement approximated by (f_best - mu) + beta * sigma
    beta = 1.5
    potential_improvement = (f_best - mu) + beta * sigma
    # Ensure non-negative scores for stability, though argmax doesn't care about shift
    return potential_improvement / cost
# EVOLVE-BLOCK-END
