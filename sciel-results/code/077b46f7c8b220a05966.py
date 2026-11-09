"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Lower Confidence Bound (LCB) for minimization.
    # We want to minimize f.
    # LCB value is mu - beta * sigma.
    # We want to pick the point with the LOWEST LCB value.
    # However, the evaluator picks the HIGHEST score.
    # So we return the negative of the LCB value: -mu + beta * sigma.
    # This makes high scores correspond to low mu (good) and high sigma (exploration).
    # Use a larger beta to encourage more exploration in the early stages
    # of optimization, which helps escape local minima in the multimodal
    # Michalewicz function.
    beta = 3.5
    return -mu + beta * sigma
# EVOLVE-BLOCK-END
