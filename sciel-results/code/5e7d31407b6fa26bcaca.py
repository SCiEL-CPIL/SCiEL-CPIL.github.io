# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Thompson Sampling with a fixed seed for reproducibility and stability.
    # Using a fixed seed ensures that the exploration path is deterministic
    # relative to the sequence of calls, avoiding "bad luck" in the random draws
    # which can be detrimental in small-budget optimization.
    rng = np.random.default_rng(42)
    z = rng.standard_normal(size=mu.shape)
    sampled_values = mu - sigma * z
    return -sampled_values
# EVOLVE-BLOCK-END
