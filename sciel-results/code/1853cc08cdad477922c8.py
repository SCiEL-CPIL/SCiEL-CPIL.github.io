"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    # Use a robust variant: UCB with an adaptive beta that balances exploration/exploitation.
    # Since we don't have access to the number of evaluations directly, we can use a heuristic
    # based on the spread of sigma or just use a fixed but well-tuned beta.
    # However, a common and strong alternative to EI is the Probability of Improvement (PI)
    # or Upper Confidence Bound (UCB). UCB is often better for global optimization.
    # Let's use UCB: mu - beta * sigma (since we are minimizing, we want low mu, so we subtract
    # a multiple of sigma to explore areas with high uncertainty).
    # Actually, for minimization:
    # EI: (f_best - mu) * Phi(z) + sigma * phi(z), where z = (f_best - mu) / sigma
    # UCB: mu - beta * sigma (we want to minimize, so we prefer low mu, but we also want to explore
    #       high sigma regions. So we subtract beta*sigma).
    # Let's try UCB with a fixed beta. A common choice is beta=1.0 or slightly larger.
    # But wait, the current program uses EI which is score 1.0. Maybe UCB will do better?
    # Alternatively, we can use a hybrid or a different acquisition function.
    # Let's try UCB with beta=1.5.
    beta = 1.5
    scores = mu - beta * sigma
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    return scores
# EVOLVE-BLOCK-END
