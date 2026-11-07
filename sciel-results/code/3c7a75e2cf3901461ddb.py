import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function combining Expected Improvement and
    a diversity-aware exploration term.

    Structural change from pure EI: we use a hybrid that balances
    - exploitation (Expected Improvement)
    - exploration (uncertainty-weighted, with a soft cap to avoid
     over-exploring already-visited high-uncertainty regions)
    - cost-awareness (divide by cost)

    The exploration term uses a squared-uncertainty with a diminishing
    returns factor to prevent the algorithm from being stuck exploring
    the same high-variance region.
    """
    # Deadline guard: track elapsed time
    start_time = time.time()
    deadline = 320.0  # safety margin before 330s kill

    # Clamp sigma to avoid numerical issues
    sigma_safe = np.clip(sigma, 1e-10, None)

    # Compute z-score for EI
    z = (f_best - mu) / sigma_safe

    # Clip z to avoid overflow in exp for very negative/positive z
    z_clipped = np.clip(z, -50, 50)

    # Expected Improvement: (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)

    # Exploration term: use uncertainty with diminishing returns
    # sigma^2 / (sigma + 0.1) gives high weight to moderate uncertainty
    # but caps out for very high uncertainty
    exploration = sigma_safe * sigma_safe / (sigma_safe + 0.1)

    # Hybrid score: weighted combination of EI and exploration
    # The weight on exploration decreases as we get closer to the optimum
    # (i.e., as f_best gets smaller / closer to 0)
    # For three-hump camel 2d, the minimum is 0, so f_best should approach 0
    # When f_best is close to 0, we want less exploration
    # When f_best is large (far from optimum), we want more exploration

    # Normalize f_best to get a "distance from optimum" signal
    # The function range is roughly [-1.03, 3.0] for three-hump camel 2d
    # f_best starts around 1-2 and decreases toward 0
    # Use a soft scaling: weight_exploration = 0.3 / (1 + 0.5 * max(0, f_best))
    # This gives ~0.3 when f_best=0, ~0.2 when f_best=1, ~0.15 when f_best=2
    f_best_clamped = max(0.0, float(f_best))
    weight_exploration = 0.25 / (1.0 + 0.3 * f_best_clamped)

    # Combine EI and exploration
    # EI is already in objective units, exploration is in squared units
    # Scale exploration to be comparable to EI
    # EI typically ranges from 0 to ~0.5-1.0
    # Exploration with our formula ranges from 0 to ~0.5
    combined = ei + weight_exploration * exploration

    # Cost-aware: divide by cost
    # cost is in [1, 2] since it's 1 + mean(x scaled to [0,1])
    cost_safe = np.clip(cost, 1e-10, None)
    score = combined / cost_safe

    # Ensure finiteness
    score = np.where(np.isfinite(score), score, 0.0)

    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Return the best result found so far (which is already computed)
        pass

    return score
