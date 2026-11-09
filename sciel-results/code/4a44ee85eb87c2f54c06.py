# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    # Thompson Sampling variant: draw a single scalar z ~ N(0,1) and use
    # sampled_values = mu - sigma * z.  The scalar draw correlates the
    # posterior samples across all 2000 candidates, which tends to produce
    # a coherent exploration direction rather than independent noise at each
    # point.  This can help the sampler "commit" to a promising region for
    # the next evaluation, reducing wasted exploration in a 2-D landscape.
    #
    # We use a deterministic seed based on f_best so that the exploration
    # direction changes as the best value improves, giving a natural
    # annealing effect: early (poor f_best) the direction is one thing,
    # later (better f_best) it shifts, encouraging new exploration.
    #
    # Additionally, we add a small exploitation bias: candidates whose mu is
    # already close to f_best get a slight boost, so that when the GP mean
    # is informative the sampler is more likely to exploit it.
    rng = np.random.default_rng(int(abs(hash(f_best)) % (2**31)))
    z_scalar = rng.standard_normal()
    sampled_values = mu - sigma * z_scalar

    # Mild exploitation boost: reward points where mu is near f_best
    # This is a small nudge that doesn't dominate the TS signal
    mu_range = np.ptp(mu)
    if mu_range > 0:
        exploit = 1.0 - np.abs(mu - f_best) / (mu_range + 1e-12)
        exploit = np.clip(exploit, 0.0, 1.0)
        sampled_values = sampled_values - 0.1 * exploit

    return -sampled_values
# EVOLVE-BLOCK-END
