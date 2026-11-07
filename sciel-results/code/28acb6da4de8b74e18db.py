"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# We need to track the number of evaluations to use in the UCB parameter.
# Since the function stateless, we can't easily track t across calls.
# However, the problem states "minimizes a fixed seeded random mixture... at each step draws 2000 fresh uniform candidates".
# The evaluator runs a loop. The function `acquisition` is called at each step.
# We don't have access to the step counter `t` directly in the signature `acquisition(mu, sigma, f_best)`.
# We can approximate `t` by inferring it from the state? No, mu and sigma don't contain t.
# But we can assume the number of evaluations is small (30 total).
# A common heuristic when t is not available is to use a fixed beta or a beta that depends on the uncertainty.
# Alternatively, we can use a simple heuristic: start with high exploration and decrease it.
# But without t, we can't decrease it.
# Let's look at the constraints: "30 evaluations including the first 5".
# So t goes from 1 to 25 (after initial 5).
# Since we can't track t, we might have to use a fixed beta or a beta based on some other metric.
# However, a very common and robust choice for GP-UCB in BO is to use a beta that balances exploration and exploitation.
# A static beta around 1-10 often works well for small search spaces.
# Let's try a static beta first, or perhaps a beta that is slightly adaptive based on the magnitude of f_best?
# Actually, since we cannot track t, let's use a fixed beta that is known to perform well in similar 2D problems.
# A value of beta = 5.0 is a reasonable starting point for GP-UCB in 2D with 30 evals.
# Let's also consider that EI is the baseline (score 1.0). GP-UCB is often comparable or better.
# To be safe, let's use a slightly higher beta to encourage exploration initially, assuming the early calls are for exploration.
# Wait, if we use a fixed beta, it might be too exploratory late in the run or not exploratory enough early on.
# But without t, we have no choice.
# Let's try beta = 4.0.

BETA = 4.0

def acquisition(mu, sigma, f_best):
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # GP-UCB: mu + sqrt(beta) * sigma
    # We want to maximize this score, so higher values are better.
    # The task says "Return 2000 finite scores, higher meaning evaluate next".
    # So we return the UCB values.
    scores = mu + np.sqrt(BETA) * sigma_safe
    
    # Ensure scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
