"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# We track the number of acquisition calls to estimate the progress of the optimization loop.
# The loop runs until 30 evaluations. The first 5 are random.
# So acquisition is called 25 times (for evaluations 6 through 30).
# However, the function is stateless between calls in the sense that we don't know the seed,
# but we can assume the global state persists within a single run of the evaluator for a specific seed.
# The prompt says "For each seed 0, 1 and 2 the evaluator starts...".
# It implies the Python process might be restarted or the module reloaded for each seed, 
# or the global state might persist if it's the same process.
# Given "Keep the function names and signatures", and standard BO loop implementations in these tasks,
# it is safer to assume the function is called sequentially within a single run.
# If the module is reloaded per seed, _call_count resets. If not, it accumulates.
# To be robust, we can cap the influence of the call count or use a heuristic that doesn't strictly depend on the absolute count if it might be off.
# However, the most robust approach for Thompson Sampling is often a fixed sampling strategy or a very simple one that doesn't require precise progress tracking.
# Let's use a standard Thompson Sampling approach:
# 1. Sample a function from the GP posterior: f_s(x) = mu(x) + sigma(x) * z, where z ~ N(0, 1).
# 2. Select the x that minimizes f_s(x).
# 
# But we must return scores for ALL 2000 candidates such that the argmax score is the selected candidate.
# We can set the score of the selected candidate to a high value and others to lower values, 
# or simply return the negative of the sampled function values.
# If we return -f_s(x) for all x, the evaluator will pick the x with the minimum f_s(x), 
# which is exactly the Thompson Sampling selection.
# 
# To ensure stability and avoid issues with extreme sigma values, we can normalize sigma or use a fixed scale.
# The GP provides mu and sigma in objective units.
# Standard TS: sample y ~ N(mu, sigma^2).
# We can do this vectorized:
# z = np.random.randn(2000)
# sampled_values = mu + sigma * z
# scores = -sampled_values
# 
# This is a pure exploration strategy that balances exploitation (low mu) and exploration (high sigma) naturally.
# It is known to perform very well on multimodal functions like Branin.
# 
# One potential issue: random seed. The evaluator might set the numpy seed.
# We should rely on the global numpy random state.
# 
# Let's add a small exploitation bias if we want, but pure TS is a strong baseline.
# The previous program used UCB with a decaying beta.
# TS is structurally different: it samples a function and picks the minimum.
# 
# We need to be careful about the "higher meaning evaluate next" constraint.
# So we return -sampled_values.
# 
# Is there a risk of timeout?
# np.random.randn(2000) is very fast.
# 
# Let's consider if we need to handle the "call count" for anything.
# Pure TS doesn't strictly need it, but sometimes a slight shift in variance or mean can help.
# However, the directive says "Make a STRUCTURAL change". Pure TS is a clean structural change from UCB.
# 
# Let's implement pure Thompson Sampling.

def acquisition(mu, sigma, f_best):
    # Ensure sigma is positive and non-zero to avoid division issues if we were doing something else,
    # but here we just multiply.
    # Clip sigma to a minimum value to prevent numerical issues if sigma is 0 or negative due to float errors.
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Sample from the posterior
    # z ~ N(0, 1)
    z = np.random.randn(len(mu))
    
    # Sampled function values
    sampled_f = mu + sigma_safe * z
    
    # We want to minimize f, so we maximize the score.
    # Score = -sampled_f
    scores = -sampled_f
    
    # Ensure scores are finite
    # If mu or sigma are inf/nan, this will propagate.
    # The GP should provide finite values, but let's be safe.
    # If any are not finite, replace with a very negative score (so they are not picked)
    # or a very positive score? 
    # If mu is inf, sampled_f is inf, score is -inf. That's good, it won't be picked.
    # If mu is nan, score is nan. The evaluator might fail on nan.
    # Let's replace non-finite scores with a large negative number (low priority).
    scores = np.where(np.isfinite(scores), scores, -1e9)
    
    return scores
# EVOLVE-BLOCK-END
