"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard to ensure we finish well before the 330s limit
# The acquisition function is called 25 times (30 total evals - 5 initial).
# Each call processes 2000 points. This is computationally trivial.
# We add a simple guard just in case of unexpected overhead, though it's likely unnecessary.
_start_time = time.time()
_DEADLINE = 320.0  # 10s safety margin

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling.
    
    Instead of scoring each candidate with a deterministic acquisition function (like EI or UCB),
    Thompson Sampling draws a single sample from the posterior distribution at each candidate location
    and selects the candidate with the lowest sampled value.
    
    For a Gaussian Process posterior, the sample at x is:
    y_sample(x) = mu(x) + sigma(x) * z
    where z is a standard normal random variable.
    
    To make this deterministic in the scoring sense (since we must return scores for all 2000 candidates
    and the evaluator picks the max), we can interpret the "score" as the negative of the sampled value.
    However, standard Thompson Sampling picks one sample. 
    If we return a score for every point, we are effectively doing a "batch" Thompson Sampling or 
    approximating the probability of being the minimum.
    
    A common deterministic approximation for Thompson Sampling in a single-shot selection context 
    (where we must rank all points) is to use the **Lower Confidence Bound (LCB)** with a fixed, 
    moderate kappa, or to simulate the sampling process.
    
    But wait: The task says "Return 2000 finite scores... higher meaning evaluate next".
    This implies we must rank the candidates.
    Standard Thompson Sampling doesn't naturally produce a ranking of all points in a single call 
    unless we interpret it as: "What is the probability that this point is the best?" 
    or "What is the expected improvement over the current best if we sample from the posterior?"
    
    Actually, a more direct structural change from "EI/UCB blend" is to use **Pure Expected Hypervolume Improvement (EHVI)** 
    or **Probability of Improvement (PI)** with a specific tuning, or **Knowledge Gradient**.
    
    However, the most distinct *design family* from "heuristic blend of EI and UCB" is **Thompson Sampling** 
    implemented via **Posterior Sampling**. Since we must return scores, we can use the **Expected Improvement** 
    but derived differently, or we can use the **Lower Bound of the Posterior Mean** adjusted by uncertainty.
    
    Let's reconsider. The current program uses a *time-adaptive* (via f_best) blend.
    A structural change is to use a **fixed, theoretically grounded** acquisition function.
    
    Let's try **Probability of Improvement (PI)** with an infinitesimal improvement parameter xi=0.
    PI(x) = P(f(x) < f_best) = Phi((f_best - mu(x)) / sigma(x))
    
    Why is this different?
    1. It is purely probabilistic, not a linear combination of mean and variance.
    2. It does not use a heuristic kappa that changes with f_best.
    3. It focuses entirely on the probability of beating the current best, ignoring the *magnitude* of improvement.
    This can lead to more aggressive exploration of uncertain regions that have a *chance* of being good, 
    whereas EI favors regions with high *expected* gain.
    
    Alternatively, **Thompson Sampling** can be approximated by returning the negative of the posterior mean 
    plus a random noise term? No, we must return deterministic scores.
    
    Let's stick to **Probability of Improvement (PI)**. It is a classic, distinct acquisition function.
    The current program uses EI (which depends on the magnitude of improvement) and UCB (which is linear).
    PI is non-linear and purely based on the CDF.
    
    Another strong candidate: **Expected Improvement with a fixed, small xi** is still EI.
    
    Let's try **Thompson Sampling** via **Sampled Lower Bound**.
    If we could sample, we'd pick argmin(mu + sigma*z).
    To rank points, we can use the **Expected Minimum** of the posterior? No, that's complex.
    
    Let's go with **Probability of Improvement (PI)** as the structural change.
    It is fundamentally different from the EI/UCB blend in that it does not account for the *size* of the improvement, 
    only the *likelihood*. This often leads to more diverse exploration in early stages and sharp exploitation later, 
    without needing an adaptive kappa.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        # Return a safe fallback: just the negative mean (greedy exploitation)
        return -mu
    
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Probability of Improvement for minimization
    # P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    z = (f_best - mu) / sigma_safe
    
    # Use the standard normal CDF
    # scipy.stats.norm.cdf is available
    from scipy.stats import norm
    pi_scores = norm.cdf(z)
    
    # PI scores are between 0 and 1.
    # Higher PI means higher probability of improving f_best.
    # This is a valid ranking.
    
    # Ensure finite values
    pi_scores = np.where(np.isfinite(pi_scores), pi_scores, 0.0)
    
    return pi_scores
# EVOLVE-BLOCK-END
