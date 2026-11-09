import numpy as np
from scipy.stats import norm
import time

# Global state
_start_time = None
_step_counter = 0
_last_f_best = None

def acquisition(mu, sigma, f_best):
    global _start_time, _step_counter, _last_f_best
    
    if _start_time is None:
        _start_time = time.time()
    
    # Detect new seed/run
    if _last_f_best is not None and f_best > _last_f_best + 0.5:
        _step_counter = 0
    else:
        _step_counter += 1
        
    _last_f_best = f_best
    
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 320:
        return -mu

    step = _step_counter
    if step < 0:
        step = 0
        
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Strategy: Stochastic Search via Thompson Sampling (TS)
    # Unlike the current deterministic EI/Phase-based approach, TS introduces
    # controlled randomness by sampling a function from the posterior distribution.
    # This allows the search to escape local optima more effectively by occasionally
    # selecting points that appear suboptimal under the mean but have high variance.
    # 
    # The "structural change" is moving from a deterministic acquisition function
    # (EI, UCB, Phase-based) to a stochastic sampling-based method (Thompson Sampling).
    # This is a different optimization paradigm:
    # - EI/UCB: Maximize a deterministic score based on mean and std.
    # - TS: Sample a value from N(mu, sigma) and maximize that sample.
    #
    # To make TS effective in a single-call setting where we return scores for 2000 points,
    # we can simulate the TS decision process. However, since the evaluator picks argmax(scores),
    # we need to return scores that reflect the TS preference.
    #
    # A common trick for TS in this constrained interface (returning scores) is to use a
    # randomized acquisition function. But we must be careful: if we randomize the scores,
    # we might pick a bad point.
    #
    # A better structural change: **Bayesian Optimization with a Randomized Restart/Reset**.
    # Or, **Hybrid EI with Randomized Perturbation**.
    #
    # Let's try a different structural change: **Knowledge Gradient (KG) Approximation**.
    # KG is often superior to EI for multimodal functions because it accounts for the
    # expected reduction in regret over the remaining budget, not just the immediate improvement.
    #
    # However, calculating exact KG is expensive. A common approximation is **Expected Hypervolume Improvement (EHVI)** or **Expected Regret (ER)**.
    #
    # Another strong candidate: **Probability of Improvement (PI) with a Dynamic Threshold**.
    # PI is simpler than EI and sometimes performs better in practice because it doesn't penalize large improvements as much as EI does (EI can be "greedy" about the magnitude of improvement).
    #
    # Let's try **PI with a decaying beta**.
    # beta starts high (exploration) and decays to 0 (exploitation).
    # score = Phi((f_best - beta - mu) / sigma)
    #
    # This is structurally different from EI because it uses a probability rather than an expected value.
    #
    # Let's also consider **Variance-based Acquisition** with a twist.
    # Instead of just maximizing sigma, we can maximize sigma * (f_best - mu) / (sigma + epsilon).
    # This is similar to UCB but with a different weighting.
    #
    # Let's go with **PI with a non-monotonic beta schedule**.
    # This is a different design family (probability-based vs expectation-based).
    
    # Beta schedule:
    # Steps 0-5: beta = 0.5 (High exploration)
    # Steps 6-15: beta = 0.1 (Moderate exploration)
    # Steps 16-24: beta = 0.0 (Exploitation)
    
    if step <= 5:
        beta = 0.5
    elif step <= 15:
        beta = 0.1
    else:
        beta = 0.0
        
    # Calculate PI
    Z = (f_best - beta - mu) / sigma_safe
    scores = norm.cdf(Z)
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
    
    return scores
