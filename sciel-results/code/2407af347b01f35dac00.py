"""Thompson Sampling acquisition: ranks candidates by a single draw from the posterior."""
# EVOLVE-BLOCK-START
import time
import numpy as np

# Global start time for the deadline guard (set on first call).
_START_TIME = None
# Hard wall-clock limit (seconds) for the whole evaluation.
_HARD_LIMIT = 330.0
# Safety margin so we return well before the limit.
_MARGIN = 5.0


def _elapsed():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return time.time() - _START_TIME


def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Ackley 2D using Thompson Sampling.
    
    Strategy:
    Instead of computing a deterministic score (like UCB or EI) that balances exploration
    and exploitation via a fixed coefficient, we use Thompson Sampling.
    
    For each candidate i, we draw a single sample z_i from the posterior distribution:
    z_i ~ N(mu_i, sigma_i^2)
    
    We then return these samples as the scores. The candidate with the lowest sample
    (since we are minimizing) is selected.
    
    Why this is structurally different:
    1. It is stochastic: The "best" candidate can change between calls even if the GP
       model doesn't change, because the samples change. This prevents the optimizer
       from getting stuck in a local exploitation loop where deterministic scores
       (like EI) might repeatedly pick the same near-best point.
    2. It naturally balances exploration: High sigma leads to a wider distribution of
       samples, increasing the chance that a high-uncertainty point gets a very low
       (good) sample, thus being selected.
    3. It requires no tuning of exploration parameters (like beta in UCB).
    
    Implementation details:
    - We use the standard normal distribution CDF/PPF or just direct sampling.
    - Since we need to return scores such that "higher meaning evaluate next" is NOT the
      case here? Wait, the contract says: "Return 2000 finite scores, higher meaning 
      evaluate next".
      
      IMPORTANT: The contract says "higher meaning evaluate next".
      However, we are MINIMIZING Ackley.
      So, a "good" candidate has a LOW function value.
      If we return the sample value `z_i`, and the evaluator picks the one with the 
      HIGHEST score, it would pick the WORST candidate (highest function value).
      
      Therefore, we must return scores such that the BEST candidate (lowest expected 
      function value) has the HIGHEST score.
      
      If we sample `z_i` from `N(mu_i, sigma_i)`, the candidates with low `z_i` are 
      good.
      To make the "best" (lowest `z_i`) have the "highest" score, we should return 
      `-z_i`.
      
      Let's verify:
      Candidate A: mu=0, sigma=0 -> sample z=0. Score = 0.
      Candidate B: mu=10, sigma=0 -> sample z=10. Score = -10.
      Evaluator picks max score: A (0 > -10). Correct, A is better (lower function value).
      
      Candidate C: mu=5, sigma=10 -> sample z might be -5. Score = 5.
      Candidate A: score 0.
      Evaluator picks C (5 > 0). This is the exploration effect. If C's sample is very 
      low, it gets a high score and is explored.
      
      So, the score to return is `-sample`.
      
    Deadline Guard:
    If time is running out, we fall back to a simple UCB or just -mu to ensure we 
    return something valid and fast.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Guard: ensure finite inputs
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.maximum(sigma, 1e-9)

    # Deadline guard
    if _elapsed() > _HARD_LIMIT - _MARGIN:
        # Fallback to simple UCB-like score (higher is better for selection)
        # UCB for minimization: -mu + beta * sigma
        # Here we return the score directly.
        scores = -mu + 1.0 * sigma
        return np.where(np.isfinite(scores), scores, 0.0)

    # Thompson Sampling: Draw one sample from each posterior
    # z_i ~ N(mu_i, sigma_i^2)
    # We use numpy's random normal. Note: this uses the global RNG state.
    # This is acceptable for the task.
    samples = mu + sigma * np.random.randn(mu.shape[0])
    
    # We want to MINIMIZE the function.
    # The evaluator picks the candidate with the HIGHEST score.
    # So we return the negative of the sample.
    # Low sample (good function value) -> High score (selected).
    scores = -samples
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
