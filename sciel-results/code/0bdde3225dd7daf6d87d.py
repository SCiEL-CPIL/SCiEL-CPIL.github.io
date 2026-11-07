"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def acquisition(mu, sigma, f_best):
    # Thompson Sampling: Draw a single sample from the posterior distribution.
    # The evaluator selects the candidate with the highest sample value (minimizing the objective).
    # This naturally balances exploration (high sigma -> high variance in sample) 
    # and exploitation (low mu -> low sample mean).
    # We add a small epsilon to sigma to avoid division by zero or numerical issues, 
    # though sigma is std dev so it should be positive.
    # The sample is mu - sigma * Z, where Z ~ N(0, 1).
    # Wait, standard TS for minimization is usually: sample f ~ N(mu, sigma^2), pick min sample.
    # Or sample f ~ N(mu, sigma^2), pick max sample if maximizing.
    # Here we minimize. So we want the candidate with the lowest predicted value.
    # However, the task says "higher meaning evaluate next".
    # So we should return a score such that higher score = better candidate.
    # If we sample from the posterior of the objective f, and we want to minimize f,
    # the "best" candidate in the sample is the one with the lowest value.
    # But the return value must be such that higher is better.
    # So we can return -sample, or we can sample from the posterior and return the sample, 
    # but the evaluator picks the MAX of the returned scores.
    # If we return the sample value s ~ N(mu, sigma^2), and the evaluator picks max(s),
    # it will pick the candidate with the highest predicted value, which is bad for minimization.
    # Therefore, we should return -sample, so that max(-sample) corresponds to min(sample).
    
    # Alternatively, we can think of it as sampling the "regret" or "improvement".
    # Let's stick to the standard TS for minimization:
    # 1. Sample f_i ~ N(mu_i, sigma_i^2) for each candidate i.
    # 2. Select candidate j = argmin_i f_i.
    # 3. To fit the "higher score is better" interface, return score_i = -f_i.
    # Then max(score_i) = -min(f_i), which corresponds to the candidate with min(f_i).
    
    # Let's verify the sign.
    # Task: "Return 2000 finite scores, higher meaning evaluate next"
    # Evaluator: "evaluates the candidate with the highest score"
    # Objective: Minimize beale 2d.
    # So we want the candidate with the lowest expected value (or sampled value) to have the highest score.
    # Score = -Sample is correct.
    
    # Generate standard normal random numbers
    # Use a fixed seed? No, TS relies on randomness. 
    # However, if the function is called multiple times in a row for the same mu/sigma (which shouldn't happen in the loop described: "at each step draws 2000 fresh uniform candidates... calls acquisition... evaluates..."), 
    # the randomness is fine.
    # The loop is:
    # for step in range(30):
    #   generate 2000 candidates
    #   compute mu, sigma
    #   scores = acquisition(mu, sigma, f_best)
    #   best_idx = argmax(scores)
    #   evaluate candidate[best_idx]
    #   update f_best
    
    # So each call to acquisition gets a fresh set of mu/sigma (from new candidates).
    # The randomness in the sample provides the exploration.
    
    z = np.random.randn(mu.shape[0])
    samples = mu - sigma * z  # Sample from N(mu, sigma^2)
    scores = -samples          # Higher score means lower sample value (better for minimization)
    
    return scores
# EVOLVE-BLOCK-END
