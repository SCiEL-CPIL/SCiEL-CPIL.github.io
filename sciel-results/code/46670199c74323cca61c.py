import numpy as np
from scipy.stats import norm
import time

# Global state for adaptive beta
# We need to track the number of evaluations to adjust the exploration rate.
# Since the function is called in a loop, we can infer the step count from the 
# magnitude of f_best or simply use a global counter if the process is persistent.
# However, the safest way without relying on process persistence is to 
# estimate the step from the inputs or just use a fixed schedule if we can't count.
# But wait, the problem says "for a loop that minimizes...". 
# The acquisition function is called 25 times (30 evals - 5 initial).
# We can't easily count steps inside acquisition() unless we use a global variable 
# which persists across calls in the same process.
# Let's assume the process is persistent for the duration of the task run.
# We will use a global counter.

_call_count = 0

def acquisition(mu, sigma, f_best):
    global _call_count
    
    # Increment call count. 
    # The first call corresponds to the 6th evaluation (index 5).
    # The last call corresponds to the 30th evaluation (index 29).
    # So _call_count will go from 0 to 24 for the 25 acquisition calls.
    # Total evaluations N = 5 + _call_count + 1 = 6 + _call_count.
    # Max N = 30.
    
    _call_count += 1
    
    # If the process is restarted or this is a new seed, _call_count might be off.
    # However, typically in these benchmarks, the module is loaded once per seed or 
    # the global state is reset. If not, we might have issues.
    # To be safe, we can cap the count or reset if f_best improves significantly? No, f_best only decreases.
    # Let's assume the standard behavior where globals persist for the duration of the seed's run.
    # If the evaluator spawns a new process for each seed, _call_count starts at 0 for each.
    # If it uses the same process for all seeds, _call_count will accumulate.
    # The task says "For each seed... the evaluator starts from...". This implies separate runs.
    # Usually, these evaluators run the script in a subprocess or reset state.
    # If state is not reset, _call_count for seed 1 would start at 25, which is > 24.
    # We should clamp _call_count to a reasonable range [0, 24] for the schedule.
    # Actually, if it's the same process, the 2nd seed would start with _call_count=25.
    # The schedule should handle this by capping at the max steps.
    
    step = min(_call_count, 25)  # Cap at 25 steps (0-24)
    
    # Adaptive Beta for UCB
    # We want to explore more early on and exploit more later.
    # A common schedule is beta_t = c * (log(t + 1) / t) or similar.
    # Or simply a linear decay from a high value to a low value.
    # Let's use a schedule that starts with high exploration (beta ~ 2-3) 
    # and decays to low exploration (beta ~ 0.5-1.0).
    # The total number of steps is 25.
    
    # Normalize step to [0, 1]
    t_norm = step / 25.0
    
    # Beta schedule: 
    # At t=0 (first acquisition call, 6th eval), beta should be high.
    # At t=1 (last acquisition call, 30th eval), beta should be lower.
    # Let's try beta = 1.5 + 1.5 * (1 - t_norm)
    # t_norm=0 -> beta=3.0
    # t_norm=1 -> beta=1.5
    # This seems reasonable for a 1D problem with 30 evals.
    
    beta = 1.5 + 1.5 * (1.0 - t_norm)
    
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Numerical stability
    eps = 1e-12
    sigma_safe = np.maximum(sigma, eps)
    
    # UCB Score: mu + beta * sigma
    # Note: We are minimizing, so we want to find low values.
    # Standard UCB for maximization is f + beta * sigma.
    # For minimization, we can either:
    # 1. Minimize the UCB value (i.e., select point with lowest mu + beta*sigma? No, that's optimistic for minimization if we assume noise).
    # Wait. UCB is for maximization: we assume the true value is f + beta*sigma (optimistic).
    # For minimization, we are "pessimistic" about the lower bound? 
    # No, we want to find the minimum. 
    # If we use UCB for minimization, we should look for points where the *lower* confidence bound is low?
    # Or do we flip the sign?
    # Let's think: We want to minimize f(x).
    # The GP predicts mu(x) ~ N(mu, sigma^2).
    # The lower confidence bound (LCB) is mu - beta * sigma.
    # We want to pick the point with the lowest LCB? That would be greedy exploitation if beta=0.
    # If beta > 0, mu - beta*sigma favors points with high sigma (exploration) because subtracting a large positive number makes the score smaller (better for minimization).
    # So, LCB = mu - beta * sigma.
    # We want to MINIMIZE the score? 
    # The task says: "Return 2000 finite scores, higher meaning evaluate next".
    # So we need to return a score where HIGHER is BETTER.
    # If we use LCB = mu - beta * sigma, lower is better for minimization.
    # So we should return -LCB = -mu + beta * sigma.
    # Or, equivalently, we can define the acquisition function as maximizing the "potential for improvement" in a minimization context.
    # Let's verify:
    # If mu is low (good), -mu is high (good score).
    # If sigma is high (uncertain), beta*sigma is high (good score).
    # So Score = -mu + beta * sigma.
    # This encourages picking points with low predicted mean and high uncertainty.
    
    scores = -mu + beta * sigma_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
