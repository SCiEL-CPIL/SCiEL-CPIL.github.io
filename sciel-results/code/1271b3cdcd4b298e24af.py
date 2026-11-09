"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Thompson Sampling acquisition function.
    
    Strategy:
    1. Instead of scoring each candidate independently (like PI or EI), we sample
       a single function realization from the posterior distribution of the 2000
       candidates.
    2. We use the Cholesky decomposition of the covariance matrix implied by mu and sigma.
       Note: The input `mu` and `sigma` are the mean and std of the GP posterior at the 2000 candidates.
       We assume the covariance between candidates is derived from the GP kernel.
       However, the contract says "mu, sigma and cost have shape (2000,). Return 2000 finite scores".
       It does NOT pass the full covariance matrix. It only passes the marginal mean and std.
       
       Wait, the contract says: "GP: Matern-5/2... mu and sigma (a standard deviation) are in objective units."
       It does NOT provide the cross-covariances K_ff.
       
       If we only have marginals (mu, sigma), we cannot do full joint Thompson Sampling directly 
       unless we assume a specific correlation structure or approximate it.
       
       However, a common approximation when only marginals are available but we know the GP structure 
       is to assume that the candidates are drawn from the same GP. 
       But without the kernel matrix K, we cannot compute the joint covariance.
       
       Let's re-read carefully: "mu, sigma and cost have shape (2000,). Return 2000 finite scores".
       "No coordinates are passed."
       
       This is a critical constraint. We do NOT have the coordinates of the 2000 candidates.
       We do NOT have the kernel matrix.
       We ONLY have the marginal posterior mean and standard deviation for each of the 2000 points.
       
       Therefore, we CANNOT perform standard joint Thompson Sampling because we don't know 
       how correlated the candidates are with each other.
       
       Alternative Structural Change:
       Since we can't do joint sampling, we must look at the *distribution* of the marginals.
       
       Current strategy: PI / (1 + cost). This is a pointwise probability.
       
       New Strategy Idea: **Expected Improvement with a Cost-Adjusted Temperature (Softmax-like)**
       Or better: **Knowledge Gradient (KG) approximation**? No, needs coordinates.
       
       Let's look at the "Different Strategy" requirement.
       The current strategy is a greedy pointwise PI.
       
       What if we use **Expected Hypervolume Improvement (EHVI)**? No, single objective.
       
       What if we use a **Bayesian Optimization with a different objective**:
       Instead of maximizing P(f(x) < f_best), we maximize the **Expected Reduction in Uncertainty** 
       or use **Information-Directed Sampling (IDS)**?
       
       Actually, there is a simpler structural change that is valid with only marginals:
       **Posterior Sampling with Independent Approximation (Thompson Sampling with Independence Assumption)**
       
       If we assume the 2000 candidates are independent (which is wrong for a GP, but a structural change 
       from "deterministic score" to "stochastic score"), we can sample a value y_i ~ N(mu_i, sigma_i) 
       for each candidate and select the minimum. But we must return scores for ALL 2000 candidates 
       so the evaluator can pick the best one.
       
       Wait, the evaluator picks the best one based on our returned scores.
       If we return scores, the evaluator picks argmax(scores).
       
       If we use Thompson Sampling, we typically sample one function and pick the argmin of that sample.
       To implement this via the `acquisition` interface which returns scores for all points:
       We can't just return the sampled values because the evaluator picks the MAX.
       We need to return a score such that the MAX score corresponds to the point that would be 
       selected by the Thompson Sampling strategy (i.e., the argmin of the sampled function).
       
       But Thompson Sampling is a stochastic process. We don't know which point will be the argmin 
       until we sample.
       
       However, we can simulate the Thompson Sampling decision:
       1. Sample y_i ~ N(mu_i, sigma_i) for all i.
       2. Identify i* = argmin(y).
       3. Return a score array where score[i*] is very high (e.g., 1.0) and all others are low (e.g., 0.0).
       
       This is a valid structural change! It changes the selection mechanism from "greedy pointwise score" 
       to "stochastic global selection via sampling".
       
       How to handle cost?
       The task is "Cost-aware". 
       In standard TS, cost is ignored. 
       We can modify the sampling to be cost-aware.
       
       Modified Cost-Aware Thompson Sampling:
       Sample y_i ~ N(mu_i, sigma_i).
       We want to minimize y_i but also account for cost.
       We can define a "cost-adjusted sample" z_i = y_i + lambda * cost_i.
       Then select i* = argmin(z_i).
       
       What is lambda?
       The cost is in [1, 2] (since 1 + mean(x_scaled) where x_scaled in [0,1] -> cost in [1,2]).
       The objective f(x) for Rosenbrock can be large.
       We need to scale lambda appropriately.
       
       Let's look at the scale of f_best.
       Initially, f_best might be large.
       As we converge, f_best approaches 0.
       
       A robust way to set lambda is to normalize the cost term relative to the uncertainty or the current best.
       
       Let's try:
       z_i = y_i + (cost_i - 1.0) * scale_factor
       
       If we want to be cost-aware, we should penalize higher costs.
       
       Let's refine the "Structural Change":
       The current program uses a deterministic, local score (PI).
       The new program uses a stochastic, global selection mechanism (Thompson Sampling) 
       implemented by returning a one-hot vector (or peaked distribution) indicating the single 
       best candidate according to a cost-adjusted posterior sample.
       
       This is fundamentally different from PI/EI/LCB which assign a continuous score to every point 
       and let the evaluator pick the max. Here, we are effectively saying "This specific point is the 
       best one to evaluate right now" by giving it a huge score and others a tiny score.
       
       Implementation Details:
       1. Check deadline.
       2. Generate standard normal noise e ~ N(0, 1) of shape (2000,).
       3. Sample y = mu + sigma * e.
       4. Compute cost penalty. 
          Cost is roughly [1, 2]. 
          We want to balance f(x) and cost.
          Let's use a dynamic lambda. 
          If f_best is large, we care more about f. If f_best is small, cost might matter more 
          to squeeze out the last bit of improvement efficiently? 
          Actually, cost is fixed per point. 
          
          Let's normalize the cost term.
          cost_range = 1.0 (from 1 to 2).
          Let's define the score for the selection as:
          val_i = y_i + lambda * (cost_i - 1.0)
          
          We need to choose lambda.
          A common heuristic is lambda = sigma_mean / cost_range or similar.
          Or simply, since we are minimizing y_i, we can just add a small penalty.
          
          Let's look at the "seed/value" scoring.
          Value is simple regret.
          
          Let's try a fixed lambda first, or one scaled by the current uncertainty.
          lambda = np.mean(sigma) * 0.1 ?
          
          Actually, let's look at the range of y.
          y is in objective units.
          Cost is in [1, 2].
          If objective values are ~100, cost is negligible.
          If objective values are ~0.01, cost is huge.
          
          We should scale the cost term by the scale of the objective.
          Let scale = np.mean(sigma) + 1e-6.
          lambda = scale / 1.0  # Since cost range is 1.
          
          So, adjusted_sample_i = y_i + scale * (cost_i - 1.0)
          
          5. Find idx_min = argmin(adjusted_sample).
          6. Create score array.
             score = np.zeros(2000)
             score[idx_min] = 1.0
             # To ensure "finite scores" and avoid ties or issues, we can add a tiny noise or 
             # just return 1.0 for the chosen one and 0.0 for others.
             # The evaluator picks the max. 1.0 > 0.0.
             
       This is a valid structural change: from "Continuous Greedy Scoring" to "Stochastic Single-Point Selection via Posterior Sampling".
       
       Why might this be better?
       PI/EI are greedy. They can get stuck in local optima if the local improvement is small but certain.
       TS explores by sampling. It might pick a high-uncertainty point that leads to a global breakthrough.
       
       One risk: If the GP is confident (sigma small) and mu is high everywhere, TS might pick a random point.
       But that's what exploration is.
       
       Let's implement this.
    """
    start_time = time.time()
    
    # Safety check for input shapes
    n = len(mu)
    
    # 1. Generate sample
    # Use a fixed seed? No, the function is called in a loop. 
    # The evaluator handles the randomness of the instances. 
    # We should use random sampling to explore.
    rng = np.random.default_rng()
    
    # Clip sigma to avoid division by zero or huge values
    sigma_safe = np.clip(sigma, 1e-9, None)
    
    # Sample from the posterior
    # y ~ N(mu, sigma^2)
    noise = rng.standard_normal(n)
    y_sample = mu + sigma_safe * noise
    
    # 2. Cost-aware adjustment
    # Cost is in [1, 2]. 
    # We want to minimize y_sample + penalty.
    # Penalty should be scaled to the magnitude of the objective function's uncertainty.
    # If the function values are large, a cost difference of 1 is small.
    # If the function values are small (near 0), a cost difference of 1 is large.
    
    # Estimate the scale of the objective
    # We can use the mean uncertainty as a proxy for the scale of potential improvements.
    scale = np.mean(sigma_safe)
    
    # If scale is very small, we are confident. 
    # If scale is large, we are uncertain.
    
    # Let's define the cost penalty weight.
    # We want the cost term to be comparable to the uncertainty term.
    # Max cost diff is 1.0 (from 1.0 to 2.0).
    # So penalty = scale * (cost - 1.0)
    
    # This means a point with cost 2.0 is penalized by `scale` units of the objective.
    # This is a reasonable heuristic: "How much objective value is worth 1 unit of cost?"
    # Answer: The current average uncertainty.
    
    penalty = scale * (cost - 1.0)
    
    adjusted_sample = y_sample + penalty
    
    # 3. Select the best (minimum) adjusted sample
    idx_min = np.argmin(adjusted_sample)
    
    # 4. Construct the score array
    # The evaluator picks the max score.
    # We want it to pick idx_min.
    scores = np.zeros(n, dtype=np.float64)
    scores[idx_min] = 1.0
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Deadline guard
    if time.time() - start_time > 320:
        # If we are too slow (unlikely for this simple op, but just in case),
        # return the current scores.
        pass
        
    return scores
# EVOLVE-BLOCK-END
