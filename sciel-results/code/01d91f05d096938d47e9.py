"""Acquisition function: Monte Carlo Knowledge Gradient for ST1D BO."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_global_start_time = None
_call_count = 0

def _ei_scores(mu, sigma, f_best):
    """Compute Expected Improvement scores (higher is better) for minimization."""
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    z = (f_best - mu) / sigma_safe
    # EI for minimization: E[max(f_best - f(x), 0)]
    return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

def acquisition(mu, sigma, f_best):
    global _global_start_time
    global _call_count
    
    n = len(mu)
    
    # Initialize global start time
    if _global_start_time is None:
        _global_start_time = time.time()
        
    elapsed = time.time() - _global_start_time
    
    # Deadline guard: 330s total budget. 
    # We reserve 10s for safety. If we are already close to the limit, 
    # or if this is a very late call, we use fast EI.
    # Also, if we have made many calls, the cost of MC-KG might accumulate.
    # Let's say if elapsed > 300s, we definitely switch to EI.
    # Also, if n is small (unlikely here, n=2000), it's fast.
    
    # Heuristic: 
    # MC-KG with S samples takes roughly O(S * n) operations.
    # With S=100 and n=2000, that's 200k ops per call. Very fast.
    # Even S=1000 is 2M ops, ~10-20ms in NumPy.
    # So we can afford a decent number of samples.
    
    # However, the prompt says "Keep an EXPLICIT deadline guard".
    # If elapsed > 310s, return EI.
    if elapsed > 310.0:
        return _ei_scores(mu, sigma, f_best)

    # Knowledge Gradient Approximation
    # We want to estimate E[ min_x f(x) | x ] - min_x E[f(x)|x] ? 
    # Actually, KG for minimization is:
    # KG(x) = E[ min(f(x), f_best) ] - min( E[f(x)], f_best ) 
    # Wait, the standard KG for minimizing f is:
    # KG(x) = E[ min_{x' in D} f(x') ] - min_{x' in D} E[f(x')]
    # Where D includes the new point x and the current best point?
    # No, the standard definition for the next evaluation point x is:
    # KG(x) = E[ min(f(x), f_best) ] - min( E[f(x)], f_best )
    # This measures the expected improvement of the *minimum observed value*.
    
    # Let's use the simpler "Expected Improvement of the Best" (which is effectively KG in 1D with 1 sample per candidate if we just look at the best).
    # But true KG samples the whole posterior.
    
    # Simplified MC-KG:
    # For each candidate i, we sample S times from N(mu_i, sigma_i).
    # We keep the minimum value seen for that candidate? No.
    # The value of evaluating x is the expected reduction in the global minimum.
    # Current global min estimate: f_best.
    # After evaluating x, the new global min is min(f_best, f(x)).
    # So the value is E[ min(f_best, f(x)) ] ? No, that's just the expected value of the new min.
    # The *improvement* is f_best - E[ min(f_best, f(x)) ].
    # This is exactly Expected Improvement (EI)!
    
    # Wait. Why is KG different?
    # Because in BO, we don't just care about the immediate improvement of the min.
    # We care about the expected reduction in the *regret* after the evaluation.
    # If we evaluate x, and f(x) > f_best, the min stays f_best.
    # If f(x) < f_best, the min becomes f(x).
    # So the expected new min is E[min(f_best, f(x))].
    # The expected regret after evaluation is E[min(f_best, f(x))] - f_true_min.
    # The current expected regret is f_best - f_true_min.
    # The KG is the difference: (f_best - f_true_min) - E[min(f_best, f(x)) - f_true_min]
    # = f_best - E[min(f_best, f(x))]
    # = E[f_best - min(f_best, f(x))]
    # = E[max(f_best - f(x), 0)]
    # = EI.
    
    # So in 1D, with a fixed f_best, EI and KG are mathematically equivalent for the *immediate* step.
    # The difference arises when we consider that the evaluation updates the GP, which changes the posterior for *future* points.
    # Standard EI ignores the update of the posterior for other points.
    # KG (or more accurately, the "Lookahead" or "Batch" EI) considers that the information gained helps future evaluations.
    
    # However, for a single-step acquisition, EI is often sufficient and much faster.
    # The current program uses a blend of UCB and EI.
    # The inspiration 3 suggests "Knowledge Gradient ... using a large number of Monte Carlo samples".
    # Perhaps they mean a *batch* KG or a KG that accounts for the uncertainty in the location of the global minimum?
    
    # Let's try a different structural change: **Thompson Sampling**.
    # Thompson Sampling samples a function from the posterior and picks the minimum.
    # This is a valid and often superior strategy to EI/UCB, especially for multimodal functions like Styblinski-Tang.
    # It naturally balances exploration and exploitation without tuning beta.
    
    # Strategy: 
    # 1. Sample S functions from the GP posterior.
    # 2. For each sample, find the candidate with the minimum value.
    # 3. The score for each candidate is the frequency with which it was the minimum (or the average rank).
    # Actually, for a single evaluation, we just want the candidate that is the minimum in the *most probable* scenario?
    # No, we want the candidate that minimizes the expected regret.
    # Thompson Sampling selects x* = argmin_x f_sample(x).
    # But we need to return scores for ALL 2000 candidates so the evaluator can pick the max.
    # So we can compute a score for each candidate based on its performance in the sampled functions.
    # Score_i = - mean( rank_i in each sample )? 
    # Or Score_i = frequency of being the minimum?
    # If we use frequency, it's a "vote" for the best candidate.
    # This is a valid acquisition function.
    
    # Let's implement Thompson Sampling with a "soft" score.
    # For each sample k:
    #   f_k = mu + sigma * z_k  (where z_k is a vector of standard normals)
    #   min_idx_k = argmin(f_k)
    #   We can accumulate a score for min_idx_k.
    # But we need a score for *every* candidate, not just the winner.
    # A common approach is to use the expected minimum value of the function *if* we were to evaluate at x_i?
    # No, that's EI again.
    
    # Let's stick to the **Knowledge Gradient** interpretation from Inspiration 3.
    # Maybe they mean **Batch Knowledge Gradient** or a more complex MC estimate.
    # But since we only pick 1 point, EI is the exact 1-step KG.
    
    # Is there a different strategy?
    # **Probability of Improvement (PI)**? Less robust.
    # **Entropy-based acquisition**? No.
    
    # Let's look at the "Stagnation" hint.
    # "Make a STRUCTURAL change: a different design or modelling family..."
    # The current program uses a time-based blend.
    # A structural change could be **Bayesian Optimization with a different surrogate model**? No, the GP is fixed.
    # A structural change in the *acquisition function* family.
    
    # How about **Upper Confidence Bound (UCB) with a specific adaptive beta** that is not time-based but *regret-based*?
    # Or **Expected Hypervolume Improvement**? No, single objective.
    
    # Let's try **Thompson Sampling** properly.
    # We can't just return the winner. We need to return scores.
    # We can define the score of x_i as the probability that x_i is the global minimum *among the sampled functions*.
    # P(x_i is min) = (1/S) * sum_k [ min(f_k) == f_k[i] ]
    # This is a discrete score. It might be sparse (only a few non-zero).
    # To make it smooth, we can use a "soft" min or a rank-based average.
    # Score_i = (1/S) * sum_k [ 1 / (1 + rank_i(k)) ]
    # This rewards candidates that are often near the minimum.
    
    # Let's implement this "Thompson Sampling Rank Score".
    # It is structurally different from EI (which is analytic) and UCB (which is linear in mu/sigma).
    # It is non-linear and captures the full posterior distribution's shape.
    
    # Parameters:
    # S = 100 samples.
    # n = 2000.
    # Cost: 100 * 2000 = 200,000 random numbers and operations. Very fast.
    
    S = 100
    # Generate S x n matrix of standard normals
    # z shape (S, n)
    z = np.random.standard_normal((S, n))
    # f_samples shape (S, n)
    f_samples = mu[None, :] + sigma[None, :] * z
    
    # For each sample, compute ranks (0 is best/min)
    # We want to invert ranks so 0 is best.
    # argsort gives indices that would sort the array.
    # rank[i] = number of elements smaller than f[i].
    # np.argsort(np.argsort(f_samples, axis=1), axis=1) gives ranks 0..n-1 where 0 is min.
    ranks = np.argsort(np.argsort(f_samples, axis=1), axis=1)
    
    # Score: Inverse rank. Best (rank 0) gets 1. Worst (rank n-1) gets 1/n.
    # This creates a smooth gradient towards the minima of the sampled functions.
    scores = np.mean(1.0 / (1.0 + ranks), axis=0)
    
    # This score is higher for candidates that are frequently near the minimum of the sampled GP functions.
    # This effectively implements a form of Thompson Sampling that returns a score for all candidates.
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores

# EVOLVE-BLOCK-END
