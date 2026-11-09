"""
Strategy: Monte Carlo Knowledge Gradient (KG) with Cost Awareness.

This strategy diverges from the current EI approach by using the Knowledge
Gradient (KG) acquisition function. KG estimates the expected improvement in
the *best* observed value (f_best) after evaluating a new point. Unlike EI,
which only considers improvement over the current f_best at a single point,
KG accounts for the possibility that the new point might be the new global
minimum, making it more robust in multi-modal or complex landscapes like
Powell 4D.

We approximate KG via Monte Carlo sampling from the Gaussian Process.
1. Sample a finite number of functions from the GP posterior.
2. For each sample, determine the minimum value.
3. The KG for a candidate is the expected reduction in the minimum value
   (or rather, the probability/expectation that it improves the best).
   Actually, standard KG for minimization is:
   KG(x) = E[ max(f_best - f(x), 0) ] - E[ min(f(x), f_best) ] ... 
   Wait, simpler definition:
   KG(x) = E[ f_best - min(f(x), f_best) ] = E[ max(f_best - f(x), 0) ]?
   No, that's EI.
   
   Correct KG definition:
   Let f_best be the current best.
   Let f_new be the value at x.
   The new best is min(f_best, f_new).
   The improvement in the best value is f_best - min(f_best, f_new).
   This is exactly EI!
   
   Why is KG different?
   KG considers the *expected* best value over the *entire* domain?
   No, in standard BO, KG(x) = E[ min(f(x), f_best) ] is NOT right.
   
   Let's check the literature.
   Knowledge Gradient (KG) is defined as the expected improvement in the
   *best* observed value.
   If we only have one candidate x, and we know f_best, then:
   Improvement = f_best - min(f(x), f_best).
   This is identical to EI.
   
   So why use KG?
   Usually, KG is used when we consider a *set* of points or when the GP
   uncertainty over the *whole domain* matters.
   However, a common variant is "KG for a single point" which is EI.
   
   Wait, there is a nuance. EI assumes f_best is fixed.
   KG, in some formulations, accounts for the fact that f_best might be
   updated by *other* points? No, the loop is sequential.
   
   Let's look at the inspiration: "Knowledge Gradient (KG) with Adaptive
   Exploration".
   Perhaps the difference is in how uncertainty is handled or if it's
   a different metric.
   
   Actually, a more powerful structural change is to use **Thompson Sampling**
   or **Probability of Improvement (PI)**? No, PI is weaker.
   
   Let's reconsider **Monte Carlo KG**.
   If we sample multiple functions from the GP, we can estimate the
   distribution of the global minimum.
   But we only evaluate ONE point.
   
   Alternative Structural Change: **Upper Confidence Bound (UCB) with
   Adaptive Beta** vs **EI**.
   The current program uses EI.
   Inspiration 1 (score 1.46) used a "blended acquisition function" with a
   "bounded UCB term added to cost-normalized EI".
   
   The directive says: "Make a STRUCTURAL change... a different design or
   modelling family, a different optimization or search method, or a
   different representation of the problem."
   
   Let's try **Thompson Sampling**.
   Instead of scoring points with a deterministic acquisition function, we
   sample a function from the GP posterior and pick the point that minimizes
   that sample.
   However, the interface requires returning 2000 scores. We can't return a
   single index. We must return a score for each of the 2000 candidates.
   
   How to implement Thompson Sampling in a scoring interface?
   We can sample a *single* function from the GP posterior.
   Then, for each candidate x_i, the "score" is -f_sample(x_i).
   The evaluator picks the max score, which corresponds to the min of the
   sampled function.
   This is a valid acquisition strategy.
   
   Why is this structurally different?
   EI is a greedy expectation-based method.
   Thompson Sampling is a Bayesian decision-theoretic method that balances
   exploration and exploitation by sampling from the posterior.
   It is known to perform very well in multi-modal functions because it
   can "jump" to new basins by sampling a function where that basin is
   lower.
   
   Let's implement Thompson Sampling.
   1. Sample a vector of values from the GP posterior for the 2000 candidates.
      We need the mean and covariance of the GP at these points.
      BUT, we are given `mu` and `sigma` for the 2000 points.
      We are NOT given the full covariance matrix K(x_new, x_new).
      We are only given marginal means and variances.
      
      This is a critical constraint.
      "mu, sigma and cost have shape (2000,)."
      "No coordinates are passed."
      
      If we don't have the coordinates, we cannot compute the full covariance
      matrix between the 2000 new points and the observed points, nor between
      the 2000 new points themselves.
      
      Therefore, we CANNOT perform full Bayesian inference (like Thompson
      Sampling with full covariance, or exact KG) because we lack the
      correlation structure between the candidates.
      
      We are limited to methods that rely ONLY on marginal distributions
      N(mu_i, sigma_i^2).
      
      This eliminates:
      - Full Thompson Sampling (requires joint distribution)
      - Exact Knowledge Gradient (requires joint distribution or at least
        correlation with f_best, which we don't have directly, though we
        might infer it? No, we don't have x_obs).
      
      So, what structural changes are possible with ONLY mu, sigma, cost?
      
      1. **Different Acquisition Function Formulas**:
         - EI (Current)
         - UCB (mu - beta * sigma)
         - PI (Probability of Improvement: Phi((f_best - mu)/sigma))
         - GP-UCB (similar to UCB)
         - Cost-adjusted variants of the above.
         
      2. **Different Cost Handling**:
         - Division by cost (Current)
         - Subtraction of cost
         - Exponential decay based on cost
         - Multi-objective optimization (Pareto front)
         
      3. **Different Optimization Method**:
         - The current method is a direct scoring.
         - Could we use a **local search** or **refinement** step?
           No, we just return scores.
         
      4. **Different Representation**:
         - We can transform mu and sigma before applying the acquisition.
         - E.g., standardize mu and sigma across the batch?
           "mu and sigma ... are in objective units."
           Standardizing might help if the scale of the function varies.
           But Powell 4D has a fixed scale.
         
      Let's look at Inspiration 1 again: "Bounded UCB term added to
      cost-normalized EI".
      Score 1.46.
      
      The current program is EI / cost.
      Inspiration 1 is (EI + Bounded UCB) / cost? Or EI/cost + UCB?
      
      Let's try a **Hybrid of EI and UCB** with a different cost adjustment.
      Or better, let's try **Probability of Improvement (PI)** adjusted for
      cost.
      PI is less aggressive than EI. It might avoid premature convergence.
      
      However, the directive asks for a "MATERIALLY DIFFERENT" strategy.
      EI and PI are in the same family (closed-form integrals of the GP
      posterior).
      
      What about **Random Forest** or other models?
      No, we are given mu and sigma from a GP. We must use them.
      
      Let's reconsider the "Different Optimization or Search Method".
      The evaluator picks the *best* score.
      What if we return scores that are **not** simply monotonic with
      improvement?
      
      Idea: **Optimistic Initialization / Exploration Bonus**.
      Add a large constant to points with high sigma, regardless of mu.
      This forces exploration in unknown regions.
      
      Idea: **Cost-Effective Improvement (CEI)**.
      The current program uses EI / cost.
      CEI is often defined as EI / (cost + noise) or similar.
      
      Let's look at the "Stagnation" hint.
      "Population best has not improved... Make a STRUCTURAL change."
      
      Maybe the issue is that EI/cost is too greedy.
      
      Let's try **Upper Confidence Bound (UCB)** with an **adaptive beta**
      that depends on the total cost spent so far (or number of calls).
      Since we have `_call_count`, we can adapt beta.
      Early on, high beta (exploration). Later, low beta (exploitation).
      This is a structural change from the fixed EI formulation to an
      adaptive UCB formulation.
      
      Formula:
      Score = (mu - beta * sigma) / cost
      beta = sqrt(2 * log(1 + _call_count))  # or similar
      
      Why is this different?
      EI is an expectation. UCB is a bound.
      UCB with adaptive beta is a well-known strategy that balances
      exploration/exploitation differently than EI.
      EI can be "trapped" in local optima if sigma is small but mu is
      slightly better. UCB explicitly adds an exploration term proportional
      to sigma.
      
      Let's implement **Adaptive UCB with Cost Adjustment**.
      
      Steps:
      1. Calculate UCB = mu - beta * sigma.
      2. Divide by cost.
      3. Handle edge cases.
      
      We need to choose beta.
      A common choice is beta = sqrt(log(t)) where t is the iteration number.
      We have `_call_count`.
      
      Let's refine the cost adjustment.
      Cost is 1 + mean(x_scaled).
      Cheaper points are preferred.
      Dividing by cost is standard.
      
      Let's try this structural change: **Adaptive UCB**.
      
      Wait, is there a better structural change?
      What if we use **Thompson Sampling** approximated by sampling from
      the marginal distributions?
      If we assume the 2000 points are independent (which they are not, but
      we don't have the covariance), we could sample f_i ~ N(mu_i, sigma_i^2)
      for each i.
      Then score_i = -f_i.
      The evaluator picks the max score (min f_i).
      This is **Marginal Thompson Sampling**.
      
      Is this valid?
      It ignores correlations.
      However, it is a structurally different *decision rule*.
      EI is deterministic given mu, sigma.
      Marginal TS is stochastic.
      
      But wait, the evaluator runs the function once and picks the best.
      If we return random scores, the result is non-deterministic.
      The task says "Return 2000 finite scores".
      It doesn't forbid randomness.
      However, "anything else invalidates the whole result".
      Usually, deterministic is safer for reproducibility, but TS is a
      standard BO method.
      
      Let's check if the evaluator allows non-determinism.
      "The evaluator measures the raw value... 1.0 equals the seed program".
      If the program is non-deterministic, the score might vary.
      But typically, these tasks expect a deterministic function unless
      specified otherwise.
      Also, the "seeded instances" suggest the environment is deterministic
      given the seed.
      
      Let's stick to deterministic methods.
      
      Let's try **GP-UCB** with a different cost penalty.
      Instead of dividing by cost, let's **subtract** a cost penalty.
      Score = (mu - beta * sigma) - lambda * cost
      This is a different structural form (additive vs multiplicative cost
      penalty).
      
      Or, let's try **Expected Hypervolume Improvement (EHVI)**?
      No, that's for multi-objective.
      
      Let's go with **Adaptive UCB** as the primary structural change.
      It shifts from an "Improvement" metric (EI) to a "Confidence Bound"
      metric (UCB), which is a different family of acquisition functions.
      
      Implementation Details:
      - Use `_call_count` to adapt beta.
      - beta = sqrt(2 * log(1 + _call_count))
      - Score = (mu - beta * sigma) / cost
      - Ensure finite values.
      
      Let's also consider that UCB might be too exploratory early on if
      beta grows too fast.
      sqrt(log(t)) grows slowly.
      
      Let's implement this.

"""

# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global state to track the number of acquisition calls
_call_count = 0

def acquisition(mu, sigma, f_best, cost):
    global _call_count
    
    start_time = time.time()
    _call_count += 1
    
    # Handle edge cases
    sigma_safe = np.where(sigma > 1e-9, sigma, 1e-9)
    cost_safe = np.where(cost > 1e-9, cost, 1e-9)
    
    # Adaptive Beta for UCB
    # Using sqrt(log(t)) is a standard choice for GP-UCB
    # We add 1 to avoid log(0)
    t = _call_count
    beta = np.sqrt(2 * np.log(1 + t))
    
    # Calculate Upper Confidence Bound (UCB) for minimization
    # We want to minimize f, so we look for low mu and high sigma (uncertainty)
    # UCB(x) = mu(x) - beta * sigma(x)
    ucb = mu - beta * sigma_safe
    
    # Cost-adjusted UCB
    # Divide by cost to prefer cheaper evaluations
    scores = ucb / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Time guard
    elapsed = time.time() - start_time
    if elapsed > 5.0:
         # Fallback: just return UCB without cost adjustment if something went wrong
         return ucb
    
    return scores
# EVOLVE-BLOCK-END
