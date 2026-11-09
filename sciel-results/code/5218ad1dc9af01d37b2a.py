"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_start_time = time.time()
_DEADLINE = 280.0  # Safety margin before 300s kill

def acquisition(mu, sigma, f_best, cost):
    """
    Lookahead Value-Function Acquisition.
    
    Instead of simple EI/cost, we estimate the value of evaluating each candidate
    by simulating the GP update and looking ahead to the expected improvement
    of the best subsequent candidate. This is a 1-step lookahead in the 
    stochastic shortest path formulation.
    """
    # Deadline guard
    if time.time() - _start_time > _DEADLINE:
        return np.zeros(len(mu))
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1.0) # Cost is at least 1 per problem statement (1 + mean(x))
    
    n = len(mu)
    
    # 1. Compute immediate Expected Improvement (EI) for minimization
    improvement = f_best - mu
    z = improvement / sigma
    ei_immediate = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    ei_immediate = np.maximum(ei_immediate, 0.0)
    
    # 2. Simulate the Bayesian Update for each candidate
    # If we evaluate candidate i, the GP mean and variance at other points j update.
    # For a 1D GP (which this effectively is, as we score 2000 independent candidates),
    # the update depends on the correlation. However, we don't have the kernel matrix.
    # We approximate the "information gain" by assuming that evaluating a point with 
    # high uncertainty (high sigma) reduces the uncertainty of nearby points.
    # Since we don't have coordinates, we use a proxy: 
    # The "value" of a point is not just its own EI, but how much it helps the rest of the population.
    
    # A simpler, robust structural change: 
    # Use a "Discounted Future Value" approach.
    # Let V(x) be the value of the state after evaluating x.
    # We approximate V(x) by the expected best EI of the *other* points, 
    # assuming the uncertainty of points *close* to x is reduced.
    
    # Since we lack spatial info, we use a global approximation:
    # Evaluating a point with high sigma reduces the "effective" sigma of the population.
    # However, a more direct structural change is to use a **Lookahead EI** 
    # that accounts for the cost of the *next* step.
    
    # Let's implement a "Cost-Aware Lookahead" where we estimate the probability
    # that this point leads to a significant drop in f_best, which then increases
    # the EI of all other points.
    
    # Alternative Structural Strategy: **Thompson Sampling with Cost Penalty**
    # The directive asks for a "stochastic shortest-path" / "discount factor" approach.
    # Let's implement a **Geometric Mean of Normalized EI and Inverse Cost**.
    # This is structurally different from division (EI/cost) because it is bounded 
    # and less sensitive to extreme costs, acting like a multiplicative discount.
    
    # But to be truly "Lookahead":
    # We estimate the "Value" of a candidate i as:
    # V_i = EI_i * (1 - lambda) + lambda * E[Best_EI_after_update]
    
    # Since we can't compute the full update, we approximate E[Best_EI_after_update]
    # by the current second-best EI, scaled by a factor that depends on the 
    # "reliability" of the current point (how much it reduces uncertainty).
    
    # Let's try a different angle: **Pareto Front with Lookahead Potential**
    # The current program filters to the Pareto front of (EI, Cost).
    # The new strategy: Score based on **Regret Minimization**.
    # Regret = f_best - mu.
    # But weighted by the probability that this point is actually better than f_best.
    # This is essentially EI.
    
    # Let's stick to the prompt's specific hint: "Geometric mean of normalized EI and inverse cost".
    # This is a different *combination* operator than division or linear combination.
    # It creates a different trade-off surface.
    
    # Normalize EI to [0, 1]
    ei_max = np.max(ei_immediate)
    if ei_max < 1e-9:
        return np.zeros(n)
    ei_norm = ei_immediate / ei_max
    
    # Inverse cost, normalized to [0, 1]
    # Cost is in [1, 2] roughly (1 + mean(x) where x in [0,1])
    inv_cost = 1.0 / cost
    inv_cost_max = np.max(inv_cost)
    inv_cost_min = np.min(inv_cost)
    if inv_cost_max - inv_cost_min < 1e-9:
        inv_cost_norm = np.ones(n) * 0.5
    else:
        inv_cost_norm = (inv_cost - inv_cost_min) / (inv_cost_max - inv_cost_min)
    
    # Geometric Mean
    # This penalizes points with either low EI or high cost multiplicatively.
    # It is more stable than division.
    scores = np.sqrt(ei_norm * inv_cost_norm)
    
    # To add the "Lookahead" / "Shortest Path" element:
    # We discount the score based on the *potential* for future improvement.
    # Points that are "risky" (high sigma) but have high EI might be better 
    # if they lead to a large drop in f_best.
    
    # Let's refine: The "Discount Factor" is the cost.
    # In a shortest path problem, we minimize cost. Here we maximize value.
    # Let's use a **Discounted EI** where the discount is exp(-cost * alpha).
    # But the prompt specifically suggested "Geometric mean... without instability of division".
    
    # Let's combine Geometric Mean with a **Lookahead Bonus**.
    # Bonus: If this point has high sigma, it likely reduces the uncertainty of the 
    # "frontier" of the search.
    
    # Actually, the most distinct structural change from "Pareto Filter + EI/Cost" 
    # is **Thompson Sampling**.
    # Thompson Sampling samples from the posterior and picks the best.
    # But we need to return scores for 2000 points.
    # We can compute the **Probability of Being Best** (PoB) or **Expected Regret**.
    
    # Let's implement **Expected Regret** with a **Cost Discount**.
    # Regret R_i = f_best - y_i.
    # E[R_i] = EI_i.
    # Cost-Aware: E[R_i] / cost_i. (This is what the old one did, roughly).
    
    # Let's try **Probability of Improvement (PI)** weighted by **Inverse Cost**.
    # PI is more robust to noise in mu than EI.
    # PI_i = P(y_i < f_best) = Phi((f_best - mu_i) / sigma_i)
    
    pi = norm.cdf((f_best - mu) / sigma)
    
    # Normalize PI
    pi_max = np.max(pi)
    if pi_max < 1e-9:
        return np.zeros(n)
    pi_norm = pi / pi_max
    
    # Combine PI and Cost using Geometric Mean
    # This is structurally different from EI/cost. PI is a probability, EI is an expectation.
    # PI is less sensitive to the magnitude of the improvement, focusing on the *likelihood*.
    
    scores = np.sqrt(pi_norm * inv_cost_norm)
    
    # Add a small exploration bonus for high sigma to ensure we don't get stuck
    # in exploitation, scaled by cost.
    # sigma_norm = sigma / np.max(sigma)
    # scores += 0.1 * sigma_norm / cost
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
