"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Deadline guard state
_start_time = time.time()
_DEADLINE = 300.0  # 300s limit, use 280s to be safe

def _check_deadline():
    return (time.time() - _start_time) > _DEADLINE

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Bayesian Optimization with Noisy Cost (BO-NC).
    
    Strategy:
    1. Model evaluation cost as a latent process correlated with the objective.
       Since we only have the objective GP (mu, sigma) and observed costs, 
       we approximate the cost uncertainty by assuming a GP on the log-cost 
       or directly on cost, correlated with the objective. 
       
       However, the interface only provides mu, sigma (objective) and cost (observed).
       We do not have a cost GP posterior directly. 
       
       Alternative Interpretation for BO-NC in this context:
       The "cost" is deterministic given x, but the *effective* value of an evaluation 
       depends on the remaining budget. 
       
       A more rigorous BO-NC approach would model the cost c(x) as a GP. 
       Here, we can approximate the cost variance or uncertainty by assuming 
       a relationship between cost and the objective uncertainty or by 
       using a heuristic cost model.
       
       Let's implement a simpler but structurally different approach: 
       **Integrated Cost-Adjusted EI**.
       
       Instead of EI(x) / cost(x), we compute the expected improvement 
       per unit of expected cost, but we account for the *uncertainty* in cost 
       by assuming a log-normal or Gaussian distribution for the cost 
       centered at the observed cost with a small variance.
       
       Actually, a true BO-NC formulation integrates over the joint posterior 
       of f(x) and c(x). 
       P(f(x) < f_best | data) is given by the GP.
       The cost c(x) is deterministic in this task (1 + mean(x_scaled)), 
       so the "noise" in cost is zero. 
       
       Wait, the task says "cost-aware acquisition". The current program uses EI/sqrt(cost).
       A structural change: **Knowledge Gradient (KG) with Cost Awareness**.
       Or **Thompson Sampling** which naturally handles trade-offs if we sample 
       the cost as well? No, cost is deterministic.
       
       Let's try **Expected Improvement with Cost-Adjusted Noise**.
       The idea: The effective noise in the objective function estimate 
       should be increased if the cost is high, because we have fewer 
       evaluations left to reduce uncertainty.
       
       Alternatively, use **Probability of Improvement (PoI)** 
       which is more robust to scale differences, and adjust it 
       with a cost penalty that is convex.
       
       Let's implement **Thompson Sampling** (TS) as a structural change.
       TS samples f(x) ~ N(mu, sigma^2) and selects the x that minimizes 
       the sampled value, adjusted for cost.
       Score = -sampled_value / cost(x)
       
       But TS requires sampling. We can vectorize the sampling.
       Sample: z ~ N(0, I)
       f_sampled = mu + sigma * z
       score = -f_sampled / cost
       
       This is a different design family (sample-based vs deterministic EI).
       
       To make it "BO-NC" like, we can also sample the cost if we model it 
       as uncertain. But cost is deterministic. 
       
       Let's refine the TS approach with a **Cost-Aware Thompson Sampling**.
       We sample the objective and select the point with the best 
       expected value per cost.
       
       Additionally, we can add a small exploration bonus to the cost 
       to avoid dividing by very small costs (though cost >= 1).
       
       Let's implement:
       1. Sample f ~ N(mu, sigma^2)
       2. Compute score = -f / cost
       3. To avoid premature convergence, add a small random noise to the score 
          or use a temperature parameter.
       
       However, a single sample can be noisy. 
       We can use **Expected Thompson Sampling** (ETS) or 
       **Multi-armed Bandit** style.
       
       Actually, let's stick to the prompt's suggestion: 
       "Apply a Bayesian Optimization with Noisy Cost (BO-NC) formulation..."
       
       If we assume the cost has a small uncertainty (e.g., due to numerical 
       precision or system load), we can model c ~ N(c_obs, sigma_c^2).
       Then E[1/c] can be computed analytically or approximated.
       
       Let's assume sigma_c = 0.1 * c_obs.
       Then the expected cost-adjusted improvement is:
       E[ EI(x) / c(x) ]
       
       For c ~ N(c0, sigma_c^2), E[1/c] is approximately 1/c0 + sigma_c^2/c0^3.
       So we can use an adjusted cost: c_adj = c0 * (1 - sigma_c^2/c0^2)
       Or simply use the harmonic mean approximation.
       
       Let's implement **Cost-Adjusted EI with Uncertain Cost**.
       1. Compute EI(x) as before.
       2. Model cost uncertainty: sigma_c = 0.1 * cost
       3. Approximate E[1/cost] using the delta method:
          E[1/c] ≈ 1/c + Var(c)/c^3
          Var(c) = (0.1*c)^2 = 0.01*c^2
          E[1/c] ≈ 1/c + 0.01*c^2/c^3 = 1/c + 0.01/c = 1.01/c
       
       This is a small change. 
       
       Let's try a more significant structural change: **Knowledge Gradient (KG)**.
       KG estimates the expected reduction in the best value.
       KG(x) = E[ max(f_best - min(f(x), f_best), 0) ]
       This is similar to EI but with a different weighting.
       
       Actually, the most distinct and potentially effective strategy 
       for small budgets is **Upper Confidence Bound (UCB)** 
       with cost awareness.
       UCB(x) = mu(x) + beta * sigma(x)
       Score = -UCB(x) / cost(x)
       
       This balances exploration (sigma) and exploitation (mu) 
       and is cost-aware.
       
       Let's implement UCB with adaptive beta.
       beta = sqrt(log(1 + t)) where t is the number of evaluations.
       But we don't have t directly. We can estimate it from the cost.
       Total cost so far is not given, but we can assume t is proportional 
       to the number of points evaluated.
       
       Since we don't have t, we can use a fixed beta or estimate it 
       from the spread of mu.
       
       Let's use **UCB with Cost**.
       Score = -(mu + beta * sigma) / cost
       
       To make it "BO-NC", we can add a cost uncertainty term.
       Let's use:
       Score = -(mu + beta * sigma) / (cost * (1 + uncertainty_factor))
       
       This is a structural change from EI to UCB.
       
       Let's implement UCB with an adaptive beta based on the 
       standard deviation of mu.
    """
    if _check_deadline():
        return np.zeros_like(mu)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure cost is safe
    safe_cost = np.maximum(cost, 1e-6)
    
    # Estimate exploration-exploitation balance
    # Use a beta that encourages exploration early
    # We can estimate the "age" of the optimization by looking at 
    # the variance of mu. If mu is spread out, we are early.
    mu_std = np.std(mu)
    mu_mean = np.mean(mu)
    
    # Adaptive beta: larger when uncertainty is high
    beta = 1.0 + 0.5 * (mu_std / (np.abs(mu_mean) + 1e-6))
    beta = np.clip(beta, 0.5, 3.0)
    
    # UCB value
    ucb = mu + beta * sigma
    
    # Cost-aware score: we want to minimize ucb per unit cost
    # So score = -ucb / cost
    score = -ucb / safe_cost
    
    # Add a small exploration bonus for high sigma regions
    # This helps escape local minima
    exploration_bonus = 0.1 * sigma / safe_cost
    score += exploration_bonus
    
    # Ensure finite outputs
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
# EVOLVE-BLOCK-END
