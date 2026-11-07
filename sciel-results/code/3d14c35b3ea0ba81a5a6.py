"""Cost-aware acquisition function for Bayesian optimization (griewank 2d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START = time.time()
_LIMIT = 325.0  # Safety margin before the 330 s wall-clock kill

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition using Monte Carlo approximation of Expected Improvement
    with a cost-aware penalty.
    
    Instead of just EI/cost, we sample from the GP to estimate the improvement
    and then apply a cost penalty that encourages finding low-cost improvements
    but doesn't strictly divide by cost (which can be noisy).
    Actually, the prompt says "higher = evaluate next".
    
    The current seed uses EI / cost.
    A different strategy: 
    1. Compute standard EI.
    2. Compute a "cost-effective" score that is more robust.
    
    Let's try a hybrid: 
    Score = EI * (1 / (1 + cost))
    This dampens high cost but doesn't divide by it, which can be unstable if cost is small.
    
    Alternatively, let's look at the "inspiration 2" which scored 1.0.
    It used "Deterministic EI".
    
    The current program (score 1.0) is basically EI / cost.
    
    To be materially different, let's try **Probability of Improvement (PI) adjusted for cost**.
    PI = Phi((f_best - mu) / sigma).
    Score = PI / cost.
    
    Or **Upper Confidence Bound (UCB) adjusted for cost**.
    UCB = mu - kappa * sigma.
    Score = UCB / cost.
    
    Let's try a **Bayesian Optimization with Cost (BOC)** approach where we maximize
    the expected improvement per unit cost, but using a more stable formulation.
    
    Actually, the most robust "different" strategy is often **Knowledge Gradient (KG)**,
    but that is expensive.
    
    Let's try **Expected Improvement with a Cost Penalty** where the penalty is 
    proportional to the cost, but we normalize the EI by the standard deviation of the GP 
    to make it scale-invariant, then divide by cost.
    
    Wait, the simplest material change that often improves BO in tight budgets is 
    **using a lower f_best (e.g., the 95th percentile of observed values) to encourage exploration**
    when the current best is stuck. But we don't have the history, only f_best.
    
    Let's try **UCB with dynamic kappa**.
    kappa = 1.0 is standard.
    
    Let's try **EI with a floor on the improvement**.
    
    Actually, let's look at the "inspiration 3" (0.9345). It used time.
    
    Let's implement **Cost-Effective EI** but with a **Monte Carlo estimate of the improvement**
    to avoid numerical issues with the normal CDF/PDF in extreme tails, and to allow for 
    a more flexible cost model.
    
    However, MC is slow. We have 2000 points. 2000 * 100 samples = 200k ops. That's fast.
    
    Strategy:
    1. Sample N=50 values from N(mu, sigma^2) for each point.
    2. Calculate improvement = max(0, f_best - sample).
    3. Mean improvement = EI_approx.
    4. Score = EI_approx / cost.
    
    This is effectively EI / cost but computed via MC. It might be more robust to 
    numerical errors in scipy.stats.norm if the values are extreme.
    
    Let's try a different angle: **Inverse Cost Scaling**.
    Score = EI * (1 / cost^0.5).
    This is less aggressive than EI / cost.
    
    Let's try **UCB / cost**.
    UCB = mu - 1.0 * sigma.
    Score = UCB / cost.
    
    Let's implement UCB / cost as it is a materially different acquisition function 
    (optimistic vs expected).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure shapes are correct and inputs are safe
    cost = np.maximum(cost, 1e-6)
    sigma = np.maximum(sigma, 1e-9)
    
    # Deadline guard: fall back to a fast single-variant score
    if time.time() - _START > _LIMIT:
        # Fallback: UCB / cost
        z = np.clip((f_best - mu) / sigma, -10.0, 10.0)
        # UCB: mu - kappa * sigma
        # For minimization, we want low mu and high sigma (uncertainty).
        # UCB = mu - kappa * sigma
        kappa = 1.0
        ucb = mu - kappa * sigma
        scores = ucb / cost
        return np.where(np.isfinite(scores), scores, 0.0)

    # Strategy: Upper Confidence Bound (UCB) divided by cost.
    # UCB is more exploratory than EI.
    # In tight budgets, exploration might be key to escaping local minima 
    # if the GP is not perfectly calibrated.
    
    # Dynamic kappa: 
    # If we are early (low total cost), explore more.
    # We don't know the total cost directly, but we can estimate it from the mean cost.
    # Or just use a fixed kappa that is slightly higher than standard.
    
    kappa = 1.5  # More exploration
    
    ucb = mu - kappa * sigma
    
    # Normalize by cost
    scores = ucb / cost
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
