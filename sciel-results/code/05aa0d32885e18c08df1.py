"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 330 - 5  # Safety margin: kill program 5s before 330s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware UCB acquisition function.
    
    Uses Upper Confidence Bound (UCB) to balance exploration and exploitation.
    The score is (mu + beta * sigma) / cost, but since we want to MINIMIZE the objective,
    we actually want to find points with LOW mu and HIGH sigma.
    
    Wait, standard UCB for minimization is:
    We want to find the minimum. The GP gives us a lower bound on the function value?
    No, GP gives us a posterior.
    
    Let's rethink the UCB formulation for minimization.
    We want to select the point that has the highest probability of being the new minimum.
    A common heuristic for minimization is to look for low expected values and high uncertainty.
    
    However, the "UCB" in the context of maximizing a reward is $E[R] + \beta \sqrt{Var[R]}$.
    For minimization, we can map this to maximizing "potential improvement" or simply using a 
    lower confidence bound (LCB) on the function value, but inverted?
    
    Actually, a robust approach for minimization is to use the Lower Confidence Bound (LCB) of the 
    function value, but we want to *maximize* the acquisition score.
    
    If we define the acquisition as:
    Score = (mu - beta * sigma) / cost
    This would favor low mu and high sigma (because subtracting a large sigma makes the score 
    lower? No, we want HIGH score).
    
    Let's look at the standard EI for minimization:
    EI(x) = E[max(f_best - f(x), 0)]
    
    UCB for minimization is often formulated as:
    We want to find the minimum. Let's consider the "worst-case" or "best-case" bound.
    Actually, a simpler and very effective alternative to EI is the "Probability of Improvement" (PI) 
    or just a weighted combination.
    
    But the prompt suggests UCB. Let's use the standard UCB logic adapted for minimization.
    We want to select x that minimizes f(x).
    Let's define the acquisition score such that higher is better.
    
    If we use:
    Score = -mu / cost + beta * sigma / cost
    This favors low mu (good) and high sigma (exploration).
    
    Alternatively, we can use the standard UCB formula for maximization on the *negative* of the function?
    Let g(x) = -f(x). We want to maximize g(x).
    GP for f(x) ~ N(mu_f, sigma_f).
    Then g(x) ~ N(-mu_f, sigma_f).
    UCB for g(x) = E[g(x)] + beta * sigma_g(x) = -mu_f + beta * sigma_f.
    So, Score = (-mu_f + beta * sigma_f) / cost.
    
    This is equivalent to:
    Score = (beta * sigma_f - mu_f) / cost.
    
    This looks correct. It rewards low mu (exploitation) and high sigma (exploration), 
    scaled by 1/cost.
    
    We need to choose beta. A common adaptive beta is sqrt(2 * log(t)) where t is the iteration number.
    However, we don't have t directly, but we have cost. The number of evaluations is roughly 
    proportional to cost.
    Since we don't know the exact iteration count, we can use a fixed beta or an estimate.
    Given the short horizon (cost < 30), a fixed beta might be sufficient. 
    Typical beta values for BO are around 1.0 to 2.0.
    
    Let's try beta = 1.0 initially, or maybe slightly higher to encourage exploration 
    in multimodal functions like Ackley/Rastrigin.
    Let's use beta = 1.5.
    
    Also, we must handle the deadline.
    """
    
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE:
        # Fallback: Simple cost-aware EI
        from scipy.stats import norm
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei / cost

    # Ensure sigma is positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Cost safe
    cost_safe = np.maximum(cost, 1e-9)
    
    # UCB for minimization:
    # We maximize: (-mu + beta * sigma) / cost
    # Let's set beta. 
    # A dynamic beta based on the number of points might be good, but we don't have the count.
    # However, the total cost is bounded by 30. The number of evaluations is small.
    # Let's use a fixed beta = 1.0 for stability, or 2.0 for more exploration.
    # Given the previous TS score of 0.71, and EI (seed) 1.0, 
    # UCB is often between EI and PI in performance but more robust to local optima.
    
    beta = 1.0
    
    scores = (-mu + beta * sigma_safe) / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -1e9)
    
    return scores
# EVOLVE-BLOCK-END
