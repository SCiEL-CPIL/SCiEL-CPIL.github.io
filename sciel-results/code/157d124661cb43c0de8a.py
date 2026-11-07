"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-adjusted Lower Confidence Bound (LCB) acquisition function.
    
    The LCB is defined as: score = mu - beta * sigma
    To account for cost, we scale the exploration term (sigma) by 1/cost,
    so that high-cost regions are penalized more heavily in the uncertainty bonus.
    The mean term (mu) remains invariant to cost in the exploration balance,
    but we also subtract a small cost penalty to discourage expensive points
    when the mean is not clearly better.
    
    Actually, the directive says: "exploration term is scaled by the inverse cost (1/cost),
    ensuring that high-cost regions are penalized more heavily in the uncertainty bonus
    while the mean term remains invariant".
    
    So: score = mu - beta * sigma * (1 / cost)
    
    Wait, if cost is high, 1/cost is small, so the penalty for uncertainty is small?
    That doesn't make sense. "High-cost regions are penalized more heavily in the uncertainty bonus".
    If I want to penalize high cost, I should make the score lower when cost is high.
    If the term is -beta * sigma / cost, then for high cost, this term is close to 0 (less negative).
    That means high cost is NOT penalized more.
    
    Let's re-read: "exploration term is scaled by the inverse cost (1/cost)... high-cost regions are penalized more heavily".
    This implies the penalty should be LARGER for high cost.
    So maybe: score = mu - beta * sigma * cost?
    Or maybe the "exploration term" is the bonus, and we want to reduce the bonus for high cost?
    If the bonus is beta * sigma, and we want high cost to have a smaller bonus, we multiply by 1/cost.
    But then the "penalty" (the negative part) is smaller.
    
    Let's think about LCB: we want to minimize mu + beta * sigma (uncertainty).
    Or maximize mu - beta * sigma.
    If we want to avoid high cost, we should subtract something proportional to cost.
    
    Let's look at the previous successful change: "Switching from EI to PoI... Measured: 1.648 -> 2.743".
    The current program uses EI.
    
    Let's try a hybrid or a different formulation.
    The directive specifically suggests: "cost-adjusted lower confidence bound (LCB) formulation where the exploration term is scaled by the inverse cost (1/cost)".
    
    If the score is `mu - beta * sigma / cost`:
    - Low cost (cost ~ 1): score = mu - beta * sigma. Standard LCB.
    - High cost (cost ~ 2): score = mu - beta * sigma / 2. The uncertainty penalty is halved.
    This makes high-cost regions look MORE attractive if they have high uncertainty, because the penalty for that uncertainty is smaller.
    This contradicts "high-cost regions are penalized more heavily".
    
    Unless "penalized more heavily" refers to the cost itself?
    Maybe the score is `mu - beta * sigma - gamma * cost`?
    
    Let's re-read carefully: "ensuring that high-cost regions are penalized more heavily in the uncertainty bonus".
    This phrasing is tricky. It might mean that the *cost* is a penalty on the *uncertainty bonus*.
    If the bonus is `beta * sigma`, and we want high cost to reduce this bonus, we multiply by `1/cost`.
    But that reduces the bonus, making the point less attractive if it relies on uncertainty.
    However, if a point has high uncertainty AND high cost, the bonus is reduced.
    If a point has high uncertainty AND low cost, the bonus is full.
    This favors low-cost high-uncertainty points.
    
    But does it penalize high-cost regions "more heavily"?
    Compared to a low-cost region with the same uncertainty, the high-cost region has a smaller bonus.
    So it is less attractive.
    
    Let's try this: `score = mu - beta * sigma / cost`.
    We need to choose `beta`.
    In standard LCB for minimization, we often use `mu + kappa * sigma`.
    Here we are maximizing, so `mu - kappa * sigma`.
    
    Let's add a deadline guard.
    """
    start_time = time.time()
    deadline = start_time + 290  # 10s safety margin
    
    # Ensure sigma is not zero
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Cost is 1 + mean(x_scaled). x in [-2, 2], scaled to [0, 1].
    # cost is in [1, 2].
    
    # LCB with cost-adjusted exploration
    # beta controls the trade-off
    # Let's try a few values or a fixed good one.
    # In 2D rosenbrock, typical beta is around 1-5 depending on sigma scale.
    # Since targets are standardized, sigma is in objective units (standardized).
    # f_best is also in standardized units.
    
    # Let's use a beta that adapts slightly or is fixed.
    # Fixed beta = 2.0 is a common starting point.
    
    # The directive says: "exploration term is scaled by the inverse cost (1/cost)"
    # So term = beta * sigma / cost
    
    score = mu - (2.0 * sigma_safe) / cost
    
    # Check time
    if time.time() > deadline:
        # Return a simple fallback if we are running out of time
        # But this is just an array operation, should be fast.
        pass
        
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
