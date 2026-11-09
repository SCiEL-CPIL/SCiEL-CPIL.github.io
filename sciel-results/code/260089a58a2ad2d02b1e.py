"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()
_DEADLINE = 320.0  # Seconds, with safety margin before 330s kill


def acquisition(mu, sigma, f_best, cost):
    # Check deadline
    elapsed = time.time() - _START_TIME
    if elapsed > _DEADLINE - 5:  # 5s safety margin
        # Return a simple score based on low cost to avoid timeout
        return -cost

    # Knowledge Gradient (KG) per unit cost
    # KG approximates the expected improvement of the best known value after
    # evaluating x. It is generally more robust than EI in high dimensions
    # because it considers the probability that x is the new global minimum
    # relative to the current best, weighted by the magnitude of improvement.
    
    eps = 1e-9
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)
    
    # z-score for the improvement of x over f_best
    # Note: In minimization, we want mu < f_best.
    # The "improvement" is f_best - mu.
    # The probability that x is better than f_best is P(f(x) < f_best) = Phi(z)
    # where z = (f_best - mu) / sigma.
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (EI) is a lower bound on KG.
    # A common approximation for KG in the single-objective case is:
    # KG(x) ≈ EI(x) + sigma * E[max(0, f_best - f(x)) | f(x) < f_best] ... 
    # Actually, for a single candidate, KG is exactly the expected value of
    # max(0, f_best - f(x)) if we assume the current best is the only other point.
    # However, the standard EI formula IS the expected improvement of the best value
    # if we assume the GP is the only source of uncertainty and we compare against f_best.
    # 
    # Wait, EI(x) = E[max(f_best - f(x), 0)].
    # KG(x) = E[max(f_best - min(f(x), f_best), 0)] ... No.
    # KG(x) is the expected reduction in the optimal value.
    # If we have a set of points, KG is complex. But with just f_best,
    # KG(x) = EI(x).
    # 
    # So why is EI failing? Because EI is greedy.
    # To differentiate from the current program (which is EI/cost), I need a
    # different acquisition function.
    # 
    # Let's use **Probability of Improvement (POI) per cost** combined with a
    # **Diversity/Uncertainty** term, or better, **Upper Confidence Bound (UCB) per cost**.
    # UCB is known to be more exploratory than EI.
    # UCB(x) = mu(x) - beta * sigma(x)  (for minimization, we want low mu, high sigma)
    # Actually, for minimization, we want to minimize f.
    # UCB for minimization: LCB(x) = mu(x) - beta * sigma(x).
    # We want to maximize the acquisition score, so we can use -LCB(x) = -mu(x) + beta*sigma(x).
    # Or simply, Score = -mu + beta*sigma.
    # 
    # Let's try **Expected Hypervolume Improvement (EHI)** or a simpler **UCB** strategy.
    # The prompt asks for a "DIFFERENT strategy". EI is greedy. UCB is exploration-driven.
    # Let's implement Cost-Aware UCB.
    
    # Beta parameter controls exploration.
    # In high dimensions, we need more exploration.
    # A common choice is beta = 2 or 3.
    # Let's make beta adaptive or fixed. Fixed 3.0 is a good start for 8D.
    beta = 3.0
    
    # UCB Score for minimization:
    # We want to pick x that has low mu and high sigma.
    # Acquisition Score = -mu + beta * sigma
    # 
    # Cost-aware:
    # Score = (-mu + beta * sigma) / cost
    
    # However, raw -mu can be dominated by the scale of mu.
    # mu and sigma are in objective units.
    # Let's normalize or just use this form.
    
    scores = (-mu + beta * sigma_safe) / cost_safe
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
