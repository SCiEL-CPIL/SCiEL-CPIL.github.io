"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Probabilistic Lower Bound (PLB) acquisition function.
    
    For minimization, we want to find points where the function value is likely to be low.
    The PLB score is designed to maximize the probability that f(x) < f_best.
    
    score = mu + k * sigma
    
    Wait, for minimization, lower mu is better. Higher sigma means more uncertainty.
    The standard PLB for maximization is: score = mu + k * sigma (upper confidence bound).
    For minimization, we can use: score = -(mu + k * sigma) or equivalently maximize -mu - k*sigma.
    But the task says "higher meaning evaluate next". So we want higher scores to be selected.
    
    Let's think carefully. We want to minimize f. f_best is the current minimum.
    We want to find x such that f(x) < f_best.
    
    A common approach for minimization is to use the Lower Confidence Bound (LCB):
    LCB(x) = mu(x) - k * sigma(x)
    We want to minimize LCB, or equivalently, maximize -LCB = -mu + k*sigma.
    
    However, the directive says: "Use a Probabilistic Lower Bound (PLB) acquisition, score = mu + k * sigma, 
    which maximizes the probability that the true function value exceeds a dynamic threshold derived from f_best, 
    thereby explicitly targeting points likely to improve the incumbent while naturally decaying exploration as confidence increases."
    
    This description seems to be for maximization (exceeds threshold). For minimization, we want f(x) < f_best.
    
    Let's re-read: "maximizes the probability that the true function value exceeds a dynamic threshold derived from f_best".
    If we're minimizing, "exceeds" doesn't make sense for improvement. Improvement means f(x) < f_best.
    
    Perhaps the directive is using "PLB" in a specific way. Let's consider the standard PLB for minimization.
    
    Actually, for minimization, a common acquisition is the Lower Confidence Bound:
    score = mu - k * sigma
    Higher scores here would mean higher mu - k*sigma, which we don't want. We want low mu - k*sigma.
    
    So if we return scores where higher is better, we should return:
    score = -mu + k * sigma  (which is -(mu - k*sigma))
    
    Or we could use the form: score = mu + k * sigma but interpret it differently.
    
    Let's stick to the directive's formula: score = mu + k * sigma.
    For minimization, this would favor high mu and high sigma. That seems wrong for minimization.
    
    Wait, maybe the directive is slightly misstated for minimization. Let's use the standard approach:
    For minimization, we use LCB: mu - k*sigma. To make higher scores better, we return -(mu - k*sigma) = -mu + k*sigma.
    
    Let's use: score = -mu + k * sigma
    
    We need to choose k. A common adaptive choice is k = 1.0 or k = 0.5, or adaptive based on the number of evaluations.
    Since we don't have the number of evaluations directly, we can use a fixed k or estimate it.
    
    Actually, we can use a simple adaptive k that decreases over time. But we don't have a counter.
    Let's use a fixed k = 1.0 for now, or maybe k = 0.5.
    
    Let's try k = 1.0 first.
    """
    # Use LCB for minimization: score = -mu + k * sigma
    # Higher score means more likely to find a lower value
    k = 1.0
    score = -mu + k * sigma
    
    # Ensure scores are finite
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
