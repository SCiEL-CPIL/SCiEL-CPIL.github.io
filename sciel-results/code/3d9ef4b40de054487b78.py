import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Knowledge Gradient (KG) for minimization.
    
    The Knowledge Gradient for minimization is:
    KG(x) = E[min(mu(x), min_{y != x} mu(y))] - min(mu(x), min_{y != x} mu(y))
    
    However, a common and computationally efficient approximation for the 
    one-step KG (or using the property that it relates to EI) is often 
    approximated or simplified. 
    
    Actually, the exact one-step KG for a single point x given current data is:
    KG(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z) - (f_best - mu(x)) * (1 - Phi(z))
    Wait, that's not quite right.
    
    Let's use the standard approximation for KG in BO:
    The KG value for a point x is the expected improvement in the *estimate of the global minimum*.
    
    A widely used, efficient surrogate for KG that balances exploration and exploitation 
    differently than EI is the **Expected Simple Regret** or simply using a 
    **Pareto-frontier aware** metric.
    
    However, a distinct and robust alternative to EI is the **Probability of Improvement (PI)** 
    with a small xi, but that's often too exploitative.
    
    Let's implement the **Knowledge Gradient** properly.
    For a single candidate x, the KG is:
    E[ min(mu(x), f_best) ] - min(mu(x), f_best) ... no, that's not it.
    
    The correct formula for the one-step KG for minimization, assuming we only care about 
    whether x beats the current best f_best, is actually equivalent to EI in some contexts, 
    but generally:
    
    KG(x) = (f_best - mu(x)) * Phi(z) + sigma(x) * phi(z)
    This is exactly EI. So KG and EI are closely related.
    
    To be materially different, let's use **UCB (Upper Confidence Bound)** with a 
    dynamic beta that starts high for exploration and decays, or a fixed high beta 
    to force exploration of the multimodal landscape. The previous attempt at blending 
    failed, but a pure UCB with a carefully chosen constant might explore the 
    multiple minima of Styblinski-Tang (which has multiple local minima) more 
    effectively than EI, which can get stuck in the first basin found.
    
    Styblinski-Tang 1D: f(x) = sum_{i=1}^n [x_i^2 - 2.5 x_i^2 cos(5.2 x_i) + 0.5]
    For n=1: f(x) = x^2 - 2.5 x^2 cos(5.2 x) + 0.5
    Global min at x = -1.71245, f(x) = -39.1662.
    There are local minima around x = 1.25, 2.5, etc.
    
    EI tends to converge to the first good point. UCB with a larger beta encourages 
    visiting unexplored regions (high sigma) which might contain the global minimum 
    if the initial points missed it.
    
    Let's implement UCB:
    Score = mu + beta * sigma
    
    We need to choose beta. 
    If beta is too small, it behaves like EI/Greedy.
    If beta is too large, it explores randomly.
    
    A common heuristic is beta = 2.0 or similar. 
    Given the scale of the function (values around -39 to +10), and sigma might be 
    in a similar range initially.
    
    Let's try a pure UCB with beta = 2.0. This is a structural change from EI.
    """
    start_time = time.time()
    deadline = start_time + 290

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative
    eps = 1e-8
    sigma = np.maximum(sigma, eps)
    
    # UCB for minimization: we want to minimize f(x).
    # The upper bound of f(x) is mu + beta * sigma.
    # We want to pick the point with the *lowest* upper bound? 
    # No, UCB for minimization is often formulated as:
    # We want to find the point with the highest potential for being low.
    # The "optimistic" value is mu - beta * sigma? 
    # No, that would be the lower bound.
    # 
    # Standard UCB for maximization: score = mu + beta * sigma (pick max score).
    # For minimization, we can invert the function or use:
    # score = -mu + beta * sigma (pick max score, which corresponds to min -mu, i.e., min mu, but with exploration bonus).
    # Or: score = mu - beta * sigma (pick min score).
    #
    # Let's stick to the "higher score is better" convention required by the task.
    # If we want to minimize f, and we use UCB, we are estimating the range [mu - beta*sigma, mu + beta*sigma].
    # The most optimistic value for being *low* is mu - beta * sigma.
    # So if we want the acquisition to return a score where higher is better, 
    # and we want to select the point with the lowest optimistic value, 
    # we should return: score = -(mu - beta * sigma) = -mu + beta * sigma.
    
    # Let's verify:
    # Point A: mu=0, sigma=1. Optimistic min = -beta. Score = beta.
    # Point B: mu=10, sigma=10. Optimistic min = 10 - 10*beta.
    # If beta=2, A score=2, B score = -10 - 20 = -30.
    # We pick A. This makes sense if A is likely lower.
    # But if B is in an unexplored region, sigma is high, so -mu + beta*sigma becomes very negative (bad score).
    # Wait, high sigma should be *good* for exploration.
    # In maximization UCB: mu + beta*sigma. High sigma -> high score. Good.
    # In minimization, if we transform to maximization of -f:
    # mu_neg = -mu, sigma_neg = sigma.
    # UCB_neg = mu_neg + beta * sigma_neg = -mu + beta * sigma.
    # We maximize UCB_neg.
    # So yes, score = -mu + beta * sigma.
    
    beta = 2.0
    scores = -mu + beta * sigma
    
    # Ensure finite scores
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    if time.time() > deadline:
        pass
        
    return scores
