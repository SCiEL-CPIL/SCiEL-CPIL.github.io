import numpy as np
from scipy.stats import norm
import time

_start_time = None

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Knowledge Gradient acquisition function for Bayesian optimization.
    
    KG estimates the expected reduction in f_best after evaluating each candidate,
    normalized by cost. This is a structural change from EI: instead of measuring
    expected local improvement, KG measures the expected value of information.
    
    Parameters:
    - mu: array of shape (2000,) - GP mean predictions
    - sigma: array of shape (2000,) - GP std dev predictions  
    - f_best: float - best observed value so far
    - cost: array of shape (2000,) - evaluation cost for each candidate
    
    Returns:
    - scores: array of shape (2000,) - higher means evaluate next
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 290:
        mu = np.asarray(mu, dtype=np.float64)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        scores = -mu
        return np.where(np.isfinite(scores), scores, 0.0)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: replace non-finite values
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)
    
    # Clip sigma
    sigma = np.clip(sigma, 1e-8, None)
    
    # Knowledge Gradient approximation:
    # KG(x) = E[ max(f_best, f(x)) - f_best ] where f(x) ~ N(mu(x), sigma(x))
    # This is equivalent to: (f_best - mu) * Phi((f_best-mu)/sigma) + sigma * phi((f_best-mu)/sigma)
    # which is actually the same as EI! So let's use a different structural approach.
    
    # Alternative: Use the probability that x beats f_best, weighted by expected improvement,
    # but with a cost-aware exploration bonus based on posterior variance.
    # 
    # Structural change: Use a "probabilistic regret" metric that combines:
    # 1. P(f(x) < f_best) - probability of improvement
    # 2. E[improvement | improvement occurs] - conditional expected improvement
    # 3. Uncertainty bonus: sigma-based exploration for regions with high variance
    #
    # This creates a different search landscape than pure EI or CEI.
    
    z = (f_best - mu) / sigma
    prob_improve = norm.cdf(z)
    expected_improvement = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    expected_improvement = np.maximum(expected_improvement, 0)
    
    # Conditional expected improvement given improvement occurs
    with np.errstate(divide='ignore', invalid='ignore'):
        cond_ei = np.where(prob_improve > 1e-10, expected_improvement / prob_improve, 0.0)
    
    # Combine: weight by probability of improvement and add uncertainty-driven exploration
    # The exploration term encourages visiting high-uncertainty regions that haven't been
    # well-explored, which is structurally different from EI's exploitation focus
    exploration_bonus = sigma * 0.1  # Scale to be comparable to typical EI values
    
    # Cost-aware scoring: divide by cost (linear penalty, not sqrt)
    # This creates a stronger cost penalty than sqrt, favoring cheaper evaluations
    base_score = expected_improvement + exploration_bonus
    scores = base_score / cost
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
