"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard: The evaluator kills after 330s.
# We are not doing a loop inside acquisition, so this is mostly a safety check.
_start_time = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware Lower Confidence Bound (LCB) acquisition function.
    
    Parameters:
    mu : np.ndarray, shape (N,)
        Mean of the GP posterior.
    sigma : np.ndarray, shape (N,)
        Standard deviation of the GP posterior.
    f_best : float
        The best observed function value so far.
    cost : np.ndarray, shape (N,)
        Cost of evaluating each candidate.
    
    Returns:
    scores : np.ndarray, shape (N,)
        Acquisition scores (higher is better).
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: avoid division by zero or negative costs
    cost = np.maximum(cost, 1e-9)
    
    # LCB for minimization:
    # We want to find points where the function value is likely low.
    # The lower confidence bound is mu - beta * sigma.
    # A smaller LCB indicates a potentially better (lower) function value.
    # To convert to a "higher is better" score, we can negate it: -mu + beta * sigma.
    
    # However, standard LCB uses a fixed beta. In BO, an adaptive beta or a 
    # balance between exploitation (low mu) and exploration (high sigma) is key.
    
    # Let's define the acquisition score as:
    # Score = (f_best - (mu - beta * sigma)) / cost
    # This is equivalent to:
    # Score = ((f_best - mu) + beta * sigma) / cost
    
    # This form is similar to EI but without the non-linearity of the expectation.
    # The term (f_best - mu) represents the current expected improvement (exploitation).
    # The term beta * sigma represents the uncertainty bonus (exploration).
    
    # How to choose beta?
    # A common heuristic is to scale beta with the number of iterations or the 
    # current uncertainty. However, we don't have iteration count directly.
    # We can use a fixed beta that balances the scales.
    # Since mu and sigma are in objective units, and f_best is in objective units,
    # beta should be dimensionless.
    
    # If beta is too small, we exploit too much. If too large, we explore too much.
    # Let's try a beta that is moderate. 
    # Note: The current EI program uses a relaxed threshold which effectively 
    # increases the exploration pressure. LCB with a reasonable beta should 
    # capture similar behavior but is more stable.
    
    # Let's use a beta that adapts slightly to the scale of the problem.
    # However, without knowing the function range, a fixed beta is safer.
    # A value around 1.0 to 2.0 is common for normalized objectives, but here 
    # objectives are not normalized. 
    # Let's look at the sigma scale. If sigma is small, the exploration term is small.
    # If sigma is large, it is large. This is self-regulating.
    
    # Let's try beta = 1.0. This means we are willing to explore if the uncertainty 
    # is comparable to the expected improvement.
    
    beta = 1.0
    
    # Calculate the LCB-based score
    # We want to maximize the potential for improvement relative to cost.
    # Potential improvement = f_best - mu (exploitation)
    # Uncertainty bonus = beta * sigma (exploration)
    # Total potential = (f_best - mu) + beta * sigma
    
    # If mu > f_best, (f_best - mu) is negative. The sigma term might still make 
    # the score positive if uncertainty is high enough. This is good for exploration.
    # If mu < f_best, (f_best - mu) is positive, adding to the score.
    
    # One issue: if f_best is very close to the minimum, (f_best - mu) is small 
    # for most points. The sigma term dominates. This encourages exploration.
    # This is exactly what we want to avoid stagnation.
    
    # Let's ensure sigma is non-negative
    sigma = np.maximum(sigma, 0.0)
    
    # Calculate scores
    # scores = ((f_best - mu) + beta * sigma) / cost
    
    # However, if f_best is very large (early iterations), (f_best - mu) might be 
    # dominated by f_best for all points, making the score roughly constant.
    # Dividing by cost still differentiates by cost.
    # But the relative differences in mu and sigma might be drowned out.
    
    # A better normalization might be to center mu around f_best?
    # Or just use the raw value. The evaluator picks the max.
    # If all scores are shifted by a constant, the argmax is unchanged.
    # So (f_best - mu) + beta * sigma is fine.
    
    scores = ((f_best - mu) + beta * sigma) / cost
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Safety check for deadline (though this function is fast)
    # If this function is called in a loop by the evaluator, it should be fast.
    # No need for complex deadline logic here unless we were doing optimization inside.
    
    return scores
# EVOLVE-BLOCK-END
