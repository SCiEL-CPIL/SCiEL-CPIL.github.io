"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.stats as stats

# Deadline guard: The evaluator kills after 330s.
_start_time = time.time()
TOTAL_BUDGET = 30.0

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-Effective Improvement (CEI) acquisition function.
    
    Instead of a static LCB, this formulates the acquisition as maximizing
    expected improvement per unit of remaining budget.
    
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
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: avoid division by zero
    cost = np.maximum(cost, 1e-9)
    
    # Estimate remaining budget.
    # We don't have the exact cumulative cost, but we can infer it from the fact that
    # we start with 5 points and total budget is 30.
    # However, a more robust dynamic approach is to use the current f_best and sigma
    # to infer the "urgency" or "budget pressure".
    
    # A simpler proxy for remaining budget in this specific loop structure:
    # The loop runs while cost < 30. 
    # We can estimate the "average cost so far" if we knew the number of evaluations.
    # Since we don't have 'n_evals', we use a heuristic:
    # As f_best gets closer to the minimum (0 for most, -3.86 for Hartmann),
    # the remaining "useful" budget is effectively lower.
    
    # Let's use a dynamic beta that depends on the "confidence" of f_best.
    # If f_best is very low, we are in a late stage, so we should be more exploitative.
    # If f_best is high, we are early, so we can be more explorative.
    
    # Normalize f_best relative to the expected range of the functions.
    # Ackley, Rastrigin, Levy have min 0. Hartmann has min -3.86.
    # We don't know which function is being run.
    # However, f_best is in objective units.
    
    # Let's use the standard EI formula but normalized by cost and a dynamic budget factor.
    # EI = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    # where Phi is CDF and phi is PDF of standard normal.
    
    z = (f_best - mu) / np.maximum(sigma, 1e-9)
    
    # Compute EI
    ei = (f_best - mu) * stats.norm.cdf(z) + sigma * stats.norm.pdf(z)
    
    # Ensure EI is non-negative (it can be slightly negative due to numerical issues or if mu > f_best)
    ei = np.maximum(ei, 0.0)
    
    # Cost-aware scaling:
    # We want to maximize EI per unit of cost, but also consider the remaining budget.
    # If the cost is high, we should only pick it if the EI is very high.
    # A common approach is to divide by cost.
    # To incorporate "remaining budget", we can scale by a factor that decreases as we spend more.
    # Since we don't know the exact spent budget, we can use the magnitude of f_best as a proxy for progress.
    # However, a simpler and more robust structural change is to use a "soft" cost penalty.
    
    # Let's try a dynamic beta-like scaling on the EI.
    # Actually, CEI is often defined as EI / cost.
    # To make it more sensitive to remaining budget, we can multiply by a factor that increases 
    # the weight of cheap points when the budget is tight.
    
    # Heuristic: The "pressure" of the budget is proportional to the cost itself relative to the average.
    # Let's stick to EI / cost^alpha. If alpha > 1, we penalize expensive points more.
    # If alpha < 1, we are less sensitive to cost.
    # Standard is alpha = 1.
    
    # To add the "dynamic remaining budget" aspect without knowing n_evals:
    # We can assume that as f_best improves, the marginal value of exploration decreases.
    # Let's scale EI by a factor that depends on how "good" f_best is.
    # But since we don't know the minimum, we can't normalize f_best.
    
    # Alternative Structural Change:
    # Use a Gaussian Process-UCB with a dynamic beta that is inversely proportional to the cost.
    # But we already tried UCB.
    
    # Let's try a different representation: 
    # Maximize the *probability* of improvement, weighted by the magnitude of improvement, 
    # divided by the cost.
    
    # P(improve) = Phi((f_best - mu)/sigma)
    # E[improve | improve] = (f_best - mu) + sigma * (phi(z)/Phi(z))  (hazard function)
    # This is exactly EI.
    
    # So CEI = EI / cost.
    
    # To make it dynamic, let's introduce a "budget urgency" factor.
    # If the cost of the current candidate is high, and f_best is already quite low, 
    # the risk of wasting budget is high.
    
    # Let's try: score = EI / (cost * (1 + cost)) 
    # This penalizes high cost quadratically, strongly favoring cheap points unless EI is huge.
    
    scores = ei / (cost * (1.0 + cost))
    
    # Handle numerical issues
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
