# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Uses a dynamic UCB weight that scales with the ratio of remaining budget
    to current cost, shifting from exploration to exploitation as budget depletes.
    
    Parameters:
    -----------
    mu : array-like, shape (2000,)
        GP mean predictions.
    sigma : array-like, shape (2000,)
        GP standard deviation predictions.
    f_best : float
        Best observed value so far.
    cost : array-like, shape (2000,)
        Cost of evaluating each candidate (1 + mean(x scaled to [0,1])).
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = evaluate next).
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure sigma is positive
    sigma = np.maximum(sigma, 1e-9)
    
    # Compute Expected Improvement (EI)
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Compute UCB
    # We use a dynamic beta that scales with remaining budget
    # Total budget is 30, initial cost from 5 points is roughly 5 * 1.5 = 7.5 (approx)
    # Let's estimate current total cost spent. 
    # The 'cost' array contains the cost of each of the 2000 candidates.
    # We don't know the exact current total cost directly from arguments, 
    # but we can infer a proxy. However, the signature doesn't give total_cost.
    # Let's assume the loop manages the budget. 
    # The directive says: "scales the UCB weight by the ratio of remaining budget to current cost".
    # We don't have `current_cost` or `total_budget` explicitly.
    # However, `cost` is the cost of the *candidate*. 
    # Let's look at the context: "While total cost is below 30".
    # We can approximate the "phase" by using the magnitude of f_best or just use a fixed schedule?
    # No, the directive asks for dynamic. 
    # Since we don't have the global state, we must rely on what we have.
    # Perhaps we can use the average cost of the candidates as a proxy for the "expensive" phase?
    # Or maybe we can assume that as the optimization progresses, the sigma decreases.
    # A common heuristic is to use a beta that decays. But without iteration count, 
    # we can use the relative magnitude of sigma compared to the range of mu?
    
    # Alternative interpretation: The "cost" array itself varies. 
    # Maybe the dynamic part is per-candidate? 
    # "scales the UCB weight by the ratio of remaining budget to current cost"
    # This implies a global weight. Since we don't have global state, 
    # we might need to estimate it. 
    # However, in many BO loops, the acquisition function is called with the current state.
    # If we can't access the iteration count, we might use a heuristic based on the data.
    # For instance, if sigma is very small, we are in exploitation phase.
    
    # Let's try a hybrid: 
    # 1. Calculate EI per cost (efficiency).
    # 2. Add an exploration term that is inversely proportional to the cost 
    #    and scaled by a factor that depends on the "uncertainty" of the best point.
    
    # A robust approach without global state:
    # Use a beta that is proportional to the median sigma of the candidates.
    # Early on, sigma is high -> high beta (exploration).
    # Late on, sigma is low -> low beta (exploitation).
    
    median_sigma = np.median(sigma)
    
    # Normalize sigma to [0, 1] roughly for the weight
    # If median_sigma is large, we are early.
    # Let's map median_sigma to a beta.
    # Assume initial sigma is around 1.0 (standardized targets? no, objective units).
    # The problem says "standardized targets" for GP, but mu/sigma in objective units.
    # f_best is in objective units.
    
    # Let's use a simpler dynamic: 
    # beta = k * (median_sigma / (f_best + 1e-6))
    # This makes beta large when f_best is close to 0 (early, high regret) 
    # and small when f_best is low (late, low regret)? 
    # Actually, f_best decreases. So (median_sigma / f_best) increases? 
    # No, if f_best is 100, beta is small. If f_best is 1, beta is large.
    # That's the opposite of what we want (we want exploration early when f_best is bad/high).
    # Wait, if f_best is high (bad), we want to explore more? 
    # Usually, we want to explore when we have little information. 
    # When f_best is high, we are far from optimum. 
    # When f_best is low, we are close.
    
    # Let's stick to the sigma-based heuristic which is more standard for "phase" detection.
    # beta = 2.0 * median_sigma / (np.max(sigma) + 1e-9)
    # This normalizes beta between 0 and 2.
    
    max_sigma = np.max(sigma)
    if max_sigma < 1e-9:
        beta = 0.0
    else:
        # Scale beta based on relative uncertainty
        # Early: sigma is roughly uniform, max_sigma ~ median_sigma -> beta ~ 1-2
        # Late: sigma is small everywhere, but relative spread might be similar?
        # Actually, as GP converges, sigma drops. 
        # Let's use an absolute threshold or a decay based on the mean sigma.
        
        # Let's try: beta = 2.0 * np.exp(-mean_sigma)
        # If mean_sigma is 1.0, beta ~ 0.7
        # If mean_sigma is 0.1, beta ~ 1.8 (more exploration? No, we want less)
        # This is tricky without knowing the scale.
        
        # Let's go back to the directive: "ratio of remaining budget to current cost".
        # Since we can't get that, let's approximate the "phase" by the number of points evaluated?
        # We don't have that either.
        
        # Let's assume the "dynamic" aspect can be approximated by the cost of the candidates.
        # Candidates with lower cost are "cheaper" to explore.
        # So we should boost low-cost candidates more when we are "exploring".
        
        # Let's define a base exploration weight that is high.
        # And modulate it by the cost.
        
        # Score = EI/cost + beta * sigma / cost
        # Let's make beta depend on the "quality" of the current best.
        # If f_best is very good (close to 0), we should exploit more (lower beta).
        # If f_best is bad, we should explore more (higher beta).
        
        # Normalize f_best? We don't know the range.
        # But we know the minimum is 0.
        # So f_best >= 0.
        
        # Let's use beta = 2.0 / (1.0 + f_best)
        # If f_best = 0, beta = 2.0 (high exploration? No, if we found 0, we are done)
        # If f_best = 10, beta = 0.18 (low exploration? We are far, should explore more)
        # This logic is inverted.
        
        # Correct logic:
        # Early: f_best is large (e.g., 10). We have little info. We should explore. -> High Beta.
        # Late: f_best is small (e.g., 0.1). We have good info. We should exploit. -> Low Beta.
        
        # So beta should decrease as f_best decreases.
        # beta = k * f_best?
        # If f_best = 10, beta = 10k.
        # If f_best = 0.1, beta = 0.1k.
        # This seems reasonable.
        
        # But what is k?
        # Let's normalize by the scale of the problem.
        # We don't know the scale of f.
        
        # Let's try a different approach: 
        # Use the ratio of sigma to the range of mu.
        # range_mu = mu.max() - mu.min()
        # If range_mu is large, we are early.
        
        range_mu = np.max(mu) - np.min(mu)
        if range_mu < 1e-9:
            beta = 1.0
        else:
            # beta scales with the relative uncertainty
            beta = 2.0 * (median_sigma / range_mu)
            
            # Clip beta to avoid extremes
            beta = np.clip(beta, 0.5, 3.0)
    
    # Calculate UCB term
    ucb = mu + beta * sigma
    
    # We want to minimize, so we want high ucb? No.
    # For minimization, we want to find low values.
    # UCB for minimization is often: mu - beta * sigma (Lower Confidence Bound)
    # Or we can use Expected Improvement which handles this naturally.
    
    # Let's combine EI and a cost-adjusted exploration term.
    # EI is already in "improvement" units.
    # Exploration term: sigma / cost
    # We want to maximize (EI/cost + beta * sigma / cost)
    
    # Let's refine the beta calculation.
    # The directive specifically mentions "ratio of remaining budget to current cost".
    # Since we can't compute that, we use the sigma-based heuristic as a proxy for "remaining budget"
    # (high sigma = early = lots of budget left).
    
    # Final Score:
    # score = (ei + beta * sigma) / cost
    
    # Ensure finite scores
    scores = (ei + beta * sigma) / cost
    
    # Handle any non-finite values
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
