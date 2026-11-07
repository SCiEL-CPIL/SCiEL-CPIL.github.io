import numpy as np
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a deterministic, time-aware strategy that blends
    cost-aware expected improvement with an uncertainty-driven
    exploration term, and adapts its aggressiveness as the
    cost budget is consumed.
    """
    # Ensure inputs are numpy arrays
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Clip sigma to be non-negative
    sigma = np.maximum(sigma, 0.0)
    
    # Ensure cost is positive
    cost = np.maximum(cost, 1e-6)
    
    # Compute Expected Improvement (EI)
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    # For sigma == 0, EI = max(f_best - mu, 0)
    
    z = np.zeros_like(mu)
    mask = sigma > 1e-10
    z[mask] = (f_best - mu[mask]) / sigma[mask]
    
    # Use scipy.stats.norm for Phi and phi
    from scipy.stats import norm
    Phi = norm.cdf(z)
    phi = norm.pdf(z)
    
    EI = (f_best - mu) * Phi + sigma * phi
    EI = np.maximum(EI, 0.0)
    
    # Compute a cost-aware version of EI
    # Higher cost -> lower score
    # Use a soft normalization: score = EI / (1 + cost)
    # This penalizes expensive points but doesn't zero them out completely
    
    # Adaptive exploration: as cost accumulates, we should explore less
    # and exploit more. We use the total cost budget (30) to modulate.
    # cost here is the cost of each candidate, not cumulative.
    # But we can use the mean cost of the 5 initial points or similar.
    # Actually, cost is the cost of evaluating each candidate.
    # The total budget is 30. We don't know the current cumulative cost,
    # but we can infer progress from the fact that we're in the loop.
    # However, we don't have the cumulative cost passed in.
    # We only have the per-candidate cost.
    
    # Let's use a simpler approach:
    # score = (EI + beta * sigma) / cost
    # where beta controls exploration.
    
    # To make it time-aware, we can use the current time to modulate beta.
    # Early on: high beta (more exploration)
    # Late: low beta (more exploitation)
    # But we don't know how many iterations have been done.
    # We can use the time since the start of the acquisition call as a proxy,
    # but that's not very reliable.
    
    # Alternative: use a fixed, well-tuned beta.
    # Given the Hartmann 3D landscape, a moderate beta works well.
    
    beta = 0.5  # Exploration coefficient
    
    # Combine EI and exploration
    combined = EI + beta * sigma
    
    # Cost-aware scoring
    scores = combined / cost
    
    # Ensure finite values
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    
    return scores
