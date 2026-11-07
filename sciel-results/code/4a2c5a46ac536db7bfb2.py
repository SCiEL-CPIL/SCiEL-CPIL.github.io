import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using Deterministic Thompson Sampling (Median) 
    with Cost Adjustment.
    
    Strategy:
    1. Estimate the posterior median for each candidate: m = mu - sigma * z_0.5
       (where z_0.5 = 0, so m = mu). Wait, median of N(mu, sigma^2) is mu.
       To get a more "exploratory" deterministic sample, we can use a quantile 
       slightly below the median, or simply use the mean (mu) but penalize 
       based on the *expected* cost and the *uncertainty* in the cost-adjusted value.
       
    Actually, standard Thompson Sampling samples theta ~ N(mu, sigma^2) and picks argmin theta.
    A deterministic approximation is to pick the point that minimizes the posterior median.
    However, for minimization, the posterior median is just mu. This is pure exploitation (Greedy).
    To add exploration deterministically, we can use the **Lower 25th Percentile** or similar,
    or use **Expected Minimum** approximation.
    
    A better deterministic TS approximation for minimization is:
    Score = - (mu - kappa * sigma) / cost_effective
    But that's LCB again.
    
    Let's try **Expected Improvement (EI)** but with a **Cost-Adjusted Target**.
    Standard EI: E[max(f_best - f(x), 0)]
    Cost-Aware EI: E[max(f_best - f(x), 0)] / Cost(x)
    
    The current program uses LCB. EI is a different design family.
    Let's implement **Cost-Effective Expected Improvement (CEI)**.
    
    CEI(x) = E[Improve(x)] / Cost(x)
    where E[Improve(x)] = (f_best - mu) * Phi((f_best - mu)/sigma) + sigma * phi((f_best - mu)/sigma)
    
    This is a well-established method for expensive/costly BO.
    """
    start_time = time.time()
    deadline = 300.0
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = mu.shape[0]
    if n == 0:
        return np.array([])
        
    # Sanitization
    mu = np.nan_to_num(mu, nan=0.0, posinf=1e6, neginf=-1e6)
    sigma = np.nan_to_num(sigma, nan=1e-8, posinf=1e6, neginf=1e-8)
    cost = np.nan_to_num(cost, nan=1.0, posinf=1e6, neginf=1.0)
    
    sigma = np.maximum(sigma, 1e-8)
    cost = np.maximum(cost, 1e-8)
    
    f_best = float(f_best)
    if not np.isfinite(f_best):
        f_best = 0.0

    # Calculate Z-score for EI
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Handle numerical stability for large |z|
    # Phi(z) is CDF, phi(z) is PDF
    # For large negative z, Phi(z) ~ 0, phi(z) ~ 0. EI ~ 0.
    # For large positive z, Phi(z) ~ 1, phi(z) ~ 0. EI ~ f_best - mu.
    
    # Use scipy.stats.norm
    # To avoid overflow/underflow issues in standard implementations, we can clip z.
    # However, norm.cdf and norm.pdf handle large values reasonably well in recent SciPy versions.
    # Let's clip z to [-10, 10] for safety in PDF calculations, though CDF is fine.
    z_clipped = np.clip(z, -10, 10)
    
    # Expected Improvement
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Note: If mu > f_best, z is negative. Phi(z) is small.
    # If mu < f_best, z is positive. Phi(z) is large.
    
    phi_z = norm.pdf(z_clipped)
    Phi_z = norm.cdf(z_clipped)
    
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Ensure EI is non-negative (numerical errors might make it slightly negative)
    ei = np.maximum(ei, 0.0)
    
    # Cost-Effective EI
    # Score = EI / Cost
    # Higher is better.
    scores = ei / cost
    
    # Alternative: Use a blend of EI and UCB to ensure some exploration 
    # even if EI is zero (e.g., if sigma is very small everywhere).
    # But EI naturally decreases as sigma decreases.
    
    # Let's add a small exploration bonus based on sigma to avoid getting stuck 
    # if the GP becomes overconfident (sigma -> 0) but hasn't found the global min.
    # Exploration term: sigma / cost
    # We can blend: Score = EI/cost + lambda * sigma/cost
    # However, pure CEI is a distinct and strong strategy.
    
    # Let's check time
    if time.time() > deadline - 5.0:
        # Fallback: simple LCB
        scores = (1.5 * sigma - mu) / cost

    # Ensure finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
