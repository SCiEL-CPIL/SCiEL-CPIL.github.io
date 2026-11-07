import numpy as np
import time
from scipy.stats import norm

_DEADLINE = time.time() + 320.0  # safety margin before 330s wall-clock


def acquisition(mu, sigma, f_best, cost):
    # --- deadline guard: if we're close to the limit, return a safe finite array
    if time.time() > _DEADLINE - 5.0:
        return np.ones_like(mu, dtype=float) * 0.0

    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)

    # Guard against non-finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 0), cost, 1.0)

    # Expected Improvement (standard, robust form)
    improvement = f_best - mu
    z = improvement / sigma
    z = np.clip(z, -30.0, 30.0)
    EI = improvement * norm.cdf(z) + sigma * norm.pdf(z)
    EI = np.where(np.isfinite(EI), EI, 0.0)
    EI = np.maximum(EI, 0.0)

    # UCB for minimization: -mu + beta * sigma
    # Use adaptive beta based on the scale of uncertainty
    beta = 2.5
    UCB = -mu + beta * sigma
    UCB = np.where(np.isfinite(UCB), UCB, 0.0)

    # Combine EI and UCB with cost-aware weighting
    # The key insight: use a multiplicative feasibility constraint
    # Define a "budget gate" based on cost efficiency
    
    # Calculate efficiency metrics
    # For EI: improvement per unit cost
    eff_ei = EI / cost
    # For UCB: potential gain per unit cost (using -mu as proxy for gain)
    eff_ucb = np.maximum(-mu, 0) / cost

    # Adaptive gate: use the median efficiency as a threshold
    # This creates a hard constraint that forces exploitation of cheap regions
    
    # Combine EI and UCB into a single score before cost adjustment
    # Weight: more weight to EI when we have good predictions, more to UCB when uncertain
    # Use the ratio of sigma to the spread of mu to determine uncertainty
    mu_spread = np.ptp(mu) if len(mu) > 1 else 1.0
    if mu_spread < 1e-8:
        mu_spread = 1.0
    
    # Average relative uncertainty
    avg_rel_sigma = np.mean(sigma / (np.abs(mu) + 1e-8))
    
    # Blend factor: higher when uncertainty is high
    blend = np.clip(avg_rel_sigma * 2.0, 0.3, 0.7)
    
    combined = (1 - blend) * EI + blend * UCB
    combined = np.where(np.isfinite(combined), combined, 0.0)
    combined = np.maximum(combined, 0.0)
    
    # Cost-aware feasibility gate
    # Compute the efficiency of each candidate
    eff = combined / cost
    
    # Use a quantile-based gate: keep candidates in the top 60% by efficiency
    # This creates a hard budget-aware constraint
    if len(eff) > 10:
        gate_threshold = np.percentile(eff, 40)  # top 60%
    else:
        gate_threshold = 0.0
    
    # Apply the gate: full score for feasible, heavily damped for infeasible
    damp_factor = 0.08
    scores = np.where(eff >= gate_threshold, combined, combined * damp_factor)
    
    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Small exploration floor
    floor = 1e-12
    scores = np.maximum(scores, floor)
    
    return scores
