"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Track start time for deadline guard
_T0 = time.time()
_DEADLINE = 300.0  # Safety margin before 330s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for minimizing levy 2d.
    
    Uses a hybrid approach combining:
    1. Cost-adjusted Expected Improvement (EI) as the primary exploitation signal
    2. A bounded exploration bonus for high-uncertainty regions
    3. A Lower Confidence Bound (LCB) term scaled by reciprocal cost for
       deterministic risk-averse exploration
    4. Deadline guard to ensure we return before timeout
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of GP posterior at candidate points
    sigma : ndarray, shape (2000,)
        Standard deviation of GP posterior at candidate points
    f_best : float
        Best observed function value so far
    cost : ndarray, shape (2000,)
        Cost of evaluating each candidate point
    
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores (higher = more desirable to evaluate)
    """
    # Deadline guard: if too close to limit, return a fast deterministic signal
    if time.time() - _T0 > _DEADLINE:
        cost_safe = np.maximum(cost, 1e-9)
        score = -mu / cost_safe
        return np.where(np.isfinite(score), score, 0.0)
    
    # Ensure inputs are numpy arrays for vectorized operations
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Safety: replace non-finite values in inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-6)
    cost = np.where(np.isfinite(cost) & (cost > 1e-12), cost, 1.0)
    
    # Clip sigma to avoid division by zero and extreme values
    sigma = np.clip(sigma, 1e-8, 1e6)
    
    # Clip cost to avoid division by zero
    cost = np.maximum(cost, 1e-8)
    
    # Ensure f_best is a finite scalar
    if not np.isfinite(f_best):
        f_best = 0.0
    f_best = float(f_best)
    
    # Compute Expected Improvement (EI) for minimization
    # z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Clip z to avoid extreme values
    z = np.clip(z, -10.0, 10.0)
    
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    cdf_val = norm.cdf(z)
    pdf_val = norm.pdf(z)
    
    ei = (f_best - mu) * cdf_val + sigma * pdf_val
    
    # Ensure EI is non-negative (numerical stability)
    ei = np.maximum(ei, 0.0)
    
    # Check deadline before adding exploration terms
    if time.time() - _T0 > _DEADLINE - 10.0:
        # Fallback: just return EI/cost
        score = ei / cost
        score = np.where(np.isfinite(score), score, 0.0)
        return score
    
    # Bounded exploration bonus: sigma/(sigma + 0.1)
    # This is bounded in [0, 1) and prevents exploration from dominating
    # when sigma is large. The scale 0.1 is chosen to be small relative to
    # the expected objective range of Levy 2d (~[0, 20+]).
    exploration_bonus = sigma / (sigma + 0.1)
    
    # Lower Confidence Bound (LCB) term for minimization:
    # LCB = mu - kappa * sigma (lower is better for minimization)
    # We want to maximize the "improvement" signal, so we use -LCB = -mu + kappa * sigma
    # Scale by 1/cost for cost-awareness
    # Choose kappa to balance exploration; 2.0 is a common choice
    kappa = 2.0
    lcb_improvement = -mu + kappa * sigma
    
    # Normalize LCB improvement to be comparable to EI scale
    # EI is in objective units, LCB improvement is also in objective units
    # But we want the LCB term to be a secondary signal, so scale it down
    lcb_weight = 0.1
    
    # Cost-aware LCB: improvement per unit cost
    lcb_per_cost = lcb_weight * lcb_improvement / cost
    
    # Combine EI, exploration bonus, and LCB
    # EI is the primary signal, exploration bonus is bounded and small,
    # LCB provides deterministic risk-averse exploration
    alpha_exploration = 0.05
    combined = ei + alpha_exploration * exploration_bonus + lcb_per_cost
    
    # Cost-aware acquisition: combined score per unit cost
    score = combined / cost
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    # Final deadline check
    if time.time() - _T0 > _DEADLINE:
        return score
        
    return score
# EVOLVE-BLOCK-END
