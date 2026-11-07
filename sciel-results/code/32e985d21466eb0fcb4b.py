"""Cost-aware acquisition function for Bayesian optimization (shekel 4d)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 320.0  # Safety margin before 330s kill

# Phase tracking: we don't know the exact iteration, but we can infer
# progress from the cost array (higher total cost = later in the loop)
# The loop runs while total cost < 30, starting with 5 uniform points.


def _norm_cdf(x):
    """Numerically stable standard normal CDF using scipy."""
    from scipy.special import erf
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def _norm_pdf(x):
    """Standard normal PDF."""
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using gamma-EI with phase-adaptive exploration.
    
    Structural change: instead of blending EI + Thompson + uncertainty,
    use a single principled acquisition: gamma-expected improvement
    normalized by evaluation cost. The exploration parameter gamma
    decays smoothly with remaining budget (inferred from mean cost
    as a proxy for how much of the 30-cost budget has been used).
    
    Parameters:
    -----------
    mu : array of shape (2000,) - GP mean predictions
    sigma : array of shape (2000,) - GP standard deviation predictions
    f_best : float - best observed value
    cost : array of shape (2000,) - cost of evaluating each candidate
    
    Returns:
    --------
    scores : array of shape (2000,) - acquisition scores (higher = better)
    """
    # Deadline guard
    if (time.time() - _START_TIME) > _DEADLINE:
        return -mu / (np.maximum(cost, 1e-9) + 1.0)
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    n = len(mu)
    
    # Ensure finite inputs
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sigma = np.where(np.isfinite(sigma), np.maximum(sigma, 1e-10), 1e-10)
    cost = np.where(np.isfinite(cost), np.maximum(cost, 1e-10), 1.0)
    
    # --- Phase inference ---
    # The loop starts with 5 uniform random points (cost ~5 * 1.5 = 7.5 avg)
    # and runs while total cost < 30. So we have roughly 22.5 cost units
    # for the BO loop itself. Each evaluation costs 1 + mean(x/10) ~ 1.5.
    # So roughly 15 evaluations in the loop.
    # We can't know the exact iteration, but the GP uncertainty pattern
    # tells us: early = high sigma everywhere, late = sigma concentrated
    # in unexplored regions.
    # Use the fraction of candidates with high sigma as a proxy for "early-ness".
    sigma_thresh = np.percentile(sigma, 75)
    high_sigma_frac = np.mean(sigma > sigma_thresh) if sigma_thresh > 1e-10 else 1.0
    
    # Also use the spread of mu: early = mu is noisy, late = mu is sharp
    mu_spread = np.std(mu)
    
    # Phase factor: 1.0 = fully exploratory (start), 0.0 = fully exploitative (end)
    # Use a sigmoid on mu_spread relative to the problem scale
    # Shekel 4d range is roughly [-10.5, 0], so std of ~2-3 is "early"
    phase = 1.0 / (1.0 + np.exp(3.0 * (mu_spread - 1.5)))
    # phase ~ 1 when mu_spread >> 1.5 (early), ~0 when mu_spread << 1.5 (late)
    # Clamp to [0, 1]
    phase = np.clip(phase, 0.0, 1.0)
    
    # --- Gamma-EI computation ---
    # gamma-EI: E[max(0, f_best - gamma * f(x))]
    # For f(x) ~ N(mu, sigma^2):
    # gamma-EI = (f_best - gamma*mu) * Phi((f_best - gamma*mu)/(gamma*sigma))
    #              + gamma*sigma * phi((f_best - gamma*mu)/(gamma*sigma))
    # where gamma controls exploration: gamma=1 is standard EI,
    # gamma < 1 rewards more aggressive exploration.
    
    # Exploration parameter: interpolate from exploratory to exploitative
    # gamma = 1.0 means standard EI (exploit improvements)
    # gamma = 0.3 means very exploratory (reward high sigma even without clear improvement)
    gamma = 0.3 + 0.7 * (1.0 - phase)  # early: gamma~1.0 (EI-like), late: gamma~0.3 (exploratory)
    # Wait, that's backwards. Let me reconsider:
    # Early on: we want to explore -> lower gamma (reward uncertainty more)
    # Late on: we want to exploit -> higher gamma (reward clear improvements)
    # So: gamma = 0.3 + 0.7 * phase  where phase=1 early, phase=0 late
    # Actually let me re-derive:
    # phase = 1.0 at start (high mu_spread), 0.0 at end (low mu_spread)
    # We want: start -> exploratory (low gamma), end -> exploitative (high gamma)
    # So gamma should increase as phase decreases:
    gamma = 0.3 + 0.7 * (1.0 - phase)
    # At start (phase=1): gamma = 0.3 (exploratory)
    # At end (phase=0): gamma = 1.0 (standard EI, exploitative)
    
    # Compute gamma-EI
    improvement = f_best - gamma * mu
    denom = gamma * sigma
    z = improvement / np.maximum(denom, 1e-10)
    z_clipped = np.clip(z, -50, 50)
    
    gamma_ei = improvement * _norm_cdf(z_clipped) + denom * _norm_pdf(z_clipped)
    gamma_ei = np.maximum(gamma_ei, 0.0)
    
    # --- Cost normalization ---
    # Normalize by cost: prefer cheaper evaluations
    # Use a soft normalization: divide by (cost + epsilon) to avoid division by zero
    cost_norm = gamma_ei / (cost + 0.5)
    
    # --- Additional cost-awareness: penalize expensive points when EI is low ---
    # If a point has high cost but low EI, it should be strongly discouraged.
    # The division above handles this, but let's add a small penalty for
    # very expensive points in the early phase (when we can afford exploration).
    # In the late phase, we should be more willing to spend on promising points.
    cost_penalty_weight = 0.5 * phase  # more penalty early
    cost_penalty = cost_penalty_weight * (cost - 1.0) / (np.std(cost) + 1e-10)
    
    scores = cost_norm - cost_penalty
    
    # Ensure finite output
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
