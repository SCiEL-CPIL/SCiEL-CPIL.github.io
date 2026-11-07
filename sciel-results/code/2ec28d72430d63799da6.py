"""Cost-aware acquisition for Bayesian optimization (hartmann 3d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Module-level deadline guard
_PROCESS_START = time.time()
_DEADLINE_MARGIN = 5.0
_DEADLINE = _PROCESS_START + 330.0 - _DEADLINE_MARGIN


def _time_remaining():
    return _DEADLINE - time.time()


def acquisition(mu, sigma, f_best, cost):
    """
    Two-phase cost-aware acquisition:
    - Early phase (f_best > -3.5): pure EI, ignoring cost.
      Rationale: when the best is still far from the optimum, the priority
      is to find good regions quickly; cost differences are less critical.
    - Late phase (f_best <= -3.5): cost-aware EI with a sub-linear cost penalty.
      Rationale: when close to the optimum, we should prefer cheap evaluations
      to squeeze out the last bit of improvement within the budget.

    Structural change from parent: adaptive phase switching based on f_best,
    combined with a smooth blend rather than a hard switch. The parent used
    a fixed sqrt(cost) penalty always; this version uses no cost penalty in
    the early phase and a tunable cost-aware penalty in the late phase.
    """
    # Deadline guard
    if _time_remaining() < 0.5:
        z = (f_best - mu) / (sigma + 1e-9)
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        cost_safe = np.maximum(cost, 1e-9)
        score = ei / np.sqrt(cost_safe)
        return np.where(np.isfinite(score), score, 0.0)

    # Compute standard Expected Improvement
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Two-phase logic:
    # Phase transition threshold: f_best = -3.5
    # Below this, we're in the "late" phase and cost matters.
    # Above this, we're in the "early" phase and cost is ignored.
    # Use a smooth blend over a range [-3.6, -3.4] to avoid discontinuities.
    
    threshold_low = -3.6   # fully cost-aware
    threshold_high = -3.4  # fully cost-agnostic
    
    if f_best <= threshold_low:
        # Fully in late phase: strong cost awareness
        cost_weight = 1.0
    elif f_best >= threshold_high:
        # Fully in early phase: no cost awareness
        cost_weight = 0.0
    else:
        # Smooth blend
        cost_weight = (threshold_high - f_best) / (threshold_high - threshold_low)
    
    cost_safe = np.maximum(cost, 1e-9)
    
    if cost_weight > 0.0:
        # Late phase: apply cost penalty
        # Use a power-law penalty: cost^gamma where gamma is tuned
        # gamma = 0.5 (sqrt) for sub-linear penalty
        cost_penalty = np.power(cost_safe, 0.5)
        cost_aware_score = ei / cost_penalty
        
        # Blend between pure EI and cost-aware EI
        score = (1.0 - cost_weight) * ei + cost_weight * cost_aware_score
    else:
        # Early phase: pure EI
        score = ei.copy()
    
    # Ensure all scores are finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
