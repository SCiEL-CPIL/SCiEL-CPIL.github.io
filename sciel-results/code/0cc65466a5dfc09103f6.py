"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Track start time for deadline guard
_START_TIME = None
_DEADLINE_MARGIN = 5.0  # seconds before the 150s limit
_MAX_EVAL_TIME = 150.0 - _DEADLINE_MARGIN

# Weibull parameters for estimating fill probability of remainders
# These are tuned for typical Weibull sequences with capacity 100
# k=2.0, lambda=15.0 implies mean ~ 15 * Gamma(1.5) = 15 * 0.886 = 13.3
_WEIBULL_K = 2.0
_WEIBULL_LAMBDA = 15.0


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.
    
    Strategy: Adversarial robustness via maximum future regret minimization.
    
    We model the bin packing process as a game-theoretic adversarial robustness problem.
    Each bin's score is the negative of the maximum possible future regret, defined as
    the size of the largest Weibull-distributed item that could fit into the resulting
    remainder. This prioritizes bins that create remainders robust against the worst-case
    immediate future demand.
    
    The idea is:
    1. For a given remainder r after placing an item, the "regret" is the expected size
       of the next item that could fit into that remainder but might not be optimally placed.
    2. We approximate this by considering the CDF of the Weibull distribution truncated
       to [0, r], which gives the probability that a random item fits in the remainder.
    3. The expected size of items that fit is the conditional expectation of the Weibull
       distribution given that the item is <= r.
    4. To make this computationally efficient, we use a smooth surrogate: the integral of
       the Weibull PDF from 0 to r, which is the CDF F(r).
    5. The regret is proportional to r * (1 - F(r)) + integral_0^r x * f(x) dx, but we
       simplify to a smooth function that captures the essence: r * S(r) where S(r) is
       the survival function, blended with a term that encourages consolidation.
    
    However, for the adversarial interpretation, we consider the worst-case immediate
    future item that could be placed in the remainder. The size of the largest possible
    item that fits is r itself (if we ignore distribution), but we weight this by the
    probability that such an item actually arrives, which is related to the survival
    function S(r) = P(X > r).
    
    A better adversarial metric: the expected "waste" if the next item that fits into
    the remainder is the largest possible one. This is complex, so we use a proxy:
    
    Regret(r) = r * S(r) * (1 + gamma * r)
    
    where gamma is a small constant that adds a penalty for large remainders to encourage
    consolidation. The term r * S(r) captures the "exposure" of the remainder to future
    items that are too large to fit, while the (1 + gamma * r) term adds a quadratic
    penalty for large gaps.
    
    Actually, let's refine: the adversarial regret should reflect the fact that a large
    remainder is risky because it might be filled by a suboptimal sequence of items.
    We use:
    
    Regret(r) = r * S(r) + delta * r^2 * S(r)^2
    
    where delta is a small constant. The second term penalizes large remainders more
    heavily, as the risk of fragmentation grows quadratically with the gap size.
    
    Score = -Regret(r)
    
    This creates a preference for remainders that are either very small (Best Fit) or
    very large (Worst Fit), with a stronger penalty for large remainders than before.
    """
    global _START_TIME
    
    # Initialize timing if not started
    if _START_TIME is None:
        _START_TIME = time.time()
        
    # Deadline guard
    elapsed = time.time() - _START_TIME
    if elapsed > _MAX_EVAL_TIME:
        # Fallback to simple Best Fit to ensure completion
        remaining = bins - item
        valid = remaining >= 0
        scores = np.full(len(bins), -np.inf)
        if np.any(valid):
            scores[valid] = -remaining[valid]
        return scores

    # Calculate remaining capacity
    remaining = bins - item
    
    # Identify valid bins (enough capacity)
    valid = remaining >= 0
    
    # Initialize scores with -inf for invalid bins
    scores = np.full(len(bins), -np.inf)
    
    if not np.any(valid):
        return scores
        
    valid_rems = remaining[valid]
    
    # Handle r=0 case and clip to non-negative
    r = np.maximum(valid_rems, 0.0)
    
    # Calculate survival probability S(r) = exp(-(r/lambda)^k)
    ratio = r / _WEIBULL_LAMBDA
    power_term = np.power(ratio, _WEIBULL_K)
    survival = np.exp(-power_term)
    
    # Adversarial regret: r * S(r) + delta * r^2 * S(r)^2
    delta = 0.05  # Small penalty coefficient for large remainders
    
    regret = r * survival + delta * np.power(r * survival, 2)
    
    # Score: minimize regret
    scores[valid] = -regret
    
    return scores
# EVOLVE-BLOCK-END
