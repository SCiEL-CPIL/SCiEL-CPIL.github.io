"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import hashlib
from collections import defaultdict

_START_TIME = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill

# Q-learning state
# State: sorted tuple of bin remainders (rounded to 2 decimal places for stability)
# Action: index of the bin selected
# Reward: -1 (for each new bin opened) or 0 (otherwise). We want to minimize bins used.
# However, the priority function is called per item. We don't see the "new bin opened" signal directly.
# We can infer it: if the best available bin was "full" (0 remaining) or close to it? 
# No, the interface is `priority(item, bins) -> scores`. The evaluator picks the argmax.
# We don't control the placement, we just suggest scores.
# Wait, the contract says: "each arriving item goes to the highest-scoring bin".
# So the evaluator implements the placement. We just return scores.
# We cannot observe the result of the placement (which bin was actually used if ties, or if my score was ignored? No, "goes to the highest-scoring bin").
# So we know which bin was used: the one with the highest score we returned.
# But we don't know if a new bin was opened. 
# Actually, if the bins array changes size, a new bin was opened. But we only see the current bins.
# We can track the history of bins to detect new bins.

# Let's track the state history to detect new bins.
_last_bins_state = None
_q_table = defaultdict(float)
_alpha = 0.1
_gamma = 0.9
_epsilon = 0.1  # Exploration rate

def _get_state(bins):
    # Round bins to 1 decimal place to reduce state space and handle float noise
    # Sort to make state invariant to bin order? 
    # The bins are passed as an array. The indices matter for the action.
    # If we sort, we lose the index mapping.
    # We must use the raw bins but rounded.
    # To keep state space manageable, we can discretize.
    # Bins have capacity 100. 
    # State: tuple(sorted(round(b, 1) for b in bins))
    # But action is index. If we sort state, we can't map action back easily unless we track permutation.
    # Simpler: State is the tuple of rounded bins in their original order.
    # This state space is huge.
    # Alternative: Use a feature-based Q-learning or just a heuristic that adapts?
    # Given the 150s limit and 1000 items, a full MDP is risky if state space explodes.
    # Let's try a simpler adaptive heuristic: 
    # The previous "Weibull CDF" strategy scored 1.031.
    # Let's try to combine Best Fit with a dynamic penalty based on the distribution of remaining capacities.
    # If many bins have small remainders, prefer Best Fit.
    # If many bins have large remainders, maybe First Fit Decreasing logic?
    
    # Actually, let's look at the "Weibull CDF" inspiration. It scored 1.031.
    # The task is Weibull sequences.
    # Weibull(shape, scale). 
    # If shape < 1, it's decreasing hazard. If shape > 1, increasing.
    # Common Weibull for sizes: shape ~ 2-5, scale such that mean is reasonable.
    
    # Let's try a different family: "Next Fit Decreasing" approximation.
    # But items arrive online.
    
    # Let's try: Score = -remaining - beta * (number of bins with remaining < threshold)
    # This encourages keeping bins "full" to avoid opening new ones?
    
    # Let's stick to the directive: "Model the problem as a Markov Decision Process... score each bin based on the expected future regret... approximated by the probability that this remainder cannot be perfectly filled by items from the known Weibull distribution."
    
    # The directive explicitly suggests: Score = -CDF(remainder) or similar.
    # The inspiration says: "Replaces linear Best-Fit with a Weibull CDF-based score... Measured: 1 -> 1.031".
    # So the Weibull CDF approach is a proven improvement over the seed (1.0).
    # The current program is a quadratic penalty (score 1.0).
    # So switching to Weibull CDF is a different strategy and has been measured to improve.
    
    # However, I must write a COMPLETE program. I can implement the Weibull CDF strategy directly.
    # What are the Weibull parameters?
    # The task says "five generated Weibull sequences". It doesn't give parameters.
    # I need to estimate them or use a robust default.
    # Or, I can use a generic "fill probability" model.
    # If items are Weibull, the probability that a random item fits in remainder R is CDF(R).
    # We want to maximize the chance that the remainder gets filled.
    # So we want to choose the bin that maximizes CDF(remainder)?
    # No, if CDF(remainder) is high, it means many items are smaller than remainder.
    # If we put an item in a bin with large remainder, we create a new remainder.
    # We want the new remainder to be "good".
    # A "good" remainder is one that is likely to be filled by future items.
    # So we want to maximize P(future item fits in new_remainder).
    # New remainder = bin_capacity - item - current_used.
    # Let r = bins[i] - item.
    # Score[i] = P(future item <= r) = CDF_Weibull(r).
    # We want to MAXIMIZE this score?
    # If CDF(r) is high, it means r is large (since CDF is increasing).
    # Wait, if r is large, CDF(r) is close to 1.
    # If r is small, CDF(r) is small.
    # We want to avoid small remainders that can't be filled?
    # Or do we want to avoid large remainders that waste space?
    # L1 lower bound is sum(items)/capacity.
    # Excess is (bins_used * capacity - sum(items)) / (bins_used * capacity)? No, "excess of bins used over the L1 lower bound".
    # Excess = (bins_used - sum(items)/capacity) / (sum(items)/capacity) * 100?
    # Or just (bins_used - LB) / LB?
    # Minimizing bins_used is the goal.
    
    # If we choose a bin with large remaining space, we might waste space if that space isn't filled.
    # If we choose a bin with small remaining space (Best Fit), we might create a tiny remainder that is useless, forcing a new bin soon.
    # The "Weibull CDF" insight from the observation: "penalizes large remainders more heavily... or differently... favoring remainders that are statistically likely to be filled".
    # If we favor remainders likely to be filled, we should prefer remainders r where CDF(r) is high?
    # No, if CDF(r) is high, it means the remainder is large.
    # If CDF(r) is low, it means the remainder is small.
    # If the remainder is small, it's unlikely to be filled by a random item (which might be larger).
    # So a small remainder is "bad" if it can't be filled.
    # A large remainder is "bad" if it wastes space.
    
    # Let's look at the specific phrasing: "score as the negative of the cumulative distribution function".
    # Score = -CDF(r).
    # If r is small, CDF(r) is small, Score is near 0 (high).
    # If r is large, CDF(r) is large, Score is very negative (low).
    # This effectively implements BEST FIT, but with a non-linear penalty for large remainders.
    # Best Fit minimizes r. 
    # -CDF(r) also minimizes r, but the penalty for large r is capped at -1 (if CDF->1).
    # The penalty for small r is shallow.
    # This is different from linear Best Fit (-r) which has unbounded penalty for large r.
    # This suggests that large remainders are "okay" up to a point, but small remainders are preferred.
    
    # Let's implement this.
    # We need Weibull parameters.
    # I will estimate them from the first few items if possible, or use a default.
    # Since `priority` is called per item, I can maintain a running estimate.
    
    pass

# Global state for adaptive Weibull parameters
_item_history = []
_weibull_shape = 2.0  # Default
_weibull_scale = 20.0 # Default
_params_estimated = False

def _estimate_weibull_params(items):
    """
    Estimate Weibull shape and scale from a list of items.
    Uses MLE approximation or simple heuristic.
    """
    if len(items) < 10:
        return 2.0, 20.0
    
    items = np.array(items)
    items = items[items > 0]
    if len(items) == 0:
        return 2.0, 20.0
        
    # Simple heuristic: 
    # Weibull mean = scale * Gamma(1 + 1/shape)
    # Weibull var = scale^2 * (Gamma(1+2/shape) - Gamma(1+1/shape)^2)
    # This is complex to invert.
    # Let's use a fixed reasonable default for general Weibull item sizes in bin packing.
    # If capacity is 100, items are likely 1-50.
    # Shape 2, Scale 20 -> Mean ~ 17.7. Median ~ 14.
    # This seems reasonable.
    return 2.0, 20.0

def _weibull_cdf(x, shape, scale):
    """
    Calculate CDF of Weibull distribution.
    CDF(x) = 1 - exp(-(x/scale)^shape)
    """
    if x <= 0:
        return 0.0
    # Clip x to avoid overflow in exp
    # (x/scale)^shape
    ratio = x / scale
    # If ratio is very large, CDF is 1
    if ratio > 100:
        return 1.0
    return 1.0 - np.exp(-np.power(ratio, shape))

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    global _item_history, _weibull_shape, _weibull_scale, _params_estimated
    
    # Deadline guard
    if time.time() - _START_TIME > _DEADLINE:
        valid = bins >= item
        if not np.any(valid):
            return np.full_like(bins, -np.inf)
        scores = -(bins - item)
        scores[~valid] = -np.inf
        return scores
    
    # Update item history for parameter estimation (optional, keep light)
    _item_history.append(item)
    if len(_item_history) > 100:
        _item_history.pop(0)
    
    # Estimate parameters if not done yet
    if not _params_estimated and len(_item_history) >= 20:
        _weibull_shape, _weibull_scale = _estimate_weibull_params(_item_history)
        _params_estimated = True
        
    n = len(bins)
    scores = np.full(n, -1e18)
    
    # Calculate remaining capacity after placement
    remaining = bins - item
    
    # Only consider bins where remaining >= 0
    valid_mask = remaining >= 0
    if not np.any(valid_mask):
        return scores
    
    valid_indices = np.where(valid_mask)[0]
    valid_remaining = remaining[valid_indices]
    
    # Strategy: Score = -CDF_Weibull(remaining)
    # This favors remainders that are small (low CDF) -> High Score.
    # It penalizes large remainders (high CDF) -> Low Score.
    # This is a "soft" Best Fit.
    
    # Vectorized CDF calculation
    # Handle remaining == 0
    # CDF(0) = 0. Score = 0.
    # CDF(large) = 1. Score = -1.
    
    # We want to maximize score.
    # So we prefer bins where CDF(remaining) is LOW.
    # i.e., remaining is SMALL.
    
    # Add a small epsilon to avoid log(0) or similar issues, though CDF is safe.
    # We can also add a tie-breaker: if CDFs are close, prefer the one with smaller remaining (linear).
    
    cdf_vals = _weibull_cdf(valid_remaining, _weibull_shape, _weibull_scale)
    
    # Primary score: -CDF
    # Secondary score: -remaining (linear best fit) to break ties or refine
    # We can combine them: Score = -CDF - epsilon * remaining
    
    epsilon = 0.01
    scores[valid_indices] = -cdf_vals - epsilon * valid_remaining
    
    return scores

# EVOLVE-BLOCK-END
