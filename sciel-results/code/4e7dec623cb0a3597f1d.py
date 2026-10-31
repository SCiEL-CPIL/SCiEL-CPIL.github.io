"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state for the adaptive bandit strategy
# We will learn a weight vector w = [w_best, w_worst, w_next] 
# that determines how much to prefer tight fits, loose fits, or fresh bins.
# This allows the strategy to adapt to the specific distribution of items and current bin states.

# Initialize weights. Start with a bias towards Best-Fit (tight) as it's generally strong, 
# but allow exploration.
_w = np.array([1.0, 0.0, 0.0])  # [best_fit, worst_fit, next_fit]
# Learning rate for the bandit updates
_lr = 0.05
# Exploration rate (epsilon-greedy)
_epsilon = 0.1
# Time tracking for safety
_t0 = time.time()
# Deadline guard: 120s limit, keep 5s margin
_DEADLINE = 115.0

def _compute_scores(item, bins, w):
    """
    Compute scores for all bins given current weights.
    Higher score is better.
    """
    n = len(bins)
    scores = np.full(n, -1e18)
    
    # Calculate remaining capacity after placement
    remaining = bins - item
    
    # Only consider bins where remaining >= 0
    valid_mask = remaining >= 0
    if not np.any(valid_mask):
        return scores, None
    
    valid_indices = np.where(valid_mask)[0]
    valid_remaining = remaining[valid_mask]
    valid_bins = bins[valid_mask]
    
    # 1. Best-Fit Score: Prefer bins with smallest remaining space (tightest fit)
    # Normalize remaining to [0, 1] based on max remaining in valid set for stability
    max_rem = np.max(valid_remaining)
    if max_rem > 0:
        norm_rem = valid_remaining / max_rem
    else:
        norm_rem = np.zeros_like(valid_remaining)
    
    # Best fit: minimize remaining -> score = -remaining
    # To make it scale-invariant and comparable to others, we use a normalized penalty
    # But raw remaining is also important. Let's use a mix.
    # Actually, simple best fit is just -remaining.
    # Let's define components in a comparable range [0, 1] roughly.
    
    # Component 1: Best Fit (tightness)
    # Higher is better. If remaining is 0, score 1. If remaining is max, score 0.
    score_best = 1.0 - norm_rem
    
    # Component 2: Worst Fit (spread)
    # Higher is better. If remaining is max, score 1. If remaining is 0, score 0.
    # This helps to fill empty bins and avoid fragmentation if items are large.
    score_worst = norm_rem
    
    # Component 3: Next Fit (freshness)
    # Higher is better if the bin is empty or nearly empty (high original capacity used? No, low used).
    # Next fit usually means picking the first available bin. 
    # A proxy for "freshness" or "low fragmentation" is high remaining capacity relative to total?
    # Or simply, prefer bins with high remaining capacity to keep them "open" for larger future items?
    # Actually, Next Fit in bin packing often correlates with leaving bins "clean".
    # Let's use "Empty-ness" as a proxy for Next-Fit-like behavior: prefer bins with high remaining capacity.
    # Wait, Worst Fit already prefers high remaining. 
    # Let's distinguish: 
    # Best Fit: Minimize waste in current bin (tight).
    # Worst Fit: Maximize remaining in current bin (spread load).
    # "Next Fit" heuristic often just picks the first bin. In a score-based system, 
    # we can emulate "freshness" by preferring bins that have the least total used space? 
    # Or simply, let's use a third dimension: "Stability".
    # Let's stick to Best and Worst for now, and add a "Random/Uniform" component if needed, 
    # or a "Middle Fit" (First Fit Decreasing style logic isn't possible online).
    
    # Let's refine the third component: "Fragmentation Penalty"
    # If a bin is half-full, it's risky. If it's 10% or 90%, it's better.
    # So we prefer bins that are either very empty or very full.
    # This is a "U-shaped" preference.
    # score_u = 1 - 4 * norm_rem * (1 - norm_rem)
    # This peaks at 0 and 1, and is 0 at 0.5.
    score_u = 1.0 - 4.0 * norm_rem * (1.0 - norm_rem)
    
    # Combine weighted sum
    # w = [w_best, w_worst, w_u]
    combined = w[0] * score_best + w[1] * score_worst + w[2] * score_u
    
    scores[valid_indices] = combined
    
    # Determine the best bin based on current weights (greedy step)
    # If epsilon-greedy, we might pick a random valid bin.
    # But the function signature requires returning scores. The caller picks max.
    # So we just return the scores. The "bandit" part needs to know which bin was picked to update.
    # However, the function `priority` is stateless in terms of "what was picked" because it only sees the state *before* the pick.
    # To make this a true bandit, we need to know the outcome. 
    # We can infer the outcome: The bin with the highest score will be picked.
    # So we can calculate the "reward" of that pick immediately?
    # Reward definition: How good was this pick?
    # A good pick minimizes the "regret" of the remaining space.
    # Let's define reward as: - (remaining_after_pick / item_size) ? No.
    # Let's define reward based on how "tight" the fit was compared to the best possible fit.
    # Best possible fit for this item is the minimum remaining among all valid bins.
    # If we pick the one with min remaining, reward is high.
    
    best_remaining = np.min(valid_remaining)
    picked_remaining = valid_remaining[np.argmax(combined)]
    
    # Reward: 1 if picked the tightest, 0 if picked the loosest (among valid).
    # Normalized reward
    if best_remaining == picked_remaining:
        reward = 1.0
    else:
        # Scale by how close we were
        # (max_rem - picked_rem) / (max_rem - best_rem)
        if max_rem > best_remaining:
            reward = (max_rem - picked_remaining) / (max_rem - best_remaining)
        else:
            reward = 0.5
            
    # Update weights based on which component contributed most to the high score?
    # This is a multi-armed bandit on the weights.
    # A simpler approach: Gradient ascent on the reward.
    # d(reward)/d(weights) is hard to compute analytically.
    # Instead, let's use a simpler heuristic: 
    # If the pick was "tight" (low remaining), increase w_best.
    # If the pick was "loose" (high remaining), increase w_worst.
    # If the pick was "middle", increase w_u.
    
    # Determine which strategy "won"
    if picked_remaining < 0.3 * max_rem:
        # Tight fit
        _w[0] += _lr * reward
    elif picked_remaining > 0.7 * max_rem:
        # Loose fit
        _w[1] += _lr * reward
    else:
        # Middle fit
        _w[2] += _lr * reward
        
    # Regularization: decay weights towards uniform or previous state to prevent drift
    _w = _w * 0.99 + np.array([1.0, 0.0, 0.0]) * 0.01
    # Normalize weights to sum to 1 to keep scale stable
    sum_w = np.sum(_w)
    if sum_w > 0:
        _w /= sum_w
        
    return scores, reward

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    Higher score is preferred.
    
    Strategy: Adaptive Bandit that learns a mix of Best-Fit, Worst-Fit, and Middle-Fit (U-shaped)
    preferences based on the quality of the previous pick.
    """
    global _w, _t0, _DEADLINE
    
    # Check time
    if time.time() - _t0 > _DEADLINE:
        # Fallback to simple best fit if we are running out of time
        scores = np.full(len(bins), -1e18)
        remaining = bins - item
        valid_mask = remaining >= 0
        if np.any(valid_mask):
            scores[valid_mask] = -remaining[valid_mask]
        return scores
    
    scores, _ = _compute_scores(item, bins, _w)
    return scores
# EVOLVE-BLOCK-END
