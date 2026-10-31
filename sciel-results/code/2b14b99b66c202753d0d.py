"""Structural change: lookahead-aware scoring using observed item-size statistics."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_START_TIME = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill

# Track observed item sizes to estimate future fillability
_item_history = []
_MAX_HISTORY = 200  # keep recent items for fast statistics

def priority(item: float, bins: np.ndarray) -> np.ndarray:
    # Deadline guard
    if time.time() > _DEADLINE:
        remaining = bins - item
        scores = np.full(len(bins), -1e9, dtype=float)
        valid = remaining >= 0
        scores[valid] = -remaining[valid]
        return scores

    # Update item history
    _item_history.append(item)
    if len(_item_history) > _MAX_HISTORY:
        _item_history.pop(0)

    remaining = bins - item
    valid_mask = remaining >= 0
    scores = np.full(len(bins), -1e9, dtype=float)

    if not np.any(valid_mask):
        return scores

    valid_remaining = remaining[valid_mask]
    n_valid = np.sum(valid_mask)

    # Estimate typical future item size from history
    if len(_item_history) >= 10:
        recent = np.array(_item_history[-min(len(_item_history), 100):])
        median_item = np.median(recent)
        mean_item = np.mean(recent)
        std_item = np.std(recent)
    else:
        median_item = item
        mean_item = item
        std_item = max(item * 0.3, 1.0)

    # Core idea: score each bin by how "wasteful" the placement is.
    # After placing item, the bin has 'remaining' capacity left.
    # That leftover capacity will need to be filled by future items.
    # If leftover is large relative to typical item size, it's "wasteful"
    # because it may require opening a new bin later.
    # If leftover is small, it's efficient (tight fit).
    #
    # But we also want to avoid creating bins with tiny slivers that
    # can't fit any future item (dead capacity).
    #
    # Score = -waste_penalty
    # waste_penalty combines:
    #   1. Slack ratio: leftover / typical_item_size (higher = more waste)
    #   2. Dead-capacity risk: if leftover < typical_item_size but > 0,
    #      it's partially wasted (can't fit a typical item)
    #   3. A small bonus for bins that are "just right" (leftover close
    #      to a multiple of typical item size, so future items can fill it)

    # Normalize leftover by typical item size
    slack_ratio = valid_remaining / (mean_item + 1e-9)

    # Penalty 1: raw slack (more leftover = more potential waste)
    # Use a soft penalty that grows superlinearly for large slack
    penalty_slack = slack_ratio ** 1.5

    # Penalty 2: dead capacity risk
    # If leftover is between 0 and ~0.7*mean_item, it's likely dead space
    # (too small for a typical item but not zero)
    dead_zone = np.where(
        (valid_remaining > 0) & (valid_remaining < 0.7 * mean_item),
        1.0 - valid_remaining / (0.7 * mean_item + 1e-9),
        0.0
    )
    penalty_dead = dead_zone * 0.5

    # Bonus: alignment with typical item size
    # If leftover is close to k * mean_item for some integer k, future items
    # can pack it well. Compute distance to nearest multiple.
    if mean_item > 1e-9:
        k_nearest = np.round(valid_remaining / mean_item)
        aligned = np.abs(valid_remaining - k_nearest * mean_item) / (mean_item + 1e-9)
        # Bonus is higher when well-aligned (small distance)
        bonus_align = np.exp(-2.0 * aligned) * 0.3
    else:
        bonus_align = np.zeros_like(valid_remaining)

    # Combine: higher score = better
    # We want to MINIMIZE waste, so score = -(penalties) + bonus
    waste = penalty_slack + penalty_dead - bonus_align
    scores_valid = -waste

    # Add tiny tiebreaker: prefer slightly larger remaining to avoid
    # premature closure when scores are nearly equal
    scores_valid += 0.005 * (valid_remaining / 100.0)

    scores[valid_mask] = scores_valid
    return scores
# EVOLVE-BLOCK-END
