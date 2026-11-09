"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(item: float, bins: np.ndarray) -> np.ndarray:
    """
    Score each bin with enough remaining capacity.
    The item goes to the highest-scoring bin.

    Strategy: Fragmentation-aware scoring with stronger bimodal penalties.
    Penalizes moderate-ratio placements more aggressively while preserving
    large remaining space for future items.
    """
    t_start = time.time()
    deadline = t_start + 0.008  # 8ms budget per call with safety margin

    remaining = bins - item
    valid = remaining >= 0

    # Base score: negative remaining capacity (best-fit) — tighter is better.
    score = -remaining.copy()

    # Invalidate bins that cannot hold the item.
    score[~valid] = -np.inf

    if not np.any(valid):
        return score

    post = remaining[valid]  # remaining capacity after placing item
    bins_valid = bins[valid]

    # --- Fragmentation-aware penalty ---
    # Leftover fraction: what fraction of the bin remains after placement
    leftover_frac = np.where(bins_valid > 1e-9, post / bins_valid, 0.0)

    # Stronger bimodal penalty: penalize moderate leftover fractions more aggressively
    # Peak penalty when leftover_frac is in [0.25, 0.45]
    center = 0.35
    width = 0.08  # Narrower width for sharper peak
    bump = np.exp(-((leftover_frac - center) / width) ** 2)
    frag_penalty = 0.6 * bump  # Increased penalty strength

    # Additional: penalize very large leftovers (waste of capacity)
    large_leftover_penalty = 0.2 * np.clip(post / 100.0, 0, 1)

    # Bonus for bins already partially filled (encourage consolidation)
    util = np.where(bins_valid > 1e-9, item / bins_valid, 0.0)
    # Stronger consolidation bonus that saturates
    consolidation_bonus = 0.15 * (1.0 - np.exp(-4.0 * util))

    # Small bonus for moderate utilization to balance load
    moderate_bonus = 0.05 * util * (1.0 - util)

    # Bonus for preserving large remaining space (keep bins viable for future items)
    # This counteracts the best-fit tendency to fill bins completely
    large_space_bonus = 0.1 * np.clip((post - 30) / 70, 0, 1)

    # Combine all components
    total_adjustment = -frag_penalty - large_leftover_penalty + consolidation_bonus + moderate_bonus + large_space_bonus

    score[valid] += total_adjustment

    # Deadline guard: if we're running long, fall back to simple best-fit
    if time.time() > deadline:
        scores = np.full_like(bins, -1e9, dtype=float)
        scores[valid] = -remaining[valid]
        return scores

    return score
# EVOLVE-BLOCK-END
# EVOLVE-BLOCK-END
