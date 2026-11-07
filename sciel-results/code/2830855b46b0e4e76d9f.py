"""Suite task: heuristic scores for ant-colony knapsack (5 constraints, 100 items)."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, weights):
    """Return one attractiveness score per item (larger = more attractive).

    Design (competitive-exclusion / opportunity-cost):
      * prize density  : total prize / total weight  (marginal utility per unit resource)
      * per-constraint scarcity: how 'free' each constraint is for this item
        (1 - w_ij / capacity). Tight constraints penalize heavy items.
      * competitive exclusion: for each constraint j, an item is penalized in
        proportion to the total weight of the *other* items on that constraint,
        so items that would crowd out many alternatives in a scarce dimension
        are less attractive.
      * weighted harmonic mean over constraints (weights = item weight share)
        so the tightest / most-crowded constraint dominates the score.

    Guarantees: finite positive output of shape (100,) for any finite input.
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    n_c, n_items = weights.shape

    if not np.all(np.isfinite(prizes)) or not np.all(np.isfinite(weights)):
        return np.ones(n_items, dtype=np.float64)

    # --- basic per-item prize / weight ---
    total_prize = np.maximum(prizes.sum(axis=0), 1e-12)
    total_weight = np.maximum(weights.sum(axis=0), 1e-12)
    prize_density = total_prize / total_weight                      # (n_items,)

    # per-constraint weight share (used as harmonic-mean weights)
    wsum_c = weights.sum(axis=1)                                     # (n_c,)
    wsum_c_safe = np.maximum(wsum_c, 1e-12)
    share = weights / wsum_c_safe[:, None]                           # (n_c, n_items)

    # per-constraint capacity
    cap = 25.0

    # --- per-constraint scarcity factor for each item ---
    # scarcity_ij = 1 - w_ij / cap, clipped to [1e-9, 1]
    scarcity = 1.0 - weights / cap                                   # (n_c, n_items)
    scarcity = np.clip(scarcity, 1e-9, 1.0)

    # --- competitive exclusion penalty ---
    # For constraint j, sum of weights of all items except item i:
    #   others_ij = wsum_c[j] - w_ij
    # Normalize by wsum_c[j] to get a relative crowding in [0,1].
    others = wsum_c[:, None] - weights                                # (n_c, n_items)
    others_rel = np.maximum(others, 0.0) / wsum_c_safe[:, None]       # (n_c, n_items) in [0,1]
    # Heavier crowding -> lower attractiveness. Use (1 - others_rel)^alpha.
    alpha = 0.5
    comp = np.power(1.0 - others_rel, alpha)                          # (n_c, n_items)

    # Combine scarcity and competitive factor per constraint
    per_c = scarcity * comp                                            # (n_c, n_items)

    # --- weighted harmonic mean over constraints, weights = share ---
    # harmonic = 1 / sum_j ( share_ij / per_c_ij )
    denom = np.sum(share / np.maximum(per_c, 1e-12), axis=0)          # (n_items,)
    harmonic = 1.0 / np.maximum(denom, 1e-12)

    score = prize_density * harmonic

    # --- sanitize: ensure positive, finite, correct shape ---
    score = np.where(np.isfinite(score), score, 1.0)
    score = np.maximum(score, 1e-9)

    if score.shape != (n_items,):
        score = np.ones(n_items, dtype=np.float64)

    return score
# EVOLVE-BLOCK-END
