"""Guided local search penalty rule for the TSP (direct approach)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = time.monotonic()
_DEADLINE = 325.0  # safety margin before 330s kill


def _elapsed():
    return time.monotonic() - _start_time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Reward-penalty hybrid: under-explored edges get a small negative offset
    (reward) to accelerate convergence on promising structural motifs, while
    current-tour edges receive a large positive penalty to force diversification.
    
    Key ideas:
    1. Infer current round from max usage count in edge_n_used
    2. Edges with usage below round index get a small reward (negative offset)
    3. Current-tour edges get a large penalty to break the local optimum
    4. Heavily-used non-tour edges still get a moderate penalty
    5. All scaled by reference distance for adaptivity
    """
    # Deadline guard
    if _elapsed() > _DEADLINE:
        return np.asarray(edge_distance, dtype=float).copy()
    
    D = np.asarray(edge_distance, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1] or D.shape[0] < 3:
        return D.copy()
    
    n = D.shape[0]
    
    # Guard against non-finite input
    if not np.all(np.isfinite(D)):
        base = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
        return base
    
    used = np.asarray(edge_n_used, dtype=float)
    if used.shape != D.shape:
        used = np.zeros_like(D)
    if not np.all(np.isfinite(used)):
        used = np.nan_to_num(used, nan=0.0, posinf=0.0, neginf=0.0)
    used = np.maximum(used, 0.0)
    
    # Compute reference scale: median edge length in the full matrix
    ref = float(np.median(D))
    if not np.isfinite(ref) or ref <= 0.0:
        ref = 1.0
    
    # Infer current round from usage history
    # The max usage count approximates the number of completed rounds
    max_used = float(used.max())
    # The "current round index" is approximately max_used (1-indexed rounds)
    # Use a smoothed estimate: round = max_used if max_used > 0, else 1
    round_idx = max(1.0, max_used)
    
    # --- REWARD for under-explored edges ---
    # Edges with usage < round_idx are under-explored relative to the search progress
    # Apply a small negative offset to pull 2-opt toward these edges
    # Use a smooth function: reward magnitude grows as usage falls below round_idx
    # reward = -reward_rate * ref * (1 - used/round_idx) for used < round_idx
    # This is 0 at used=round_idx and -reward_rate*ref at used=0
    reward_rate = 0.05
    under_explored = np.clip(1.0 - used / round_idx, 0.0, 1.0)
    reward = -reward_rate * ref * under_explored
    
    # --- PENALTY for heavily-used edges ---
    # Edges with usage >= round_idx are over-explored; apply soft penalty
    # penalty = penalty_rate * ref * (used / (1 + used)) for used > 0
    penalty_rate = 0.25
    penalty = penalty_rate * ref * used / (1.0 + used)
    
    # Combine reward and penalty (reward can be negative, penalty is positive)
    adjustment = reward + penalty
    
    # --- TOUR EDGE PENALTY (large, to force diversification) ---
    tour = np.asarray(local_opt_tour, dtype=int)
    tour_boost = 0.4 * ref
    
    if tour.ndim == 1 and len(tour) >= 2:
        for i in range(len(tour) - 1):
            a, b = tour[i], tour[i + 1]
            if 0 <= a < n and 0 <= b < n:
                adjustment[a, b] += tour_boost
                adjustment[b, a] += tour_boost
        # Close the tour
        a, b = tour[-1], tour[0]
        if 0 <= a < n and 0 <= b < n:
            adjustment[a, b] += tour_boost
            adjustment[b, a] += tour_boost
    
    # Hard bounds to prevent extreme distortions
    max_pos = 0.8 * ref
    max_neg = -0.1 * ref  # allow small rewards but not too large
    adjustment = np.clip(adjustment, max_neg, max_pos)
    
    # Add adjustment to the distance matrix
    out = D + adjustment
    
    # Enforce exact symmetry
    out = 0.5 * (out + out.T)
    
    # Ensure diagonal is zero
    np.fill_diagonal(out, 0.0)
    
    # Ensure all values are positive (distances must be non-negative for valid TSP)
    out = np.maximum(out, 0.0)
    
    if not np.all(np.isfinite(out)):
        base = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
        out = 0.5 * (base + base.T)
        np.fill_diagonal(out, 0.0)
    
    return out
# EVOLVE-BLOCK-END
# EVOLVE-BLOCK-END
