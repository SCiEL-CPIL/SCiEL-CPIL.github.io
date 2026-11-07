"""Guided local search penalty rule for the TSP (hybrid targeted and historical penalty)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_DEADLINE_S = 300.0
_START = None


def _elapsed():
    global _START
    if _START is None:
        _START = time.monotonic()
    return time.monotonic() - _START


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    # Deadline guard: if we ever approach the limit, return the safe base matrix.
    if _elapsed() > _DEADLINE_S - 30.0:
        return np.asarray(edge_distance, dtype=float).copy()

    D = np.asarray(edge_distance, dtype=float)
    if D.ndim != 2 or D.shape[0] != D.shape[1] or D.shape[0] < 3:
        return D.copy()

    n = D.shape[0]

    # Guard against non-finite input; fall back to the (clean) base matrix.
    if not np.all(np.isfinite(D)):
        base = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
        return base

    used = np.asarray(edge_n_used, dtype=float)
    if used.shape != D.shape:
        used = np.zeros_like(D)
    if not np.all(np.isfinite(used)):
        used = np.nan_to_num(used, nan=0.0, posinf=0.0, neginf=0.0)
    used = np.maximum(used, 0.0)

    # --- Step 1: Identify edges in the current local optimum tour ---
    # local_opt_tour is a sequence of node indices.
    # The tour is closed: 0->1->2->...->n-1->0
    tour_mask = np.zeros_like(D, dtype=bool)
    try:
        tour = np.asarray(local_opt_tour, dtype=int)
        if tour.shape[0] == n:
            # Get unique edges in the tour
            for i in range(n):
                u = tour[i]
                v = tour[(i + 1) % n]
                if u != v:
                    tour_mask[u, v] = True
                    tour_mask[v, u] = True
    except Exception:
        # If tour parsing fails, ignore this penalty component
        tour_mask = np.zeros_like(D, dtype=bool)

    # --- Step 2: Calculate Historical Penalty/Reward ---
    # Dynamic reward-penalty hybrid:
    # - Edges with high usage get a positive penalty (discourage)
    # - Edges with low usage get a small negative offset (reward) to explore under-explored areas
    # - Current tour edges get a strong positive penalty to force diversification
    
    max_used = np.max(used)
    
    # Reference distance for scaling penalties
    ref_dist = float(np.median(D[D > 0])) if np.any(D > 0) else 1.0
    
    if max_used > 0:
        # Normalize usage counts to [0, 1]
        scaled_used = used / max_used
        
        # Historical penalty: positive for frequently used edges
        # Use a smooth sigmoid-like function to create a gradient
        k_hist = 2.0
        A_hist_pos = 0.8  # Max positive penalty for historical edges
        
        # Positive penalty component (for frequently used edges)
        hist_pos = A_hist_pos * (1.0 - np.exp(-k_hist * scaled_used))
        
        # Negative reward component (for rarely used edges)
        # Only apply reward to non-tour edges to avoid conflicting with diversification
        A_hist_neg = 0.15  # Small negative reward for under-explored edges
        
        # Reward is strongest for edges with zero or very low usage
        hist_neg = -A_hist_neg * (1.0 - scaled_used)
        
        # Combine: positive penalty dominates for high usage, negative reward for low usage
        hist_penalty = hist_pos + hist_neg * (1.0 - tour_mask)
    else:
        hist_penalty = np.zeros_like(D)

    # --- Step 3: Apply Current Tour Penalty ---
    # Strong penalty on edges in the current tour to force diversification.
    A_tour = 2.0  # Stronger penalty for current tour edges
    
    # Initialize output as a copy of D
    out = D.copy()
    
    # Calculate total penalty
    tour_penalty = np.where(tour_mask, A_tour, 0.0)
    
    total_penalty = hist_penalty + tour_penalty
    
    # Cap the total penalty: allow small negative values (reward) but limit positive
    # Max penalty of 2.5 means distances can at most 2.5x
    # Min penalty of -0.2 means distances can be reduced by up to 20%
    total_penalty = np.clip(total_penalty, -0.2, 2.5)
    
    # Apply penalty by scaling
    out = D * (1.0 + total_penalty)

    # Enforce exact symmetry
    out = 0.5 * (out + out.T)

    # Ensure non-negative distances (safety)
    out = np.maximum(out, 0.0)

    if not np.all(np.isfinite(out)):
        base = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
        out = 0.5 * (base + base.T)

    return out
# EVOLVE-BLOCK-END
