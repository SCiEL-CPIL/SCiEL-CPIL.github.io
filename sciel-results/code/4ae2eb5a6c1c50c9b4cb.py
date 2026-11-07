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

    # --- Step 2: Calculate Historical Penalty ---
    # Use a decaying exponential penalty for edges in edge_n_used.
    # We want to discourage recently used edges but not forbid them permanently.
    # Normalize by the max usage count to keep penalties bounded.
    max_used = np.max(used)
    if max_used > 0:
        # Scale used counts to [0, 1] roughly, then apply exponential decay
        # A smaller 'k' means faster decay of influence.
        k_hist = 1.0
        A_hist = 0.5  # Lower penalty for historical edges compared to current tour
        
        scaled_used = used / max_used
        hist_penalty = A_hist * (1.0 - np.exp(-k_hist * scaled_used * max_used)) 
        # Actually, let's keep it simple: penalty proportional to count but capped.
        # penalty = A_hist * min(used, cap)
        # Let's use the exponential form from before but with smaller A.
        # ref = float(np.median(D[used > 0])) if np.any(used > 0) else 1.0
        # hist_penalty = A_hist * (1.0 - np.exp(-k_hist * used / ref))
        
        # Simpler: linear penalty with cap
        hist_penalty = np.minimum(used, 5.0) * (A_hist / 5.0)
        # hist_penalty ranges from 0 to A_hist
    else:
        hist_penalty = np.zeros_like(D)

    # --- Step 3: Apply Current Tour Penalty ---
    # Strong penalty on edges in the current tour to force diversification.
    A_tour = 1.5  # Stronger penalty for current tour edges
    
    # Initialize output as a copy of D
    out = D.copy()
    
    # Apply historical penalty
    # Multiply distances by (1 + penalty)
    out *= (1.0 + hist_penalty)
    
    # Apply current tour penalty
    # This overrides or adds to the historical penalty?
    # GLS typically adds the penalty to the cost. So cost' = cost + penalty.
    # Here we are scaling the matrix. Multiplying by (1 + p) is equivalent to adding p*D.
    # For tour edges, we want a significant boost.
    # Let's set the penalty for tour edges to be at least A_tour.
    # If hist_penalty is already high, we might double up. 
    # Better to take the max of the two penalties or add them?
    # Adding is standard in GLS (penalty accumulates).
    
    # Let's refine: 
    # Total penalty = hist_penalty + tour_penalty
    # tour_penalty is A_tour if edge is in tour, else 0.
    tour_penalty = np.where(tour_mask, A_tour, 0.0)
    
    total_penalty = hist_penalty + tour_penalty
    
    # Cap the total penalty to avoid extreme distortions
    # Max penalty of 2.0 means distances can at most double.
    total_penalty = np.clip(total_penalty, 0.0, 2.0)
    
    out = D * (1.0 + total_penalty)

    # Enforce exact symmetry
    out = 0.5 * (out + out.T)

    if not np.all(np.isfinite(out)):
        base = np.nan_to_num(D, nan=0.0, posinf=0.0, neginf=0.0)
        out = 0.5 * (base + base.T)

    return out
# EVOLVE-BLOCK-END
