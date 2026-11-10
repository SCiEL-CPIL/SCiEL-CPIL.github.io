# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_P(support, weights):
    """Compute P[X1+X2+X3 < 2*X4] exactly in float64."""
    s = np.asarray(support, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    n = len(s)
    if n == 0:
        return 0.0
    
    w_sum = w.sum()
    if w_sum <= 1e-9:
        return 0.0
    w = w / w_sum
    
    # Sort by support value
    order = np.argsort(s)
    s_sorted = s[order]
    w_sorted = w[order]
    
    # Precompute all pairwise sums and their probabilities
    pair_sums = s_sorted[:, None] + s_sorted[None, :]
    pair_probs = w_sorted[:, None] * w_sorted[None, :]
    
    # Flatten and sort pair sums for efficient lookup
    flat_sums = pair_sums.flatten()
    flat_probs = pair_probs.flatten()
    
    # Sort by sum value
    sort_idx = np.argsort(flat_sums, kind='stable')
    sorted_sums = flat_sums[sort_idx]
    sorted_probs = flat_probs[sort_idx]
    
    # Cumulative sum of probabilities for fast prefix queries
    cum_probs = np.cumsum(sorted_probs)
    
    # Vectorized computation over (k, l) pairs
    thresholds = 2.0 * s_sorted[None, :] - s_sorted[:, None]
    
    # For each threshold, find count of pair sums strictly less
    counts = np.searchsorted(sorted_sums, thresholds.ravel(), side='left')
    counts = counts.reshape(n, n)
    
    safe_counts = np.maximum(counts - 1, 0)
    inner_sums = np.where(counts > 0, cum_probs[safe_counts], 0.0)
    
    P = np.sum(w_sorted[:, None] * w_sorted[None, :] * inner_sums)
    
    return float(P)

def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    
    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights (need not be normalized).
    """
    deadline = time.time() + 580  # Safety margin
    
    # Try several candidate measures and pick the best
    candidates = []
    
    # Candidate 1: Uniform on {0, 1}
    candidates.append((np.array([0.0, 1.0]), np.array([0.5, 0.5])))
    
    # Candidate 2: Heavier at 0
    candidates.append((np.array([0.0, 0.5, 1.0]), np.array([0.6, 0.25, 0.15])))
    
    # Candidate 3: Three-point with more mass at low values
    candidates.append((np.array([0.0, 0.3, 1.0]), np.array([0.7, 0.2, 0.1])))
    
    # Candidate 4: Four-point distribution
    candidates.append((np.array([0.0, 0.2, 0.5, 1.0]), np.array([0.5, 0.25, 0.15, 0.1])))
    
    # Candidate 5: Five-point distribution
    candidates.append((np.array([0.0, 0.15, 0.35, 0.6, 1.0]), np.array([0.45, 0.25, 0.15, 0.1, 0.05])))
    
    # Candidate 6: Geometric spacing
    candidates.append((np.array([0.0, 0.1, 0.2, 0.4, 0.8]), np.array([0.4, 0.25, 0.15, 0.12, 0.08])))
    
    # Candidate 7: More mass concentrated
    candidates.append((np.array([0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0]), np.array([0.35, 0.2, 0.15, 0.12, 0.08, 0.06, 0.04])))
    
    # Evaluate all candidates
    best_support = None
    best_weights = None
    best_P = -1.0
    
    for support, weights in candidates:
        if time.time() > deadline:
            break
        try:
            P = _compute_P(support, weights)
            if P > best_P:
                best_P = P
                best_support = support.copy()
                best_weights = weights.copy()
        except Exception:
            pass
    
    # Ensure validity
    if best_support is None:
        best_support = np.array([0.0, 1.0], dtype=np.float64)
        best_weights = np.array([0.5, 0.5], dtype=np.float64)
    
    # Clean up: ensure no NaN/inf, non-negative
    best_support = np.nan_to_num(best_support, nan=0.0, posinf=1e6, neginf=0.0)
    best_weights = np.nan_to_num(best_weights, nan=0.0, posinf=1.0, neginf=0.0)
    best_support = np.maximum(best_support, 0.0)
    best_weights = np.maximum(best_weights, 0.0)
    
    # Ensure sum of weights is positive
    if best_weights.sum() <= 1e-9:
        best_weights = np.ones_like(best_weights)
    
    # Remove zero-weight points to keep solution sparse
    mask = best_weights > 1e-10
    if mask.sum() > 0 and mask.sum() < len(best_weights):
        best_support = best_support[mask]
        best_weights = best_weights[mask]
    
    return best_support, best_weights

# EVOLVE-BLOCK-END
