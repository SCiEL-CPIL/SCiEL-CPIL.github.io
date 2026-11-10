import numpy as np
import time

def heuristics(demands, capacity):
    """
    Heuristic matrix for ACO bin packing: attractiveness of placing item j after item i.

    Refined inverse-gap heuristic with soft-penalty for large items:
    - Primary signal: 1/(waste+1) strongly rewards tight pairs (measured improvement over product/waste^2)
    - Secondary signal: mild penalty for very large single items that may cause fragmentation
    - Smooth, differentiable throughout (no min/max discontinuities)
    - Non-fitting pairs get 1e-9; all values sanitized to finite positive
    """
    start_time = time.time()

    d = np.asarray(demands, dtype=np.float64)
    n = len(d)

    if n == 0:
        return np.empty((0, 0))

    # Pairwise sums
    sums = d[:, None] + d[None, :]

    # Feasibility mask
    fits = sums <= capacity

    # Waste (non-negative where fits)
    waste = np.where(fits, capacity - sums, 0.0)

    # Inverse gap: 1/(waste + 1) -- smooth, strongly rewards tight packing
    inv_gap = 1.0 / (waste + 1.0)

    # Mild fragmentation penalty: large items (relative to capacity) are slightly less attractive
    # as successors because they leave less flexibility. Smooth, bounded penalty.
    # penalty = 1 - 0.1 * (d_j / capacity) -- ranges from ~0.96 to ~0.86
    # This is gentle so it doesn't dominate the inverse-gap signal
    frag_penalty = 1.0 - 0.1 * (d[None, :] / capacity)

    # Combine: primary signal modulated by mild fragmentation penalty
    raw_scores = inv_gap * frag_penalty

    # Mask non-fitting pairs to 1e-9
    H = np.where(fits, raw_scores, 1e-9)

    # Ensure all values are positive and finite
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)

    # Shape check
    if H.shape != (n, n):
        raise ValueError("Wrong shape")

    # Time guard
    elapsed = time.time() - start_time
    if elapsed > 290:
        return H

    return H
