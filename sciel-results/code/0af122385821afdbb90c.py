import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def heuristics(demands, capacity):
    """
    Heuristic matrix for ACO bin packing: attractiveness of placing item j after item i.
    
    Uses a maximum weight matching approach:
    1. Compute base tightness scores for all feasible pairs
    2. Find optimal disjoint pairs via maximum weight matching
    3. Boost matched pairs in the heuristic matrix to guide ACO toward proven good pairings
    """
    start_time = time.time()

    d = np.asarray(demands, dtype=np.float64)
    n = len(d)

    if n == 0:
        return np.empty((0, 0))

    # Pairwise sums and products
    sums = d[:, None] + d[None, :]
    products = d[:, None] * d[None, :]

    # Feasibility mask
    fits = sums <= capacity

    # Waste = capacity - sum (non-negative where fits)
    waste = np.where(fits, capacity - sums, 1.0)

    # Base score: product / (waste + 1)^2
    # Strongly rewards tight fits with large items
    denom = (waste + 1.0) ** 2
    base_scores = products / denom

    # Mask non-fitting pairs
    base_scores = np.where(fits, base_scores, 0.0)

    # Sanitize base scores
    if not np.all(np.isfinite(base_scores)):
        base_scores = np.where(np.isfinite(base_scores), base_scores, 0.0)

    # Find maximum weight matching of disjoint pairs
    # We want to maximize total weight, so use linear_sum_assignment on negative weights
    # Only consider feasible pairs (i < j to avoid double counting and self-pairs)
    matching_weights = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if fits[i, j]:
                matching_weights[i, j] = base_scores[i, j]
                matching_weights[j, i] = base_scores[i, j]

    # Find optimal assignment (minimizes negative = maximizes positive)
    # linear_sum_assignment finds minimum cost, so we negate
    row_ind, col_ind = linear_sum_assignment(-matching_weights)

    # Build set of matched pairs
    matched_pairs = set()
    for r, c in zip(row_ind, col_ind):
        if r != c and matching_weights[r, c] > 0:
            matched_pairs.add((r, c))
            matched_pairs.add((c, r))

    # Construct final heuristic: boost matched pairs significantly
    # Matched pairs get 10x boost to strongly guide ACO
    H = base_scores.copy()
    for (i, j) in matched_pairs:
        H[i, j] = base_scores[i, j] * 10.0

    # Ensure non-fitting pairs are very low
    H = np.where(fits, H, 1e-9)

    # Sanitize: replace any non-finite values
    if not np.all(np.isfinite(H)):
        H = np.where(np.isfinite(H), H, 1e-9)

    # Ensure all values are positive (as per task requirements)
    H = np.maximum(H, 1e-9)

    # Shape check
    if H.shape != (n, n):
        raise ValueError("Wrong shape")

    # Time guard
    elapsed = time.time() - start_time
    if elapsed > 290:
        return H

    return H
