"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _greedy_is_size(adjacency, scores):
    """Simulate the greedy decoder: pick vertices in descending score order, skip neighbors."""
    n = len(scores)
    order = np.argsort(-scores, kind='stable')  # descending score
    selected = np.zeros(n, dtype=bool)
    blocked = np.zeros(n, dtype=bool)
    count = 0
    for v in order:
        if not selected[v] and not blocked[v]:
            selected[v] = True
            count += 1
            # Block neighbors
            nbrs = np.where(adjacency[v])[0]
            blocked[nbrs] = True
    return count


def priority(adjacency):
    """Return a score per vertex for the greedy independent set decoder."""
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s limit

    adjacency = np.asarray(adjacency, dtype=bool)
    n = adjacency.shape[0]
    degrees = adjacency.sum(axis=1).astype(float)
    mean_deg = degrees.mean() + 1e-9

    # Compute neighbor statistics
    sum_nbr_deg = (adjacency @ degrees).astype(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        mean_nbr_deg = np.where(degrees > 0, sum_nbr_deg / degrees, 0.0)

    # Minimum and max degree among neighbors
    min_nbr_deg = np.full(n, 0.0)
    max_nbr_deg = np.zeros(n, dtype=float)
    for v in range(n):
        nbrs = np.where(adjacency[v])[0]
        if len(nbrs) > 0:
            min_nbr_deg[v] = degrees[nbrs].min()
            max_nbr_deg[v] = degrees[nbrs].max()

    # Number of neighbors with degree <= current vertex's degree
    # (rough proxy for "local density" around v)
    # For each vertex, count neighbors with degree <= degrees[v]
    # This is expensive if done naively; use a vectorized approach
    # Build a matrix: nbr_deg_leq[v, u] = 1 if u is neighbor of v and deg[u] <= deg[v]
    # deg_matrix: deg[u] for all u
    deg_matrix = degrees[None, :]  # shape (1, n)
    deg_v_col = degrees[:, None]   # shape (n, 1)
    nbr_deg_leq = (adjacency & (deg_matrix <= deg_v_col)).sum(axis=1).astype(float)
    # Normalize by degree
    with np.errstate(divide='ignore', invalid='ignore'):
        frac_nbrs_lower = np.where(degrees > 0, nbr_deg_leq / degrees, 1.0)

    # Candidate scoring formulas
    candidates = []

    # 1. Negative degree
    candidates.append(-degrees)

    # 2. Negative degree + alpha * mean_nbr_deg
    for alpha in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        candidates.append(-degrees + alpha * mean_nbr_deg)

    # 3. Negative degree + alpha * min_nbr_deg
    for alpha in [0.1, 0.2, 0.3, 0.5]:
        candidates.append(-degrees + alpha * min_nbr_deg)

    # 4. Negative degree * (1 + beta * mean_nbr_deg / mean_deg)
    for beta in [0.5, 1.0, 1.5, 2.0]:
        candidates.append(-degrees * (1.0 + beta * mean_nbr_deg / mean_deg))

    # 5. Centered: -degree + alpha * (mean_nbr_deg - mean_deg)
    for alpha in [0.3, 0.5, 0.7, 1.0]:
        candidates.append(-degrees + alpha * (mean_nbr_deg - mean_deg))

    # 6. Quadratic penalty
    candidates.append(-degrees * degrees)

    # 7. Sum of neighbor degrees (unnormalized)
    for alpha in [0.01, 0.05, 0.1, 0.2]:
        candidates.append(-degrees + alpha * sum_nbr_deg)

    # 8. Ratio: -degree / (1 + mean_nbr_deg)
    candidates.append(-degrees / (1.0 + mean_nbr_deg))

    # 9. -degree + alpha * (min_nbr_deg - mean_nbr_deg)
    for alpha in [0.2, 0.5, 1.0]:
        candidates.append(-degrees + alpha * (min_nbr_deg - mean_nbr_deg))

    # 10. Fraction of lower-degree neighbors
    for alpha in [0.5, 1.0, 2.0]:
        candidates.append(-degrees + alpha * frac_nbrs_lower)

    # 11. -degree * (1 + beta * frac_nbrs_lower)
    for beta in [0.5, 1.0, 2.0]:
        candidates.append(-degrees * (1.0 + beta * frac_nbrs_lower))

    # 12. Log-based: -log(1 + degree) + alpha * mean_nbr_deg
    for alpha in [0.1, 0.3, 0.5]:
        candidates.append(-np.log1p(degrees) + alpha * mean_nbr_deg)

    # 13. -degree + alpha * (max_nbr_deg - min_nbr_deg)  (neighbor degree spread)
    for alpha in [0.1, 0.3, 0.5]:
        candidates.append(-degrees + alpha * (max_nbr_deg - min_nbr_deg))

    # 14. Inverse: 1/(1+degree) based
    candidates.append(1.0 / (1.0 + degrees) - alpha_dummy * 0)  # just 1/(1+deg)
    # Actually let's do it properly
    candidates.pop()
    candidates.append(1.0 / (1.0 + degrees))

    # 15. -degree + alpha * (sum_nbr_deg / (degree+1))
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_nbr = np.where(degrees > 0, sum_nbr_deg / (degrees + 1), 0.0)
    for alpha in [0.3, 0.5, 1.0]:
        candidates.append(-degrees + alpha * avg_nbr)

    # Evaluate all candidates and pick the best
    best_size = -1
    best_candidate = candidates[0]

    for i, cand in enumerate(candidates):
        if time.time() > deadline:
            break
        size = _greedy_is_size(adjacency, cand)
        if size > best_size:
            best_size = size
            best_candidate = cand

    return best_candidate
# EVOLVE-BLOCK-END
