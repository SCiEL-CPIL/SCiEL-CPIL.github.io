"""Guided local search penalty rule for the TSP – multiplicative context-aware penalty."""
import numpy as np
import time

_START_TIME = time.time()
_TIME_LIMIT = 315.0  # Safety margin before 330s kill


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Multiplicative context-aware penalty for guided local search on TSP.

    Key insight from observations: multiplicative penalties that target
    only current tour edges preserve global geometry for 2-opt moves while
    specifically breaking long, frequently-used 'stuck' edges.

    Strategy:
    - For each edge in the current local-opt tour, apply a multiplicative
      penalty factor that grows with both the edge's usage count and its
      relative length compared to the tour average.
    - Edges that are both long and frequently used get the strongest penalty,
      as they are the most likely to be 'stuck' edges preventing escape from
      local optima.
    - A small additive component ensures even short, frequently-used edges
      are penalized to some degree.
    - No global penalties are applied, preserving the distance structure
      for edges outside the current tour.
    """
    # Time guard
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        return edge_distance.copy()

    n = edge_distance.shape[0]
    tour = local_opt_tour

    if n < 2 or len(tour) < 2:
        return edge_distance.copy()

    # Work in float64
    D = np.asarray(edge_distance, dtype=np.float64).copy()
    U = np.asarray(edge_n_used, dtype=np.float64)

    # --- Extract tour edges and compute statistics ---
    tour_edges = []
    tour_lens = []
    for i in range(len(tour)):
        u = tour[i]
        v = tour[(i + 1) % len(tour)]
        tour_edges.append((u, v))
        tour_lens.append(D[u, v])

    avg_len = float(np.mean(tour_lens)) if tour_lens else 1.0
    if avg_len < 1e-9:
        avg_len = 1.0

    # Infer round number from max usage count
    max_used = int(U.max()) if U.size > 0 else 0
    round_num = max_used + 1  # 1-based round estimate

    # --- Apply multiplicative + small additive penalty to tour edges ---
    # Base multiplier that grows gently with round number
    base_mult = 1.0 + 0.08 * np.log1p(round_num)

    for u, v in tour_edges:
        d_uv = D[u, v]
        usage = float(U[u, v])

        # Relative length factor: longer edges get more penalty
        # Clamp to avoid extreme values
        rel_len = d_uv / avg_len if avg_len > 0 else 1.0
        rel_len = np.clip(rel_len, 0.5, 3.0)

        # Usage factor: grows sub-linearly to avoid explosion
        usage_factor = 1.0 + 0.3 * np.log1p(usage)

        # Multiplicative penalty: stronger for long, frequently-used edges
        mult_factor = base_mult * (0.5 + 0.5 * rel_len) * usage_factor

        # Small additive component for consistency
        add_component = 0.02 * avg_len * np.log1p(usage)

        new_d = d_uv * mult_factor + add_component

        D[u, v] = new_d
        D[v, u] = new_d

    # Ensure symmetry
    D = (D + D.T) * 0.5
    # Ensure finite
    D = np.nan_to_num(D, nan=0.0,
                      posinf=np.finfo(np.float64).max / 10,
                      neginf=0.0)
    # Zero diagonal
    np.fill_diagonal(D, 0.0)

    return D
