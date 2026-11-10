"""
Structural change: use a rank-based heuristic blended with a demand-aware
savings-based signal, instead of distance-only inverse-distance plus a small
reference-edge boost.

Key ideas:
1. Rank-based base: rank the inverse distances so the heuristic is less
   sensitive to scale and outliers, and more about relative ordering.
2. Demand-aware savings: compute Clarke-Wright savings that incorporate
   demand (capacity feasibility) so that edges which would naturally be
   combined in a feasible route are emphasised.
3. Blend: geometric mean of the rank-based inverse-distance heuristic and a
   normalized savings-based heuristic, then normalize to unit mean off-diagonal.

This is a different modelling family (rank + savings) rather than another
tune of 1/d^alpha plus a boost.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ant colony optimization.
    """
    deadline = time.time() + 290.0

    d = np.asarray(distance_matrix, dtype=np.float64)
    n = d.shape[0]

    if n == 0:
        return np.array([])

    if d.shape != (n, n):
        # Fallback: safe default
        return 1.0 / (np.asarray(distance_matrix, dtype=np.float64) + 1e-9)

    eps = 1e-9

    # Ensure demands is a flat float array of length n
    dem = np.asarray(demands, dtype=np.float64).ravel()
    if dem.shape[0] != n:
        dem = np.zeros(n, dtype=np.float64)

    # ------------------------------------------------------------------
    # 1. Rank-based inverse-distance heuristic
    # ------------------------------------------------------------------
    # Work on off-diagonal entries only.
    offdiag_mask = ~np.eye(n, dtype=bool)

    # Inverse distance with a moderate exponent
    inv_dist = 1.0 / (d + eps)

    # Rank off-diagonal entries in descending order (larger = better).
    # We use scipy.stats.rankdata if available, else a manual approach.
    try:
        from scipy.stats import rankdata
        # Flatten, rank, reshape
        flat = inv_dist[offdiag_mask]
        ranks = rankdata(flat, method='ordinal')  # 1 = smallest
        # Convert to "attractiveness" rank: larger rank value = more attractive
        # rankdata gives 1..m where m is largest; we want largest distance -> smallest rank value
        # So attractiveness = m + 1 - rank
        m = len(flat)
        attr = (m + 1) - ranks
        rank_eta = np.zeros((n, n), dtype=np.float64)
        rank_eta[offdiag_mask] = attr
    except Exception:
        # Fallback: manual rank
        flat = inv_dist[offdiag_mask]
        order = np.argsort(flat)  # ascending
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(flat) + 1)
        m = len(flat)
        attr = (m + 1) - ranks
        rank_eta = np.zeros((n, n), dtype=np.float64)
        rank_eta[offdiag_mask] = attr

    # Normalize rank_eta to unit mean off-diagonal
    mean_rank = np.mean(rank_eta[offdiag_mask])
    if mean_rank > 0 and np.isfinite(mean_rank):
        rank_eta = rank_eta / mean_rank
    else:
        rank_eta = 1.0 / (d + eps)
        mean_rank = np.mean(rank_eta[offdiag_mask])
        if mean_rank > 0 and np.isfinite(mean_rank):
            rank_eta = rank_eta / mean_rank
        else:
            rank_eta[~np.isfinite(rank_eta)] = 1e-9
            rank_eta[rank_eta <= 0] = 1e-9

    # ------------------------------------------------------------------
    # 2. Demand-aware savings-based heuristic
    # ------------------------------------------------------------------
    # Clarke-Wright savings: saving(i, j) = d[i,0] + d[0,j] - d[i,j]
    # But we want a demand-aware version: only consider (i,j) if
    # demands[i] + demands[j] <= capacity (feasible to combine in one route).
    # We'll compute savings and set infeasible pairs to a low value.

    savings = d[:, 0:1] + d[0:1, :] - d  # shape (n, n)
    # For i, j both customers (>=1), check feasibility
    # Create a feasibility mask for customer-customer pairs
    feasible = np.ones((n, n), dtype=bool)
    # For pairs (i, j) where i != j and both are customers
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                if dem[i] + dem[j] > capacity:
                    feasible[i, j] = False
                    feasible[j, i] = False

    # Savings heuristic: larger savings = more attractive
    # But savings can be negative; clip to a small positive floor
    sav = savings.copy()
    sav[~feasible] = -np.inf
    # Also, for depot-customer edges, savings is not directly applicable;
    # we'll handle them separately. For now, set depot rows/cols to a
    # moderate value based on inverse distance.
    sav[0, :] = 0.0
    sav[:, 0] = 0.0

    # Normalize savings: shift so min feasible is 0, then scale
    feasible_vals = sav[feasible & offdiag_mask]
    if len(feasible_vals) > 0 and np.any(np.isfinite(feasible_vals)):
        min_sav = np.min(feasible_vals[np.isfinite(feasible_vals)])
        max_sav = np.max(feasible_vals[np.isfinite(feasible_vals)])
        if max_sav > min_sav:
            sav_norm = (sav - min_sav) / (max_sav - min_sav)
        else:
            sav_norm = np.zeros_like(sav)
    else:
        sav_norm = np.zeros_like(sav)

    # Replace non-finite with 0
    sav_norm[~np.isfinite(sav_norm)] = 0.0
    sav_norm[sav_norm < 0] = 0.0

    # For depot-customer edges, use inverse distance as a proxy
    # since savings don't apply. This keeps depot edges reasonable.
    depot_mask = (np.arange(n) == 0) | (np.arange(n) == 0)
    # Actually, simpler: set row 0 and col 0 to inv_dist normalized
    inv_dist_norm = 1.0 / (d + eps)
    mean_inv = np.mean(inv_dist_norm[offdiag_mask])
    if mean_inv > 0 and np.isfinite(mean_inv):
        inv_dist_norm = inv_dist_norm / mean_inv
    else:
        inv_dist_norm[~np.isfinite(inv_dist_norm)] = 1e-9
        inv_dist_norm[inv_dist_norm <= 0] = 1e-9

    sav_norm[0, :] = inv_dist_norm[0, :]
    sav_norm[:, 0] = inv_dist_norm[:, 0]

    # Normalize sav_norm to unit mean off-diagonal
    mean_sav = np.mean(sav_norm[offdiag_mask])
    if mean_sav > 0 and np.isfinite(mean_sav):
        sav_norm = sav_norm / mean_sav
    else:
        sav_norm[~np.isfinite(sav_norm)] = 1e-9
        sav_norm[sav_norm <= 0] = 1e-9

    # ------------------------------------------------------------------
    # 3. Blend: geometric mean of rank_eta and sav_norm
    # ------------------------------------------------------------------
    # Geometric mean: sqrt(rank_eta * sav_norm)
    # This balances the two signals.
    blend = np.sqrt(rank_eta * sav_norm + eps)

    # Normalize to unit mean off-diagonal
    mean_blend = np.mean(blend[offdiag_mask])
    if mean_blend > 0 and np.isfinite(mean_blend):
        blend = blend / mean_blend
    else:
        blend[~np.isfinite(blend)] = 1e-9
        blend[blend <= 0] = 1e-9

    # ------------------------------------------------------------------
    # 4. Ensure positivity and finiteness
    # ------------------------------------------------------------------
    blend[~np.isfinite(blend)] = 1e-9
    blend[blend <= 0] = 1e-9

    return blend
# EVOLVE-BLOCK-END
