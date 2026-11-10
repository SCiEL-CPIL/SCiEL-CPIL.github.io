"""Heuristic matrix for ant colony optimization of orienteering."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    """
    Edge-attractiveness matrix for ACO orienteering.

    Design (same family: cost-aware ratio with return-to-origin, smoothed):
      1. Feasibility-aware cost: an ant can only go to j if
         len + d[i,j] + d[j,0] <= max_len.  So the "effective cost" of the
         i->j edge is d[i,j] + d[j,0] (the return leg is the binding term).
      2. Base attractiveness: prize_j / (d[i,j] + d[j,0] + eps)^alpha,
         which rewards high-prize nodes that are cheap to reach AND return.
      3. Smooth feasibility shaping: multiply by a soft-min factor that
         gently down-weights edges whose return cost threatens the budget,
         using logaddexp (smooth surrogate for the hard min/max cutoff).
      4. Local density: reward nodes sitting in prize-dense regions so the
         colony clusters on valuable pockets reachable within the short budget.
      5. All entries floored to 1e-9 (non-positive handled by the caller too).
    """
    prizes = np.asarray(prizes, dtype=float)
    D = np.asarray(distance_matrix, dtype=float)
    n = D.shape[0]
    eps = 1e-9

    # --- 1. effective cost of edge i->j (move + return-to-origin) ---
    # c[i,j] = d[i,j] + d[j,0]
    c = D + D[:, 0:1]

    # --- 2. base cost-aware ratio ---
    # prize of the destination node j (row broadcast)
    prize_row = prizes[None, :] + eps          # (1, n)
    cost_term = (c + eps) ** 0.5               # alpha = 0.5 (smooth, not too sharp)
    base = prize_row / cost_term               # (n, n)

    # --- 3. smooth feasibility shaping (soft-min surrogate) ---
    # The binding constraint is len + c[i,j] <= max_len.  Since len >= 0,
    # edges with c[i,j] near max_len are risky.  Use a soft penalty that
    # grows as c[i,j] approaches max_len, via logaddexp for smoothness.
    # scale so that the penalty is ~1 well below budget and grows above.
    slack = max_len - c                        # (n, n), can be negative
    # soft penalty: exp(-lambda * relu-like slack) via logaddexp
    # relu-like: max(0, -slack) smoothed = logaddexp(0, -slack)
    lambda_budget = 3.0
    soft_penalty = np.exp(-lambda_budget * np.logaddexp(0.0, -slack))
    base = base * soft_penalty

    # --- 4. local prize density factor (node-level) ---
    # For each node j, average prize of its k nearest neighbors (cheap proxy
    # for "is this node in a valuable cluster").  This is node-specific, so
    # broadcast across rows.
    k = 5
    # neighbor prize for each j: top-k prizes among other nodes weighted by proximity
    # Use inverse-distance weighted sum of prizes of the k nearest nodes.
    inv_d = 1.0 / (D + eps)                   # (n, n)
    # zero self
    np.fill_diagonal(inv_d, 0.0)
    # for each j, take the k largest inv_d entries (nearest neighbors)
    # inv_d[i,j] is weight from i to j; we want for destination j the
    # neighbors i.  So column j of inv_d.  Take top-k per column.
    col = inv_d.T                             # (n, n) where row j = weights from j's neighbors
    # top-k per row
    kk = min(k, n - 1)
    # indices of top-k per row
    top_idx = np.argpartition(-col, kk, axis=1)[:, :kk]   # (n, kk)
    rows = np.arange(n)[:, None]
    top_w = col[rows, top_idx]                # (n, kk)
    density = top_w.sum(axis=1)               # (n,)
    # normalize to [~0.5, ~1.5] centered around 1
    density_norm = 0.5 + density / (density.mean() + eps)
    base = base * density_norm[None, :]

    # --- 5. ensure positivity and finite; floor to 1e-9 ---
    base = np.where(np.isfinite(base), base, 1e-9)
    base = np.where(base > 0, base, 1e-9)
    base = np.maximum(base, 1e-9)

    return base
# EVOLVE-BLOCK-END
