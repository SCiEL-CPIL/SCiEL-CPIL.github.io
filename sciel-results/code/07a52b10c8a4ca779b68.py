"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import linprog


def priority(adjacency):
    t_start = time.time()
    deadline = 110.0  # safety margin before 120s kill

    n = adjacency.shape[0]
    adj = np.asarray(adjacency, dtype=bool)
    
    # Get edges
    edges = np.argwhere(adj)
    edges = edges[edges[:, 0] < edges[:, 1]]  # unique undirected edges
    m = len(edges)

    if m == 0:
        return np.ones(n, dtype=float)

    # LP: max sum(x) s.t. x_u + x_v <= 1 for each edge, 0 <= x <= 1
    # linprog minimizes, so we minimize -sum(x)
    c = -np.ones(n)

    # Constraints: A_ub @ x <= b_ub
    # For each edge (u,v): x_u + x_v <= 1
    A_ub = np.zeros((m, n))
    for i, (u, v) in enumerate(edges):
        A_ub[i, u] = 1.0
        A_ub[i, v] = 1.0
    b_ub = np.ones(m)

    # Bounds: 0 <= x_v <= 1
    bounds = [(0, 1)] * n

    # Try to solve the LP
    try:
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if result.success:
            elapsed = time.time() - t_start
            if elapsed < deadline:
                # x_v values are the priority scores (higher = more likely in IS)
                return result.x.astype(float)
    except Exception:
        pass

    # Fallback: use degree-based heuristic if LP fails or times out
    elapsed = time.time() - t_start
    if elapsed < deadline:
        degrees = adj.sum(axis=1).astype(float)
        return -degrees
    else:
        return -adj.sum(axis=1).astype(float)
# EVOLVE-BLOCK-END
