# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay, cKDTree
import time

_T0 = time.time()
_DEADLINE = _T0 + 285.0  # Safety margin before 300s limit


def heuristics(distance_matrix, coordinates, demands, capacity):
    """Compute edge-attractiveness matrix for ACO on CVRP.
    
    Combines Delaunay triangulation structure with Clarke-Wright savings
    and demand-aware weighting to guide ants toward non-crossing,
    capacity-feasible routes.
    """
    if time.time() > _DEADLINE:
        # Fallback: simple inverse distance
        dm = np.asarray(distance_matrix, dtype=np.float64)
        d_safe = np.where(dm > 0, dm, 1e-9)
        base = 1.0 / (d_safe + 1e-9)
        np.fill_diagonal(base, 1e-9)
        base[base <= 0] = 1e-9
        return base

    n = distance_matrix.shape[0]
    coords = np.asarray(coordinates, dtype=np.float64)
    dem = np.asarray(demands, dtype=np.float64)
    dm = np.asarray(distance_matrix, dtype=np.float64)

    # Base inverse distance heuristic
    d_safe = np.where(dm > 0, dm, 1e-9)
    base = 1.0 / (d_safe + 1e-9)

    # Build Delaunay triangulation on all points (depot + customers)
    try:
        tri = Delaunay(coords)
    except Exception:
        # Fallback if triangulation fails
        np.fill_diagonal(base, 1e-9)
        base[base <= 0] = 1e-9
        return base

    # Extract edges from triangulation
    delaunay_edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            a = int(simplex[i])
            b = int(simplex[(i + 1) % 3])
            if a != b:
                delaunay_edges.add((min(a, b), max(a, b)))

    # Add nearest-neighbor edges (k=12) to ensure better connectivity
    try:
        tree = cKDTree(coords)
        _, nn_indices = tree.query(coords, k=13)  # 13 including self
        for i in range(n):
            for j in nn_indices[i, 1:]:
                j = int(j)
                if j != i:
                    delaunay_edges.add((min(i, j), max(i, j)))
    except Exception:
        pass

    # Compute Clarke-Wright savings ratio for customer-customer edges
    d0 = dm[0, :]
    d0_rows = d0[:, np.newaxis]
    d0_cols = d0[np.newaxis, :]
    num = d0_rows + d0_cols - dm
    den = d0_rows + d0_cols
    den_safe = np.where(den > 0, den, 1e-9)
    savings_ratio = np.clip(num / den_safe, 0.0, 1.0)

    # Build the heuristic matrix
    heuristic = np.full((n, n), 1e-9, dtype=np.float64)

    # For edges in the sparse graph, compute a rich weight
    for (a, b) in delaunay_edges:
        dist_ab = dm[a, b]

        # Base: inverse distance (shorter edges are more attractive)
        w = 1.0 / (dist_ab + 1e-9)

        # Clarke-Wright savings bonus for customer-customer edges
        if a > 0 and b > 0:
            sr = savings_ratio[a, b]
            w *= (1.0 + 1.0 * sr)

        # Demand compatibility bonus
        if a == 0 or b == 0:
            # Depot edges: slight bonus for lower-demand customers
            other = b if a == 0 else a
            if other > 0:
                w *= (1.0 + 0.15 * (capacity - dem[other]) / capacity)
        else:
            # Customer-customer: bonus if both fit in one route segment
            if dem[a] + dem[b] <= capacity:
                w *= 1.3
            else:
                w *= 0.7

        heuristic[a, b] = w
        heuristic[b, a] = w

    # For non-graph edges, apply a smooth distance-decay penalty
    # This softly enforces the planar structure while allowing flexibility
    edge_set = set()
    for (a, b) in delaunay_edges:
        edge_set.add((a, b))
        edge_set.add((b, a))
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if (i, j) not in edge_set:
                # Smooth penalty: closer non-graph edges get higher (but still low) weights
                dist_ij = dm[i, j]
                # Scale so that typical non-graph edges get ~0.01-0.05 of graph edge weight
                penalty = 1.0 / (1.0 + 5.0 * dist_ij)
                heuristic[i, j] = base[i, j] * penalty * 0.1

    # Ensure diagonal is zero (no self-loops)
    np.fill_diagonal(heuristic, 1e-9)

    # Ensure all off-diagonal entries are positive
    mask = heuristic <= 0
    heuristic[mask] = 1e-9

    # Normalize to a reasonable scale using graph-edge median
    graph_vals = []
    for (a, b) in delaunay_edges:
        graph_vals.append(heuristic[a, b])
    if graph_vals:
        med = np.median(graph_vals)
        if med > 0:
            heuristic = heuristic / med

    return heuristic
# EVOLVE-BLOCK-END
