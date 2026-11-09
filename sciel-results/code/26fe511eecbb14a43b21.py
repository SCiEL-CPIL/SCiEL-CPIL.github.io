# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
import time


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges.
    Uses MILP to minimize the maximum edge load by introducing binary routing
    variables and a continuous peak-load variable.
    """
    start_time = time.time()
    deadline = 600.0  # safety margin before 630s kill

    n_nodes = 10
    demands = [
        (0, 5, 3),
        (1, 7, 2),
        (2, 9, 4),
        (3, 6, 1),
        (0, 4, 5),
        (2, 8, 2),
        (5, 9, 3),
        (1, 4, 2),
        (6, 0, 4),
        (3, 8, 1),
        (7, 2, 3),
        (4, 9, 2),
        (8, 1, 5),
        (0, 6, 1),
        (5, 2, 2),
    ]
    D = len(demands)
    weights = np.array([w for _, _, w in demands], dtype=np.float64)

    # Precompute CW and CCW edge incidence matrices
    cw_matrix = np.zeros((D, n_nodes), dtype=np.float64)
    ccw_matrix = np.zeros((D, n_nodes), dtype=np.float64)
    for i, (s, t, w) in enumerate(demands):
        e = s
        while e != t:
            cw_matrix[i, e] = 1.0
            e = (e + 1) % n_nodes
        all_edges = np.ones(n_nodes)
        ccw_matrix[i] = all_edges - cw_matrix[i]

    # MILP formulation:
    # Variables: x_0..x_{D-1} (binary, 1=CCW), t (continuous, peak load)
    # Minimize: t
    # Subject to: for each edge e:
    #   sum_i weights[i] * (cw_matrix[i,e] * (1 - x_i) + ccw_matrix[i,e] * x_i) <= t
    #   = sum_i weights[i]*cw_matrix[i,e] + sum_i weights[i]*(ccw_matrix[i,e] - cw_matrix[i,e]) * x_i <= t

    n_vars = D + 1  # D binary + 1 continuous (t)
    c = np.zeros(n_vars)
    c[D] = 1.0  # minimize t

    # Constraints: for each edge e, load_e <= t
    # load_e = base_e + sum_i w_i * diff_ei * x_i <= t
    # => base_e + sum_i w_i * diff_ei * x_i - t <= 0
    # In standard form: A @ vars <= b
    diff_matrix = ccw_matrix - cw_matrix  # (D, n_nodes)
    base_loads = weights @ cw_matrix  # (n_nodes,)

    # A matrix: n_nodes rows, n_vars columns
    A = np.zeros((n_nodes, n_vars))
    for e in range(n_nodes):
        A[e, :D] = weights * diff_matrix[:, e]
        A[e, D] = -1.0

    b = -base_loads

    # Bounds: x_i in [0,1], t in [0, inf)
    lb = np.zeros(n_vars)
    ub = np.ones(n_vars)
    ub[D] = np.inf

    # Integrality: first D variables are integer (binary), t is continuous
    integrality = np.zeros(n_vars)
    integrality[:D] = 1.0  # binary
    integrality[D] = 0.0   # continuous

    # Solve MILP
    result = milp(
        c=c,
        constraints=LinearConstraint(A, -np.inf, b),
        integrality=integrality,
        bounds=Bounds(lb, ub),
    )

    if result.status == 0 and result.x is not None:
        best_routing = np.zeros(D, dtype=np.float64)
        for i in range(D):
            best_routing[i] = 1.0 if result.x[i] > 0.5 else 0.0
        return best_routing

    # Fallback: if MILP fails, use exhaustive search
    if time.time() - start_time > deadline - 60:
        # Not enough time for exhaustive; return all CW
        return np.zeros(D, dtype=np.float64)

    # Exhaustive search fallback
    cw_edges_list = []
    ccw_edges_list = []
    for i, (s, t, w) in enumerate(demands):
        cw = set()
        e = s
        while e != t:
            cw.add(e)
            e = (e + 1) % n_nodes
        cw_edges_list.append(cw)
        ccw_edges_list.append(set(range(n_nodes)) - cw)

    best_max_load = float('inf')
    best_mask = 0

    n_masks = 1 << D
    chunk_size = 4096
    for chunk_start in range(0, n_masks, chunk_size):
        if time.time() - start_time > deadline - 30:
            break
        chunk_end = min(chunk_start + chunk_size, n_masks)
        for mask in range(chunk_start, chunk_end):
            loads = np.zeros(n_nodes)
            for i in range(D):
                if (mask >> i) & 1:
                    for e in ccw_edges_list[i]:
                        loads[e] += weights[i]
                else:
                    for e in cw_edges_list[i]:
                        loads[e] += weights[i]
            max_load = np.max(loads)
            if max_load < best_max_load:
                best_max_load = max_load
                best_mask = mask

    best_routing = np.array([(best_mask >> i) & 1 for i in range(D)], dtype=np.float64)
    return best_routing


# EVOLVE-BLOCK-END
