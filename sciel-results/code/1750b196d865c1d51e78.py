# EVOLVE-BLOCK-START
import numpy as np


def ring_routing() -> np.ndarray:
    """
    Solve the SONET ring loading problem on a ring of n = 10 nodes with 10 edges,
    where a FIXED set of D = 15 demands must each be routed either clockwise or
    counter-clockwise around the ring. The goal is to choose a routing direction
    for every demand so as to MINIMIZE the maximum load over all 10 edges.

    Each entry of the returned array selects the direction for one demand:
        0 -> route the demand clockwise
        1 -> route the demand counter-clockwise

    Returns:
        routing: np.ndarray of shape (15,) with one 0/1 entry per demand.
    """
    n_nodes = 10
    n_demands = 15
    
    # Demands: (source, target, weight)
    demands = [
        (0, 5, 3), (1, 7, 2), (2, 9, 4), (3, 6, 1), (0, 4, 5),
        (2, 8, 2), (5, 9, 3), (1, 4, 2), (6, 0, 4), (3, 8, 1),
        (7, 2, 3), (4, 9, 2), (8, 1, 5), (0, 6, 1), (5, 2, 2)
    ]
    
    # Precompute the clockwise edge incidence matrix A_cw
    # A_cw[d, e] = 1 if edge e is on the clockwise path for demand d
    A_cw = np.zeros((n_demands, n_nodes), dtype=float)
    weights = np.zeros(n_demands)
    
    for i, (s, t, w) in enumerate(demands):
        weights[i] = w
        # Clockwise edges: s, s+1, ..., t-1 (mod n)
        edge = s
        while edge != t:
            A_cw[i, edge] = 1.0
            edge = (edge + 1) % n_nodes
            
    # Generate all 2^15 routing combinations
    # Shape: (32768, 15)
    # Using base-2 representation of integers 0 to 2^15 - 1
    # Note: We need to be careful with bit ordering to match demand index i.
    # Let's just generate all combinations using np.indices or bit operations.
    
    # Efficient way to generate binary matrix
    # k is index 0..2^15-1
    # bit i of k corresponds to demand i?
    # Let's define X such that X[k, i] is the bit i of k.
    # k = sum_{i} X[k, i] * 2^i
    
    total_combinations = 1 << n_demands
    # Create array of indices
    indices = np.arange(total_combinations)
    
    # Unpack bits into a matrix
    # bits = (indices[:, None] >> np.arange(n_demands)) & 1
    # This gives shape (32768, 15) where column i is the i-th bit of the index.
    X = (indices[:, None] >> np.arange(n_demands)) & 1
    X = X.astype(float)
    
    # Calculate loads for all combinations
    # L[:, e] = C[e] + sum_d X[:, d] * D[e, d]
    # C[e] = sum_d w_d * A_cw[d, e]
    # D[e, d] = w_d * (1 - 2 * A_cw[d, e])
    
    # C: shape (10,)
    C = A_cw.T @ weights
    
    # D: shape (10, 15)
    # B[d, e] = 1 - 2 * A_cw[d, e]
    B = 1.0 - 2.0 * A_cw
    D = B.T * weights[None, :] # weights is (15,), so D[e, d] = B[d, e] * w_d
    
    # L: shape (32768, 10)
    # L = C[None, :] + X @ D
    L = C[None, :] + X @ D
    
    # Find the combination with minimum max load
    max_loads = np.max(L, axis=1)
    best_idx = np.argmin(max_loads)
    best_routing = X[best_idx]
    
    return best_routing


# EVOLVE-BLOCK-END
