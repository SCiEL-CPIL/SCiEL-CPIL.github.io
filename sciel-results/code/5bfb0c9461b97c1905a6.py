"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    """
    Compute an edge-attractiveness heuristic matrix for ACO orienteering.
    
    The heuristic encourages transitions (i -> j) that:
    1. Lead to nodes with high prizes.
    2. Keep the total path length (current length + d[i,j] + d[j,0]) well within max_len.
    3. Are geometrically efficient (short d[i,j]).
    
    Since the heuristic is computed once before the colony runs and is static,
    we design it to favor nodes that are "safe" to visit (close to start, so they
    leave room for more visits) and high value.
    
    Key insight: In orienteering with a length constraint, visiting nodes far from
    the start (node 0) is risky because it consumes budget that could be used for
    other nodes. A good heuristic should weight prizes by the "affordability" of
    visiting that node.
    
    We define the heuristic for edge (i, j) as:
    h[i, j] = prize[j] * exp(-lambda * d[j, 0]) / (d[i, j] + epsilon)
    
    Where:
    - prize[j] is the prize at node j.
    - d[j, 0] is the distance from node j to the start. Nodes closer to start are
      preferred because they are less likely to violate the length constraint when
      combined with other nodes.
    - d[i, j] is the distance from i to j. Shorter steps are preferred.
    - lambda controls the strength of the preference for nodes close to the start.
    
    We also add a small constant to ensure all entries are positive (as required:
    non-positive become 1e-9, but we want strictly positive for better gradient).
    """
    prizes = np.asarray(prizes, dtype=np.float64)
    distance_matrix = np.asarray(distance_matrix, dtype=np.float64)
    
    n = len(prizes)
    if n == 0:
        return np.array([])
    
    eps = 1e-9
    
    # d_j0: distance from each node j to node 0 (shape: (n,))
    d_j0 = distance_matrix[:, 0]
    
    # We want to favor nodes that are close to the start (small d_j0) and have high prizes.
    # Let's use an exponential decay based on d_j0 relative to max_len.
    # max_len is the total budget. A node at distance d_j0 from start uses at least
    # d_j0 + d_j0 = 2*d_j0 if you go there and come back directly. But since we can
    # chain, the constraint is more complex. However, nodes far from start are generally
    # harder to fit into a tour.
    
    # Let's try a heuristic that combines:
    # 1. Prize value
    # 2. Proximity to start (encourages "hub" nodes or nodes in a compact region)
    # 3. Step length (local proximity)
    
    # Normalization: d_j0 can be up to sqrt(2) in unit square. max_len is 3.
    # Let's scale the penalty by max_len to be robust.
    scale = max_len + 1e-9
    
    # Affinity for node j: prize[j] * exp(-d_j0[j] / scale)
    # This downweights nodes far from start.
    node_affinity = prizes * np.exp(-d_j0 / scale)
    
    # Edge heuristic: node_affinity[j] / (d[i, j] + eps)
    # This encourages moving to attractive nodes that are close by.
    
    # Reshape node_affinity for broadcasting: (1, n)
    node_affinity_row = node_affinity[None, :]
    
    # Distance matrix is (n, n)
    # Add epsilon to avoid division by zero
    d_plus_eps = distance_matrix + eps
    
    # h[i, j] = node_affinity[j] / d[i, j]
    h = node_affinity_row / d_plus_eps
    
    # Ensure all values are positive. The problem states: "Heuristic entries that are
    # not positive become 1e-9". We can just clip to 1e-9.
    h = np.maximum(h, 1e-9)
    
    # Optional: Normalize to prevent overflow/underflow in pheromone * heuristic product.
    # The ACO algorithm uses pheromone * heuristic. Pheromone starts at 1.
    # If heuristic values are very large or small, it might affect convergence.
    # Let's normalize so that the mean is 1 or max is 1?
    # The task doesn't specify normalization, but it's often good practice.
    # However, the "seed" program didn't normalize. Let's check the scale.
    # prizes in [0,1], exp term in [exp(-sqrt(2)/3), 1] ~ [0.5, 1].
    # d[i,j] in [0, sqrt(2)].
    # So h is roughly in [0, 2]. This is a reasonable scale.
    
    return h
# EVOLVE-BLOCK-END
