import numpy as np
import time

# Global deadline guard
_DEADLINE = time.time() + 110.0  # 110s safety margin within 120s limit

def _softmin(a, axis, beta=50.0):
    """
    Smooth approximation to np.min(a, axis) using log-sum-exp.
    softmin(a) = -1/beta * log(sum(exp(-beta * a)))
    As beta -> inf, this approaches the true min.
    """
    a = np.asarray(a, dtype=np.float64)
    # Shift for numerical stability
    a_shifted = -beta * a
    a_shifted -= np.max(a_shifted, axis=axis, keepdims=True)
    s = np.sum(np.exp(a_shifted), axis=axis)
    return -(1.0 / beta) * (np.log(s) + np.max(a_shifted, axis=axis))

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Pick the next city in a TSP tour using a multi-criteria heuristic with
    smooth (soft-min) surrogates and a minimax/bottleneck component.
    
    Strategy: For each candidate node j in unvisited_nodes, compute a composite score
    that balances:
      1. Distance from current_node to j (greedy)
      2. Distance from j to destination_node (goal-seeking)
      3. Soft-min distance from j to other unvisited nodes (smooth connectivity)
      4. Soft-max distance from j to other unvisited nodes (bottleneck/minimax penalty)
    
    The soft-min and soft-max use log-sum-exp to smooth the nonsmooth min/max,
    avoiding the instability of hard argmin choices while still approximating
    them closely with a large beta.
    """
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return current_node
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Check deadline
    if time.time() > _DEADLINE:
        # Fallback: greedy nearest neighbor
        d_curr = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(d_curr)])
    
    # Vectorized computations
    # d_curr: (N,) distances from current to all unvisited
    d_curr = distance_matrix[current_node, unvisited_nodes].astype(np.float64)
    
    # d_dest: (N,) distances from each unvisited to destination
    d_dest = distance_matrix[unvisited_nodes, destination_node].astype(np.float64)
    
    # d_unvisited: (N, N) distances between unvisited nodes
    d_unvisited = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)].astype(np.float64)
    
    # Set diagonal to inf so we exclude self
    np.fill_diagonal(d_unvisited, np.inf)
    
    # Soft-min: smooth approximation to min distance from j to nearest other unvisited
    # Use moderate beta for smoothness while still approximating min closely
    beta_min = 50.0
    soft_min_next = _softmin(d_unvisited, axis=1, beta=beta_min)
    
    # Soft-max: smooth approximation to max distance from j to farthest other unvisited
    # This captures the bottleneck/minimax aspect: penalize candidates that are
    # far from some remaining city, preventing long edges in the final tour
    beta_max = 30.0
    # softmax(a) = 1/beta * log(sum(exp(beta * a)))
    a_shifted = beta_max * d_unvisited
    a_shifted -= np.max(a_shifted, axis=1, keepdims=True)
    s = np.sum(np.exp(a_shifted), axis=1)
    soft_max_next = (1.0 / beta_max) * (np.log(s) + np.max(beta_max * d_unvisited, axis=1))
    
    # Composite score with tuned weights
    # w_greedy: primary greedy component
    # w_dest: goal-seeking toward destination
    # w_conn: connectivity (want low min-distance to neighbors)
    # w_bottleneck: penalize high max-distance (minimax)
    w_greedy = 1.0
    w_dest = 0.25
    w_conn = 0.12
    w_bottleneck = 0.08
    
    scores = (w_greedy * d_curr + 
              w_dest * d_dest + 
              w_conn * soft_min_next + 
              w_bottleneck * soft_max_next)
    
    best_idx = np.argmin(scores)
    
    return int(unvisited_nodes[best_idx])
