import numpy as np
import time

# Global start time to enforce the 330s limit
_start_time = None

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard: 300s limit, leave 10s margin
    if time.time() - _start_time > 310:
        return edge_distance.copy()

    n = edge_distance.shape[0]
    
    # Ensure edge_n_used is float for calculation
    usage = edge_n_used.astype(float)
    
    # Calculate penalty
    # Strategy: Additive penalty scaled by usage, with a boost for short edges.
    # The boost helps escape local optima dominated by short, "greedy" choices.
    
    # Base penalty: linear in usage
    # Boost: inverse distance, so short edges get penalized more heavily per unit of usage
    # We add a small epsilon to avoid division by zero
    
    eps = 1e-6
    # dist_inv = 1.0 / (edge_distance + eps)
    # To prevent the penalty from exploding for very short edges, we can cap it or use a softer form
    # Let's use: penalty = alpha * usage * (1.0 + 0.5 / (edge_distance + eps))
    
    alpha = 0.05
    penalty = alpha * usage * (1.0 + 0.5 / (edge_distance + eps))
    
    # The resulting distance matrix
    new_distances = edge_distance + penalty
    
    # Ensure symmetry (just in case, though inputs are symmetric)
    # new_distances = (new_distances + new_distances.T) / 2.0
    
    # Ensure diagonal is 0 (though 2-opt usually ignores it, it's good practice)
    np.fill_diagonal(new_distances, 0.0)
    
    return new_distances
