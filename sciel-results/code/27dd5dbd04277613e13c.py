"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Guided local search perturbation for TSP.
    
    Strategy:
    1. Geometry-aware repulsive penalty: edges that are short (high-value) receive
       larger penalties because they are more likely to be part of good tours.
       This forces the search to explore alternative short connections.
    2. Cumulative usage penalty with saturation to avoid numerical issues.
    3. Immediate penalty on edges in the current local-opt tour to guarantee perturbation.
    4. The penalty is applied multiplicatively to keep distances positive.
    
    All operations are vectorized and fast to stay within time limits.
    """
    n = edge_distance.shape[0]
    
    # Deadline guard: track start time
    t_start = time.time()
    
    # --- Step 1: Build tour mask for current local-opt tour (vectorized) ---
    tour_arr = np.asarray(local_opt_tour, dtype=np.int64)
    a = tour_arr[:-1]
    b = tour_arr[1:]
    # Also need the last edge connecting last city to first city
    a_last = tour_arr[-1]
    b_last = tour_arr[0]
    
    tour_mask = np.zeros((n, n), dtype=np.float64)
    tour_mask[a, b] = 1.0
    tour_mask[b, a] = 1.0
    tour_mask[a_last, b_last] = 1.0
    tour_mask[b_last, a_last] = 1.0
    
    # --- Step 2: Geometry-aware repulsive penalty ---
    # Short edges are penalized more heavily because they are "valuable" and
    # likely to be reused. We use an inverse-distance weighting.
    # Avoid division by zero: add small epsilon.
    eps = 1e-12
    inv_dist = 1.0 / (edge_distance + eps)
    
    # Normalize inv_dist to [0, 1] range for stable scaling
    inv_dist_max = np.max(inv_dist)
    if inv_dist_max > 0:
        inv_dist_norm = inv_dist / inv_dist_max
    else:
        inv_dist_norm = np.zeros_like(inv_dist)
    
    # The repulsive force is stronger for short edges
    # Use a power to make it more nonlinear
    repulsive = 0.40 * np.power(inv_dist_norm, 2.0)
    
    # --- Step 3: Cumulative usage penalty with saturation ---
    # alpha controls the strength of the usage-based penalty
    alpha = 0.35
    # Saturating function: usage / (1 + usage) maps [0, inf) -> [0, 1)
    usage_sat = edge_n_used / (1.0 + edge_n_used)
    usage_penalty = alpha * usage_sat
    
    # --- Step 4: Immediate penalty on current tour edges ---
    # Ensure the perturbed matrix differs from the true matrix on the current tour
    immediate_scale = 0.15
    immediate_penalty = immediate_scale * tour_mask
    
    # --- Step 5: Combine penalties ---
    total_penalty = repulsive + usage_penalty + immediate_penalty
    
    # --- Step 6: Apply multiplicatively ---
    result = edge_distance * (1.0 + total_penalty)
    
    # --- Step 7: Ensure finite values ---
    result = np.nan_to_num(result, nan=edge_distance, posinf=edge_distance, neginf=0.0)
    
    # --- Step 8: Deadline guard ---
    t_elapsed = time.time() - t_start
    # This function is called 20 times per instance, 3 instances total = 60 calls.
    # Each call should be fast. The main cost is the loop building tour_mask.
    # For n=100, this is negligible. No early exit needed here.
    
    return result
# EVOLVE-BLOCK-END
