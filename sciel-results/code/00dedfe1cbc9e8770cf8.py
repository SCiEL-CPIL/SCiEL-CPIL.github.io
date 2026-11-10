# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(prizes, distance_matrix, max_len):
    """
    Heuristic for ant colony orienteering.
    
    Strategy:
    - Compute a node-specific "Cluster Value" V[j] which represents the total prize
      mass reachable from node j within the max_len constraint (considering return to origin).
    - This replaces the generic prize p[j] with a constraint-aware marginal utility.
    - Blend multiple prize/distance exponent combinations using V[j].
    - Cost[i,j] = d[i,j] + d[j,0] (step + return to origin).
    - Variants: V/cost^2.5, V^2/cost^3, V/cost^3.5, V^2/cost^4.
    - Normalize each variant and blend with tuned weights.
    - Add a small origin-proximity bonus to stabilize early moves.
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 330s kill

    prizes = np.asarray(prizes, dtype=float)
    distance_matrix = np.asarray(distance_matrix, dtype=float)
    
    n = len(prizes)
    
    # Safety check for finite values
    if not np.all(np.isfinite(prizes)) or not np.all(np.isfinite(distance_matrix)):
        return np.ones((n, n), dtype=float) * 1e-9
    
    # cost[i, j] = d[i, j] + d[j, 0]
    cost = distance_matrix + distance_matrix[:, 0][None, :]
    
    # Ensure cost is positive to avoid division by zero
    eps = 1e-9
    cost_safe = np.maximum(cost, eps)
    
    # --- Compute Node-Specific Cluster Value V[j] ---
    # V[j] represents the potential prize collectible if we are at j and must return to 0.
    # We include p[j] and prizes of all neighbors k that are feasible to visit from j 
    # and return to 0 within max_len.
    # Feasibility condition for static heuristic: d[j,k] + d[k,0] <= max_len.
    
    V = prizes.copy()
    dist_to_origin = distance_matrix[:, 0] # d[k,0]
    
    # Vectorized calculation of feasibility mask for all pairs (j,k)
    # d[j,k] is distance_matrix
    # d[k,0] is dist_to_origin[k]
    # We want mask[j,k] = (distance_matrix[j,k] + dist_to_origin[k] <= max_len)
    # Note: distance_matrix is (N,N), dist_to_origin is (N,).
    # dist_to_origin[k] needs to be broadcast to columns.
    # dist_to_origin[None, :] shape (1, N)
    
    # Check feasibility for all j, k
    # We exclude k=0 (origin) from being a "neighbor" in this density calc as we must return there, 
    # but visiting origin mid-tour is usually not allowed/beneficial in this setup (prize 0).
    # We also exclude j=0 as it's the start.
    
    feasible_mask = (distance_matrix + dist_to_origin[None, :]) <= max_len
    
    # We want to sum prizes[k] for all feasible k for each j.
    # prizes_masked = prizes * feasible_mask (but careful with j=0 and k=0)
    
    # For each j, V[j] = prizes[j] + sum_{k != j, k != 0} prizes[k] * feasible_mask[j,k]
    
    # Create a masked prize matrix where non-feasible entries are 0
    # Also zero out the diagonal (j==k) to avoid double counting self? 
    # The formula V[j] = p[j] + sum_{k != j} ... implies we add p[j] separately.
    # So in the sum, we should exclude k=j.
    
    prizes_col = prizes[None, :] # Shape (1, N)
    
    # Mask: 1 if feasible and k != j and k != 0
    # feasible_mask is (N, N)
    # diag_mask is (N, N) with 0 on diagonal
    # origin_mask is (N, N) with 0 in column 0
    
    diag_mask = 1.0 - np.eye(n)
    origin_mask = np.ones((n, n))
    origin_mask[:, 0] = 0.0
    
    sum_mask = feasible_mask * diag_mask * origin_mask
    
    # Sum of prizes of feasible neighbors
    neighbor_sum = np.sum(prizes_col * sum_mask, axis=1) # Shape (N,)
    
    # Total cluster value
    V = prizes + neighbor_sum
    
    # Ensure V is positive
    V_safe = np.maximum(V, eps)
    
    # Check time
    if time.time() > deadline - 1.0:
        # Fallback to simple heuristic if cluster calc took too long (unlikely for N=50)
        H = V_safe[None, :] / (cost_safe ** 3)
        H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
        return H
    
    # Compute multiple heuristic variants using V[j]
    # V1: V / cost^2.5
    H1 = V_safe[None, :] / (cost_safe ** 2.5)
    
    # V2: V^2 / cost^3
    H2 = (V_safe[None, :] ** 2) / (cost_safe ** 3)
    
    # V3: V / cost^3.5
    H3 = V_safe[None, :] / (cost_safe ** 3.5)
    
    # V4: V^2 / cost^4
    H4 = (V_safe[None, :] ** 2) / (cost_safe ** 4)
    
    # Check time
    if time.time() > deadline - 0.5:
        H = H1
        H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
        return H
    
    # Normalize each variant to [0, 1] range for blending
    def normalize(H_mat):
        h = np.where(np.isfinite(H_mat) & (H_mat > 0), H_mat, 0.0)
        max_val = np.max(h)
        if max_val > 0:
            return h / max_val
        return h
    
    H1_n = normalize(H1)
    H2_n = normalize(H2)
    H3_n = normalize(H3)
    H4_n = normalize(H4)
    
    # Blend with weights: favor the variants that balance local contrast with reachability
    # The cluster value V[j] already incorporates reachability, so we can use standard weights.
    w1, w2, w3, w4 = 0.25, 0.25, 0.25, 0.25
    H = w1 * H1_n + w2 * H2_n + w3 * H3_n + w4 * H4_n
    
    # Refinement: Add a small bonus for nodes that are close to the origin
    # This helps in the early stages (i=0) to pick nodes that are easy to return from.
    origin_bonus = 1.0 / (1.0 + dist_to_origin[None, :])  # shape (1, n)
    
    # Combine: H = H * (1 + 0.15 * origin_bonus)
    H = H * (1.0 + 0.15 * origin_bonus)
    
    # Check time
    if time.time() > deadline - 0.2:
        H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
        return H
    
    # Ensure no negative or non-finite values
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    
    # Final safety: ensure shape is correct
    if H.shape != (n, n):
        return np.ones((n, n), dtype=float) * 1e-9
    
    return H
# EVOLVE-BLOCK-END
