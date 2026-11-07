"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(prizes, distance_matrix, max_len):
    n = len(prizes)
    D = np.asarray(distance_matrix, dtype=float)
    P = np.asarray(prizes, dtype=float)
    
    # d_0i: distance from start (node 0) to each node i. Shape (n, 1)
    d_0i = D[0, :].reshape(-1, 1)
    
    # d_j0: distance from each node j to start (node 0). Shape (1, n)
    d_j0 = D[:, 0].reshape(1, -1)
    
    # d_ij: distance matrix. Shape (n, n)
    d_ij = D
    
    # Calculate the minimum possible tour length if the path was 0 -> i -> j -> 0.
    # This is a lower bound on the actual path length to i plus the extension to j and return.
    # slack = max_len - (lower_bound_path_to_i + d_ij + return_from_j)
    slack = max_len - d_0i - d_ij - d_j0
    
    # Base attractiveness: Prize of node j divided by distance from i to j.
    # We add a small epsilon to distance to avoid division by zero.
    eps = 1e-9
    base_attractiveness = P[None, :] / (d_ij + eps)
    
    # Saturation factor based on slack.
    # We normalize slack by max_len to make it scale-invariant.
    # We use a power of 2 to strongly penalize moves that consume a large fraction of the remaining budget,
    # thereby encouraging ants to preserve budget for future high-value nodes.
    # Moves with non-positive slack are infeasible even under the best-case path assumption, 
    # so they get a factor of 0 (and will be clipped to 1e-9 later).
    norm_slack = np.maximum(slack, 0.0) / max_len
    saturation_factor = norm_slack ** 2.0
    
    # Final heuristic
    H = base_attractiveness * saturation_factor
    
    # Ensure all values are positive and finite.
    # The problem statement says: "Heuristic entries that are not positive become 1e-9".
    # We explicitly apply this to ensure compliance and handle any numerical issues.
    H = np.where(np.isfinite(H) & (H > 0), H, 1e-9)
    
    return H
# EVOLVE-BLOCK-END
