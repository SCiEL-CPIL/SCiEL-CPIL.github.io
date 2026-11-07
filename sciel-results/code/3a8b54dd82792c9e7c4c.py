import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Computes the heuristic matrix for the ACO algorithm.
    Combines inverse distance with a savings-based multiplier and angular continuity
    to guide ants towards edges that form compact, non-crossing routes.
    """
    # Ensure inputs are numpy arrays of type float
    d = np.asarray(distance_matrix, dtype=float)
    coords = np.asarray(coordinates, dtype=float)
    demands = np.asarray(demands, dtype=float)
    
    n = d.shape[0]
    
    # 1. Base Distance Heuristic: 1 / (d^2 + eps)
    # We use a softened denominator (1e-4) to prevent extreme values for very close points,
    # which can cause premature convergence in ACO.
    eps = 1e-4
    dist_heur = 1.0 / (d**2 + eps)
    
    # 2. Savings Heuristic (Clarke-Wright style)
    # S[i, j] = d[i, 0] + d[0, j] - d[i, j]
    # This measures how much distance is saved by visiting i and j consecutively 
    # instead of separately from the depot.
    d_i0 = d[:, 0].reshape(-1, 1)  # (n, 1)
    d_0j = d[0, :].reshape(1, -1)  # (1, n)
    
    S = d_i0 + d_0j - d
    
    # Clip negative savings to 0 (shouldn't happen with Euclidean, but safe)
    S = np.maximum(S, 0)
    
    # Normalize savings to [0, 1] to act as a multiplier
    S_max = np.max(S)
    if S_max > 0:
        S_norm = S / S_max
    else:
        S_norm = np.zeros_like(S)
    
    # 3. Angular Continuity Heuristic
    # Compute angles from the depot (node 0) to all other nodes.
    # Edges that preserve angular order (small change in angle) are encouraged
    # to reduce route crossings and create compact, spiral-like routes.
    depot = coords[0]
    # Vector from depot to all nodes
    vecs = coords - depot
    # Angles (radians)
    angles = np.arctan2(vecs[:, 1], vecs[:, 0])
    
    # Compute angular difference matrix
    # For each pair (i, j), compute the angular difference.
    # We want small angular differences to be more attractive.
    # Note: For i=0 or j=0, the angle is not well-defined in the same sense, 
    # but savings handles those. We can just compute for all and let savings dominate for depot edges.
    # Actually, for depot edges, the "angular continuity" concept is less clear.
    # Let's compute the angular difference for all pairs, but note that for depot,
    # the angle is the angle of the node itself.
    
    # Angle difference: abs(angle_i - angle_j), wrapped to [0, pi]
    # Since angles are in [-pi, pi], the difference is (angle_i - angle_j) % (2*pi)
    # Then take min(diff, 2*pi - diff) to get the shortest arc.
    
    angle_diff = np.abs(angles[:, np.newaxis] - angles[np.newaxis, :])
    # Wrap to [0, 2*pi]
    angle_diff = angle_diff % (2 * np.pi)
    # Shortest arc
    angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
    
    # Convert angular difference to a multiplier.
    # Small angle_diff -> high multiplier.
    # Use cosine: cos(angle_diff) is 1 at 0, 0 at pi/2, -1 at pi.
    # We want positive values. Let's use (1 + cos(angle_diff)) / 2, which is in [0, 1].
    # Or simply cos(angle_diff) clipped to [0, 1]? No, we want to penalize large differences.
    # Let's use exp(-k * angle_diff) or similar.
    # The observed change mentioned "angular continuity... reduce route crossings".
    # A simple way: multiplier = 1 + lambda_a * (1 - angle_diff / pi)
    # Or: multiplier = cos(angle_diff) + 1 (range [0, 2])
    
    # Let's use a smooth decay: exp(-angle_diff)
    # Or, to be consistent with the savings multiplier style (1 + lambda * norm),
    # let's normalize angle_diff to [0, 1] and use (1 - norm_angle).
    
    max_angle = np.pi
    norm_angle = angle_diff / max_angle  # in [0, 1]
    
    # Angular multiplier: higher when angle_diff is small.
    # mult = 1 + lambda_a * (1 - norm_angle)
    lambda_a = 0.5
    angular_mult = 1.0 + lambda_a * (1.0 - norm_angle)
    
    # 4. Combine Heuristics
    # eta[i, j] = dist_heur[i, j] * (1 + lambda_s * S_norm[i, j]) * angular_mult[i, j]
    
    lambda_s = 1.5
    
    savings_mult = 1.0 + lambda_s * S_norm
    
    heuristic = dist_heur * savings_mult * angular_mult
    
    # 5. Handle Diagonal and Invalid Values
    # The diagonal represents self-loops, which are invalid moves.
    # The task states: "Heuristic entries that are not positive become 1e-9"
    # We explicitly set diagonal to 0 first, then apply the mask.
    np.fill_diagonal(heuristic, 0.0)
    
    # Mask for non-finite or non-positive values
    mask = ~np.isfinite(heuristic) | (heuristic <= 0)
    heuristic[mask] = 1e-9
    
    return heuristic
