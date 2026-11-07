"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Construct a heuristic (edge-attractiveness) matrix for ACO-CVRP.
    
    Strategy: Combine distance-based attractiveness with a demand-aware bias.
    The base heuristic is 1/(distance + eps), which favors short edges.
    We then modulate this by a factor that accounts for the demand of the
    destination node. High-demand nodes are slightly more attractive because
    visiting them first can help pack the vehicle efficiently, reducing the
    number of trips needed.
    
    Additionally, we apply a mild angular bias to encourage route continuity
    (i.e., favoring edges that maintain the current direction of travel),
    which helps in finding smoother, shorter tours.
    
    This heuristic is static but designed to encode good local search behavior.
    """
    n = len(coordinates)
    dm = np.asarray(distance_matrix, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)
    dem = np.asarray(demands, dtype=np.float64)
    
    # Ensure valid shape
    if dm.shape != (n, n):
        raise ValueError("distance_matrix shape mismatch")
    
    # Base heuristic: inverse distance
    # Add small epsilon to avoid division by zero
    base = 1.0 / (dm + 1e-9)
    
    # Demand bias: prefer visiting high-demand nodes earlier in a tour
    # This encourages packing the truck with high-demand customers first,
    # which can reduce the number of vehicles needed.
    # Factor: 1 + (demand_j / capacity) * alpha
    # alpha is a small multiplier to control the strength of the bias.
    alpha = 0.5  # Tuned for balance between distance and demand
    demand_bias = 1.0 + alpha * (dem[np.newaxis, :] / capacity)
    
    # Angular continuity bias:
    # For each edge (i, j), we compute the angle between the incoming edge
    # to i and the outgoing edge to j. Edges that maintain direction are
    # more attractive.
    # This is a heuristic that encourages smooth paths.
    # We compute this for all pairs (i, j) where i != j.
    
    # To keep it efficient, we precompute the angle bias matrix.
    # For each i, we look at the vector from i to its nearest neighbor (as a proxy
    # for the "incoming" direction) and the vector from i to j.
    # Actually, a simpler and more robust approach: for each pair (i, j),
    # compute the angle between (coords[i] - coords[0]) and (coords[j] - coords[0])
    # where 0 is the depot. This encourages routes that go out from the depot
    # and come back, favoring edges that are "aligned" with the radial direction.
    
    # A better angular bias: for each edge (i, j), compute the angle between
    # the vector (j - i) and the average direction of the tour so far.
    # Since we don't have tour history in a static heuristic, we use a proxy:
    # the angle between (j - i) and (j - depot) or (i - depot).
    
    # Let's use: angle between (coords[j] - coords[i]) and (coords[j] - coords[0])
    # This favors edges that move away from the depot in a straight line,
    # which is good for the outbound leg.
    
    # Compute vectors
    vec_ij = coords[np.newaxis, :, :] - coords[:, np.newaxis, :]  # (i, j) -> coords[j] - coords[i]
    vec_id = coords[np.newaxis, :, :] - coords[0:1, :]  # (i, d) -> coords[j] - coords[0] (depot)
    
    # Actually, vec_id should be (j - depot) for the destination j.
    vec_jd = coords - coords[0]  # (j) -> coords[j] - coords[0]
    
    # Dot product and norms
    # vec_ij[i, j, k] is the component k of (coords[j] - coords[i])
    # vec_jd[j, k] is the component k of (coords[j] - coords[0])
    
    # For each (i, j), compute cos(theta) = (vec_ij[i,j] . vec_jd[j]) / (|vec_ij[i,j]| * |vec_jd[j]|)
    
    # To avoid division by zero, add epsilon
    vec_ij_norm = np.linalg.norm(vec_ij, axis=2) + 1e-9  # (i, j)
    vec_jd_norm = np.linalg.norm(vec_jd, axis=1) + 1e-9  # (j,)
    
    # Dot product: vec_ij[i, j, :] dot vec_jd[j, :]
    # This is a sum over k
    dot = np.sum(vec_ij * vec_jd[np.newaxis, :, :], axis=2)  # (i, j)
    
    cos_theta = dot / (vec_ij_norm * vec_jd_norm[np.newaxis, :])
    
    # Clip to [-1, 1] to avoid numerical issues
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    
    # Angular bias: favor edges with cos_theta close to 1 (same direction)
    # bias = 1 + beta * cos_theta
    beta = 0.3  # Strength of angular bias
    angular_bias = 1.0 + beta * cos_theta
    
    # Combine all biases
    eta = base * demand_bias * angular_bias
    
    # Ensure positivity and finiteness
    eta = np.where(np.isfinite(eta), eta, 1e-9)
    eta = np.maximum(eta, 1e-9)
    
    # Set diagonal to a small value (no self-loops)
    np.fill_diagonal(eta, 1e-9)
    
    return eta
# EVOLVE-BLOCK-END
