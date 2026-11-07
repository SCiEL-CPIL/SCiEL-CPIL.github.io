import numpy as np

def heuristics(distance_matrix, coordinates, demands, capacity):
    n = distance_matrix.shape[0]
    # demands is a list/array of floats. Node 0 is depot, demand 0.
    # We want to create a matrix eta[i, j]
    
    # 1. Calculate vectors from depot (0) to all nodes
    # coordinates shape (n, 2)
    depot = coordinates[0]
    # vectors[i] = coordinates[i] - depot
    vectors = coordinates - depot
    
    # 2. Calculate norms (distances from depot)
    # dist_depot[i] = ||vectors[i]||
    dist_depot = np.linalg.norm(vectors, axis=1)
    
    # 3. Calculate cosine of angles between all pairs (i, j)
    # We need cos(theta_ij) = (v_i . v_j) / (||v_i|| ||v_j||)
    # To avoid division by zero for depot (which is 0 vector), we handle it separately or mask.
    # The heuristic is used for i->j where i is current, j is next.
    # If i is depot, we pick a customer.
    # If i is customer, we pick next customer or depot.
    
    # Matrix of dot products
    # dot[i, j] = v_i . v_j
    dot = np.dot(vectors, vectors.T)
    
    # Matrix of products of norms
    # norm_prod[i, j] = dist_depot[i] * dist_depot[j]
    norm_prod = np.outer(dist_depot, dist_depot)
    
    # Cosine matrix
    # Avoid division by zero
    cos_angles = np.zeros((n, n))
    mask = norm_prod > 1e-9
    cos_angles[mask] = dot[mask] / norm_prod[mask]
    
    # For depot (index 0), the vector is 0. 
    # The angle concept doesn't apply directly from depot in the same way.
    # However, if i=0 (depot), we are starting a route. The "direction" preference is less clear 
    # or we can assume any direction is fine initially, or bias towards high demand/short dist.
    # The formula (1 + cos)/2 will be 0.5 for depot if we set cos=0.
    # Let's set cos_angles[0, :] = 0 and cos_angles[:, 0] = 0.
    # This gives a neutral directional bias of 0.5 when leaving or returning to depot.
    cos_angles[0, :] = 0.0
    cos_angles[:, 0] = 0.0
    
    # 4. Construct Heuristic
    # eta[i, j] = (demand[j] / distance[i, j]) * ((1 + cos_angles[i, j]) / 2)^gamma
    # We want larger eta to be more attractive.
    # demand[j] is the demand of the node we are moving TO.
    # This encourages filling capacity.
    
    # Create demand matrix where demand_matrix[i, j] = demands[j]
    demand_mat = np.tile(demands, (n, 1))
    
    # Distance matrix is given. Add small epsilon to avoid div by zero (i==j)
    # Note: i==j is not allowed in ACO usually, but heuristic is computed for all.
    # The ACO loop will mask visited nodes.
    safe_dist = distance_matrix + 1e-9
    
    # Base term: Demand / Distance
    base = demand_mat / safe_dist
    
    # Directional term: ((1 + cos) / 2)^gamma
    # Let's use gamma = 2 to strongly penalize zig-zags (angles > 90 deg)
    directional = ((1.0 + cos_angles) / 2.0) ** 2
    
    # Combine
    eta = base * directional
    
    # Ensure positivity and finite values
    # The task says: "Heuristic entries that are not positive become 1e-9"
    # Our eta is positive as long as demand > 0 and dist > 0.
    # For depot (j=0), demand is 0, so eta[i, 0] = 0.
    # This is fine because the ACO logic usually handles return to depot separately or 
    # the probability for depot is handled by the "if none fits" rule.
    # However, if the ACO loop *does* consider the depot as a candidate during the build phase 
    # (which it shouldn't if it checks "unvisited customer"), it might be an issue.
    # The contract says: "picks an unvisited customer... When none fits, it returns to the depot".
    # So the depot is not a candidate in the main probability calculation.
    # Thus, eta[i, 0] values don't matter much, but we should set them to a small positive 
    # or 1e-9 to be safe if the implementation is sloppy.
    
    # Set non-positive or zero to 1e-9
    eta[eta <= 0] = 1e-9
    eta[~np.isfinite(eta)] = 1e-9
    
    return eta
