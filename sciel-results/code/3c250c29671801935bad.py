import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes the heuristic matrix for the ant colony optimization.
    
    The heuristic eta_ij represents the attractiveness of moving from node i to node j.
    We model the marginal cost of visiting node j as the sum of the distance from i to j
    and the distance from j to the depot (0), which approximates the increase in path length
    when j is appended to a path ending at i and then returning to the start.
    
    eta_ij = (prize_j + eps) / (d_ij + d_j0 + eps)^gamma
    
    A gamma > 1 penalizes long moves more heavily, encouraging ants to stay in
    high-prize-density regions and make efficient local moves.
    """
    prizes = np.asarray(prizes)
    distance_matrix = np.asarray(distance_matrix)
    
    n = distance_matrix.shape[0]
    eps = 1e-9
    gamma = 1.5
    
    # d_ij is the distance from i to j
    # d_j0 is the distance from j to depot 0
    # We compute the marginal cost matrix: cost_ij = d_ij + d_j0
    # Note: d_j0 is the same for all i for a given j, so it's a column vector added to the matrix.
    
    d_to_depot = distance_matrix[:, 0]  # Shape (n,). d[j, 0]
    
    # Cost matrix: d[i, j] + d[j, 0]
    marginal_cost = distance_matrix + d_to_depot[None, :]
    
    # Add epsilon to avoid division by zero and ensure positivity
    marginal_cost = marginal_cost + eps
    
    # Prize vector, add epsilon to ensure positive numerator even if prize is 0 (though node 0 has 0 prize, 
    # ants don't move TO node 0 usually, but the matrix is full).
    # The problem states prize 0 at node 0.
    # If prize is 0, eta should be low.
    prize_vec = prizes + eps
    
    # Calculate heuristic
    # eta[i, j] = prize[j] / (cost[i, j])^gamma
    # Using broadcasting: prize_vec is (n,), marginal_cost is (n, n)
    # We want eta[i, j] to depend on prize[j].
    # So we broadcast prize_vec along rows.
    numerator = prize_vec[None, :]  # Shape (1, n)
    
    eta = numerator / (marginal_cost ** gamma)
    
    # Ensure all values are positive and finite
    # The problem states: "Heuristic entries that are not positive become 1e-9"
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    return eta
