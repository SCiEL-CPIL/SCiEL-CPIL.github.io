import numpy as np

def heuristics(prizes, distance_matrix, max_len):
    """
    Computes the heuristic matrix for the ant colony optimization.
    
    Structural Change: Incorporates node centrality (inverse distance sum to other nodes)
    to encourage ants to move towards clusters of nodes, which reduces the total
    travel cost per prize collected.
    
    $\eta_{ij} = \frac{prize_j \cdot (1 + \text{centrality}_j)}{d_{ij} \cdot d_{j0}}$
    """
    prizes = np.asarray(prizes)
    distance_matrix = np.asarray(distance_matrix)
    
    n = distance_matrix.shape[0]
    eps = 1e-9
    
    d = distance_matrix + eps
    d_to_depot = d[:, 0]
    
    # Calculate centrality: sum of inverse distances to other non-depot nodes
    # c_j = sum_{i != j, i != 0} 1/d_ij
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    mask[0, :] = False  # Exclude depot (node 0) as a source i
    
    inv_d = 1.0 / d
    inv_d_masked = inv_d * mask
    
    centrality = np.sum(inv_d_masked, axis=0)
    
    # Normalize centrality to mean 1 for non-depot nodes to keep scale reasonable
    non_depot_centrality = centrality[1:]
    mean_c = np.mean(non_depot_centrality) if len(non_depot_centrality) > 0 else 1.0
    if mean_c > 0:
        centrality_norm = centrality / mean_c
    else:
        centrality_norm = np.ones(n)
        
    # Node attraction: Prize scaled by (1 + centrality)
    # This boosts nodes that are well-connected to other nodes
    node_attraction = prizes * (1.0 + centrality_norm)
    
    # Edge attractiveness
    eta = node_attraction[None, :] / d
    
    # Ensure all values are positive and finite
    eta = np.where(np.isfinite(eta) & (eta > 0), eta, 1e-9)
    
    return eta
