"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Potential-field heuristic with density-aware gradient for ACO CVRP.
    
    Design (structural change from depot-pressure field):
    - Base: power-law inverse distance (1/d^1.5) to penalize long jumps more strongly
    - Node potential field: each node has a potential based on local demand density
      and proximity to the centroid of demand. Edges are boosted when they align
      with the gradient toward high-density, central regions.
    - This creates a stronger gradient toward dense, central demand clusters,
      encouraging ants to form compact routes that serve high-density areas first.
    
    The key difference from previous approaches:
    - Instead of a simple depot-proximity * demand boost, we compute a proper
      potential field where each node's potential depends on the weighted
      average distance to all other demand nodes (a Voronoi-like density measure).
    - Edges are scaled by how much they decrease the potential (moving toward
      denser/central regions), creating a natural gradient-following behavior.
    """
    n = distance_matrix.shape[0]
    demands = np.asarray(demands, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)
    
    # Base: power-law inverse distance (stronger penalty for long jumps)
    base = 1.0 / (distance_matrix + 1e-9) ** 1.5
    
    # Compute demand-weighted centroid
    total_demand = demands.sum()
    if total_demand > 0:
        centroid = np.average(coords, weights=demands, axis=0)
    else:
        centroid = coords.mean(axis=0)
    
    # Distance from each node to the demand centroid
    dist_to_centroid = np.linalg.norm(coords - centroid, axis=1)
    
    # Local density: for each node, compute the demand-weighted average distance
    # to other nodes. Lower values indicate higher local density.
    # Use a kernel-based density estimate
    sigma = 0.15  # Bandwidth for density estimation
    dist_matrix = distance_matrix.copy()
    np.fill_diagonal(dist_matrix, 0)
    
    # Gaussian kernel density weighted by demand
    kernel = np.exp(-dist_matrix ** 2 / (2 * sigma ** 2))
    # Density at node i: sum over j of demand[j] * kernel(i,j)
    density = kernel.T @ demands  # density[i] = sum_j kernel[i,j] * demands[j]
    
    # Normalize density to [0, 1]
    max_density = density.max() if density.max() > 0 else 1.0
    density_norm = density / max_density
    
    # Node potential: high density + close to centroid = high potential
    # Normalize dist_to_centroid
    max_dist_centroid = dist_to_centroid.max() if dist_to_centroid.max() > 0 else 1.0
    dist_centroid_norm = dist_to_centroid / max_dist_centroid
    
    # Potential: higher is better (want to be in dense, central regions)
    potential = density_norm * (1.0 - 0.5 * dist_centroid_norm)
    
    # For edge (i, j): boost if moving from i to j increases potential
    # (i.e., j has higher potential than i)
    # Use a soft version: boost = max(0, potential[j] - potential[i]) + small base
    potential_diff = potential[None, :] - potential[:, None]  # shape (n, n)
    
    # Only boost when moving toward higher potential (positive diff)
    # Use a sigmoid-like softplus for smoothness
    gamma = 5.0  # Sharpness of the gradient following
    gradient_boost = np.exp(gamma * potential_diff) - 1.0  # softplus-like
    gradient_boost = np.maximum(gradient_boost, 0)  # Only boost toward higher potential
    
    # Normalize gradient boost to reasonable range
    max_gradient = gradient_boost.max() if gradient_boost.max() > 0 else 1.0
    gradient_boost = gradient_boost / max_gradient
    
    # Combine: base * (1 + alpha * gradient_boost)
    alpha = 0.8
    
    heuristic = base * (1.0 + alpha * gradient_boost)
    
    # Ensure positive values
    heuristic = np.maximum(heuristic, 1e-9)
    
    return heuristic
# EVOLVE-BLOCK-END
