"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(adjacency):
    # Convert to numpy array if not already
    adj = np.asarray(adjacency, dtype=np.int32)
    n = adj.shape[0]
    
    # Degree of each vertex
    degree = adj.sum(axis=1).astype(float)
    
    # For each vertex, compute the average degree of its neighbors
    # This helps identify vertices whose neighbors are "expensive" to keep out
    # If a vertex's neighbors have high degree, picking this vertex removes many
    # high-degree vertices from consideration, which is good
    
    # Compute sum of neighbor degrees for each vertex
    neighbor_degree_sum = adj @ degree
    
    # Average neighbor degree (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_neighbor_degree = np.where(degree > 0, neighbor_degree_sum / degree, 0)
    
    # Strategy: prefer vertices with low degree (fewer conflicts)
    # But also consider that picking a vertex with low-degree neighbors is less
    # valuable because those neighbors would likely be picked anyway
    
    # Score: lower is better for the greedy decoder (it picks highest score first)
    # We want to pick vertices that are "cheap" to include (low degree) 
    # and whose exclusion of neighbors is "expensive" (neighbors have high degree)
    
    # Primary: low degree is good
    # Secondary: high average neighbor degree means we're removing "valuable" vertices
    # that would have blocked many others
    
    # Normalize components to comparable scales
    max_degree = degree.max() if n > 0 else 1
    max_avg_neighbor = avg_neighbor_degree.max() if n > 0 else 1
    
    # Score: we want high priority for low degree and high avg neighbor degree
    # Invert degree (low degree = high priority)
    degree_score = 1.0 - (degree / max_degree) if max_degree > 0 else np.ones(n)
    
    # High avg neighbor degree = high priority
    neighbor_score = avg_neighbor_degree / max_avg_neighbor if max_avg_neighbor > 0 else np.zeros(n)
    
    # Combine: weight degree more heavily, then add neighbor consideration
    # The idea: pick low-degree vertices first, but among similar degrees,
    # prefer those whose neighbors are high-degree (so we "save" the low-degree
    # vertices that are neighbors of these high-degree ones)
    combined = 0.7 * degree_score + 0.3 * neighbor_score
    
    # Return negative so that higher combined score = lower value (picked first)
    return -combined
# EVOLVE-BLOCK-END
