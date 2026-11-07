"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state to track time and best result
_start_time = time.time()
_DEADLINE = 320.0  # Safety margin before 330s kill

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    """
    Targeted Edge Disruption strategy for Guided Local Search.
    
    Instead of a global penalty, we identify 'trapped' edges in the current 
    local optimum (those with high usage counts) and apply a large penalty 
    specifically to them. This forces the 2-opt to break these specific 
    structural bottlenecks while preserving the true geometry of the rest 
    of the graph.
    """
    n = len(local_opt_tour)
    if n <= 2:
        return edge_distance.copy()

    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        return edge_distance.copy()

    # Extract edges of the current local optimum tour
    # local_opt_tour is a sequence of nodes, e.g., [0, 1, 2, ..., n-1]
    # The tour is closed, so edges are (tour[i], tour[(i+1)%n])
    tour_edges = []
    for i in range(n):
        u = local_opt_tour[i]
        v = local_opt_tour[(i + 1) % n]
        tour_edges.append((u, v))

    # Identify the "trapped" edges: those in the current tour with high usage.
    # We define 'trapped' as edges whose usage count is above the median usage 
    # of all edges in the current tour. This targets the most persistent 
    # components of the current local optimum.
    usage_counts = [edge_n_used[u, v] for u, v in tour_edges]
    if len(usage_counts) == 0:
        return edge_distance.copy()
    
    median_usage = np.median(usage_counts)
    
    # If all usages are 0 (first iteration), we don't have a "trapped" set 
    # in the historical sense, but we still want to perturb. 
    # However, in GLS, the first penalty is usually uniform or based on 
    # current tour. Here, if median is 0, we might just penalize the whole 
    # tour lightly or return original. 
    # But the task says edge_n_used is accumulated over rounds. 
    # In round 1, edge_n_used is likely 0 or just added the current tour.
    # Let's assume edge_n_used includes the current tour's edges already 
    # (as per "add the current tour's edges to edge_n_used" before calling).
    
    # If max usage is 0, we can't distinguish. Return original to avoid 
    # unnecessary distortion in the very first step? 
    # Actually, if we return original, 2-opt on original distances from 
    # a 2-opt optimal tour will do nothing. 
    # So we MUST perturb. 
    # If median is 0, it means all edges in the tour have 0 count? 
    # That contradicts "add current tour to edge_n_used". 
    # So usage_counts should be >= 1 for all tour edges.
    
    # Let's refine: We want to break the edges that are part of the 
    # "core" of the local optimum. 
    # A simple heuristic: Penalize the edges in the current tour that have 
    # the highest usage counts. 
    # Let's take the top 20% most used edges in the current tour.
    
    sorted_edges = sorted(tour_edges, key=lambda e: edge_n_used[e[0], e[1]], reverse=True)
    num_to_penalize = max(1, n // 5)  # Penalize at least 1, up to 20%
    target_edges = sorted_edges[:num_to_penalize]
    
    # Create a copy of the distance matrix
    new_dist = edge_distance.copy()
    
    # Apply a large penalty to the target edges
    # The penalty should be large enough to make 2-opt avoid them, 
    # but not so large that it creates non-Euclidean artifacts that 
    # confuse the subsequent 2-opt on true distances.
    # A common value in GLS is 1.1x to 1.5x, but for "breaking" a tour, 
    # we might need more. 
    # Let's try a multiplier of 2.0 for the target edges.
    for u, v in target_edges:
        new_dist[u, v] *= 2.0
        new_dist[v, u] *= 2.0
        
    return new_dist
# EVOLVE-BLOCK-END
