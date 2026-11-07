"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_GLOBAL_START_TIME = time.time()
_WALL_CLOCK_LIMIT = 325.0  # Safety margin before 330s kill

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    # Check time limit - return original distances if we're running out of time
    if time.time() - _GLOBAL_START_TIME > _WALL_CLOCK_LIMIT:
        return edge_distance.copy()
    
    # Create a copy of the true distance matrix to modify
    modified_dist = edge_distance.copy()
    
    n = edge_distance.shape[0]
    
    # Strategy: Hard Edge Exclusion (Structural Perturbation)
    #
    # Rationale:
    # The standard GLS approach (and the current program) uses a "soft" penalty
    # (additive or multiplicative) to make previously used edges less attractive.
    # However, 2-opt is a local search method. If the penalty is not large enough
    # to overcome the immediate gain from swapping back in a good edge, the search
    # might oscillate or fail to escape the local basin effectively.
    #
    # By setting the distance of all edges in the current local-optimal tour to a
    # very large value (effectively infinite), we force the 2-opt algorithm to
    # remove these edges. Since a valid tour must have exactly 2 edges per node,
    # removing an edge from the current tour forces the 2-opt to replace it with
    # a completely new edge that was not part of the current local optimum.
    #
    # This creates a "hard constraint" perturbation. It guarantees that the next
    # 2-opt pass will result in a tour that is structurally different from the
    # current one, breaking the symmetry of the local search and potentially
    # jumping to a different basin of attraction.
    #
    # We use a large finite value (1e6) instead of np.inf to ensure the matrix
    # remains finite and compatible with any downstream operations that might
    # not handle inf gracefully, although 2-opt typically handles large values
    # fine. 1e6 is much larger than any Euclidean distance in the unit square
    # (max ~1.414), so it effectively acts as a barrier.
    
    # Get the edges in the current local_opt_tour
    # local_opt_tour is a list/array of node indices representing the tour order
    # Edges are (tour[i], tour[i+1]) for i in 0..n-2 and (tour[n-1], tour[0])
    
    # We need to modify both (u, v) and (v, u) since the matrix is symmetric
    # and 2-opt might check either direction depending on implementation,
    # though typically it's symmetric.
    
    # Vectorized approach to set these edges to a large value
    # Create indices for the edges
    # u = local_opt_tour[:-1]
    # v = local_opt_tour[1:]
    # Also need the closing edge: (local_opt_tour[-1], local_opt_tour[0])
    
    # Let's construct the full list of edges
    # Using numpy for speed, though n=100 is small
    # u_nodes = np.array(local_opt_tour[:-1])
    # v_nodes = np.array(local_opt_tour[1:])
    # closing_u = local_opt_tour[-1]
    # closing_v = local_opt_tour[0]
    
    # Simpler: just loop or use advanced indexing
    # Since n is small (100), a loop is fast enough and clear, but let's use numpy indexing.
    
    # Get all pairs
    # pairs = np.column_stack((local_opt_tour[:-1], local_opt_tour[1:]))
    # pairs = np.vstack((pairs, [[local_opt_tour[-1], local_opt_tour[0]]]))
    
    # Even simpler:
    u = local_opt_tour[:-1]
    v = local_opt_tour[1:]
    
    # Set distances to large value
    modified_dist[u, v] = 1e6
    modified_dist[v, u] = 1e6
    
    # Don't forget the closing edge
    modified_dist[local_opt_tour[-1], local_opt_tour[0]] = 1e6
    modified_dist[local_opt_tour[0], local_opt_tour[-1]] = 1e6
    
    return modified_dist
# EVOLVE-BLOCK-END
