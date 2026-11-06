# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard state
_t0 = None
_DEADLINE = 115.0  # Safety margin before 120s limit

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Constructive heuristic for TSP: picks the next city to visit.
    
    Uses a lookahead-based scoring that balances immediate distance with
    future connection costs (savings heuristic variant).
    """
    global _t0
    
    if _t0 is None:
        _t0 = time.time()
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return int(current_node)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # Check deadline - if we're running low on time, fall back to nearest neighbor
    if time.time() - _t0 > _DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # For small sets, use a more sophisticated scoring
    if n_unvisited <= 50:
        # Score each candidate by:
        # 1. Distance from current node to candidate
        # 2. Plus a fraction of the minimum distance from candidate to any other unvisited node
        #    (this encourages connecting nodes that have good future connections)
        # 3. Plus a small penalty if the candidate is far from the destination node
        #    (to ensure we don't get stuck far from where we need to end up)
        
        dist_from_current = distance_matrix[current_node, unvisited_nodes]
        
        # Minimum distance from each candidate to any other unvisited node (excluding itself)
        # This is O(n^2) but n <= 50 so it's fine
        min_dist_to_other = np.full(n_unvisited, np.inf)
        for i in range(n_unvisited):
            node_i = unvisited_nodes[i]
            # Distance from node_i to all other unvisited nodes except itself
            other_nodes = np.delete(unvisited_nodes, i)
            if len(other_nodes) > 0:
                min_dist_to_other[i] = np.min(distance_matrix[node_i, other_nodes])
        
        # Distance from each candidate to destination
        dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
        
        # Combine scores:
        # Primary: distance from current
        # Secondary: how well-connected the candidate is (lower min_dist_to_other is better)
        # Tertiary: distance to destination (small weight)
        
        # Normalize components to comparable scale
        max_dc = np.max(dist_from_current) if np.max(dist_from_current) > 0 else 1.0
        max_mo = np.max(min_dist_to_other) if np.max(min_dist_to_other) > 0 else 1.0
        max_dd = np.max(dist_to_dest) if np.max(dist_to_dest) > 0 else 1.0
        
        # Weighted combination
        # Higher weight on immediate distance, moderate on connectivity, low on destination
        score = (
            0.6 * (dist_from_current / max_dc) +
            0.3 * (min_dist_to_other / max_mo) +
            0.1 * (dist_to_dest / max_dd)
        )
        
        return int(unvisited_nodes[np.argmin(score)])
    
    # Fallback for very large sets (shouldn't happen with 50 cities)
    return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
# EVOLVE-BLOCK-END
