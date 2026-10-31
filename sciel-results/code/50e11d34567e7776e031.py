# EVOLVE-BLOCK-START
import numpy as np
import time

# Module-level state for deadline tracking and adaptive parameters
_start_time = None
_LIMIT = 145.0  # Safety margin before 150s kill

def _compute_mst_weight(nodes, distance_matrix):
    """Compute MST weight for a set of nodes using Prim's algorithm (O(n^2))."""
    n = len(nodes)
    if n <= 1:
        return 0.0
    
    # Use nodes as indices into a local array
    in_mst = np.zeros(n, dtype=bool)
    min_dist = np.full(n, np.inf)
    min_dist[0] = 0.0
    total_weight = 0.0
    
    for _ in range(n):
        # Find the node with minimum distance
        best_idx = -1
        best_val = np.inf
        for i in range(n):
            if not in_mst[i] and min_dist[i] < best_val:
                best_val = min_dist[i]
                best_idx = i
        
        if best_idx == -1:
            break
            
        in_mst[best_idx] = True
        total_weight += best_val
        
        # Update distances
        node_global = nodes[best_idx]
        for j in range(n):
            if not in_mst[j]:
                node_global_j = nodes[j]
                d = distance_matrix[node_global, node_global_j]
                if d < min_dist[j]:
                    min_dist[j] = d
                    
    return total_weight


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Hybrid TSP constructive heuristic combining:
    1. Detour-based geometric alignment (from current program)
    2. MST connectivity preservation (from inspiration 1)
    
    The detour score captures local geometric efficiency (how much extra distance
    the candidate adds to the path from current to destination).
    The MST penalty captures global structural impact (how much removing the candidate
    degrades the connectivity of the remaining unvisited set).
    
    This combination addresses both local path efficiency and global tour structure,
    potentially avoiding the "tunneling" behavior of pure greedy methods.
    """
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > _LIMIT:
        # Fallback to nearest neighbor for safety
        if len(unvisited_nodes) == 0:
            return -1
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    if n_unvisited == 2:
        # For 2 nodes, evaluate both orderings
        c1 = int(unvisited_nodes[0])
        c2 = int(unvisited_nodes[1])
        cost1 = distance_matrix[current_node, c1] + distance_matrix[c1, c2] + distance_matrix[c2, destination_node]
        cost2 = distance_matrix[current_node, c2] + distance_matrix[c2, c1] + distance_matrix[c1, destination_node]
        return c1 if cost1 <= cost2 else c2
    
    # Compute detour-based scores (geometric alignment)
    dist_to_candidates = distance_matrix[current_node, unvisited_nodes]
    dist_to_dest = distance_matrix[unvisited_nodes, destination_node]
    dist_current_to_dest = distance_matrix[current_node, destination_node]
    
    detours = dist_to_candidates + dist_to_dest - dist_current_to_dest
    
    # Normalize detours
    max_detour = np.max(detours)
    if max_detour > 1e-10:
        normalized_detours = detours / max_detour
    else:
        normalized_detours = detours
    
    # Compute MST-based penalties (structural connectivity)
    # Only do this if we have time budget
    remaining_time = _LIMIT - elapsed
    if remaining_time > 1.0 and n_unvisited <= 40:
        base_mst = _compute_mst_weight(unvisited_nodes, distance_matrix)
        
        # For each candidate, compute MST of remaining
        mst_penalties = np.zeros(n_unvisited)
        for i in range(n_unvisited):
            if time.time() - _start_time > _LIMIT - 0.5:
                break
            c = unvisited_nodes[i]
            remaining = np.concatenate([unvisited_nodes[:i], unvisited_nodes[i+1:]])
            if len(remaining) == 0:
                new_mst = 0.0
            else:
                new_mst = _compute_mst_weight(remaining, distance_matrix)
            mst_penalties[i] = max(0.0, new_mst - base_mst)
        
        # Normalize MST penalties
        max_mst_pen = np.max(mst_penalties)
        if max_mst_pen > 1e-10:
            normalized_mst_pen = mst_penalties / max_mst_pen
        else:
            normalized_mst_pen = mst_penalties
    else:
        # Fallback: use distance to destination as a proxy for structural impact
        normalized_mst_pen = dist_to_dest / (np.max(dist_to_dest) + 1e-10)
    
    # Adaptive weighting: earlier in the tour, weight MST penalty more heavily
    # (structure matters more when many nodes remain); later, weight local distance more
    fraction_remaining = n_unvisited / 50.0  # Assuming 50-city instances
    alpha = 0.5 + 0.3 * (1 - fraction_remaining)  # Local distance weight
    beta = 0.2 + 0.3 * fraction_remaining       # Detour weight
    gamma = 0.3 * fraction_remaining             # MST penalty weight
    
    # Normalize local distances
    max_dist = np.max(dist_to_candidates)
    if max_dist > 1e-10:
        normalized_local = dist_to_candidates / max_dist
    else:
        normalized_local = dist_to_candidates
    
    scores = alpha * normalized_local + beta * normalized_detours + gamma * normalized_mst_pen
    
    best_idx = np.argmin(scores)
    return int(unvisited_nodes[best_idx])
# EVOLVE-BLOCK-END
