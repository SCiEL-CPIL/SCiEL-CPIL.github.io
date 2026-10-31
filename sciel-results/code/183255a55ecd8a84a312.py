"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline guard
_DEADLINE = None
_DEADLINE_MARGIN = 5.0  # seconds of safety margin before the 120s limit (or 150s)

def _init_deadline():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = time.time() + 150.0 - _DEADLINE_MARGIN
    return _DEADLINE

def _time_exceeded():
    global _DEADLINE
    if _DEADLINE is None:
        _init_deadline()
    return time.time() > _DEADLINE

# Cache for MST computations and other per-instance data
_instance_cache = {}

def _get_instance_key(distance_matrix):
    # Use a hash of the distance matrix shape and a few values to identify instances
    # Since the same instance is used for multiple calls, we can cache based on shape and a fingerprint
    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        return None
    n = distance_matrix.shape[0]
    # Simple fingerprint: first few elements
    fp = tuple(distance_matrix.flat[:min(10, distance_matrix.size)])
    return (n, fp)

def _compute_mst_weight(distance_matrix, nodes):
    """Compute MST weight of the subgraph induced by nodes using Prim's algorithm."""
    if len(nodes) <= 1:
        return 0.0
    if len(nodes) == 2:
        return float(distance_matrix[nodes[0], nodes[1]])
    
    n = len(nodes)
    # Map node indices to 0..n-1
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    # Prim's algorithm
    in_mst = [False] * n
    min_edge = [float('inf')] * n
    min_edge[0] = 0.0
    total_weight = 0.0
    
    for _ in range(n):
        # Find the node with minimum edge weight not yet in MST
        u = -1
        best = float('inf')
        for i in range(n):
            if not in_mst[i] and min_edge[i] < best:
                best = min_edge[i]
                u = i
        
        if u == -1:
            # Graph disconnected, shouldn't happen for complete graph
            break
        
        in_mst[u] = True
        total_weight += min_edge[u]
        
        # Update min_edge for neighbors
        u_node = nodes[u]
        for v in range(n):
            if not in_mst[v]:
                v_node = nodes[v]
                w = distance_matrix[u_node, v_node]
                if w < min_edge[v]:
                    min_edge[v] = w
    
    return total_weight

def _mst_reduction_score(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Score each candidate node by the reduction in MST weight if that node is added to the tour.
    We want to maximize the reduction (i.e., minimize the remaining MST weight).
    
    Actually, let's think about this differently. When we pick a node, we're adding an edge from
    current_node to that node. The remaining unvisited nodes will form the rest of the tour.
    
    A better heuristic: for each candidate, estimate the total tour cost if we pick that candidate.
    This is expensive. Instead, let's use a simpler but still global heuristic.
    
    Alternative: pick the node that minimizes the sum of:
    1. Distance from current_node to candidate
    2. Distance from candidate to destination_node (or to the nearest unvisited node)
    3. Some measure of how "central" the candidate is to the remaining nodes
    
    Let's try: for each candidate, compute the MST weight of the remaining nodes after removing it.
    We want to minimize: dist(current, candidate) + MST_weight(remaining_after_candidate)
    
    But MST computation for each candidate is O(n^2) per candidate, and there are O(n) candidates,
    so total O(n^3) per step, which for n=50 is 125,000 operations per step, and 50 steps = 6.25M.
    That should be fine within time limits.
    """
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # For each candidate, compute the score
    best_score = float('inf')
    best_node = int(unvisited_nodes[0])
    
    n = len(unvisited_nodes)
    
    for i in range(n):
        candidate = unvisited_nodes[i]
        
        # Cost to reach candidate from current
        cost_to_candidate = distance_matrix[current_node, candidate]
        
        # Remaining nodes after picking candidate
        remaining = np.delete(unvisited_nodes, i)
        
        if len(remaining) == 0:
            # This is the last node, cost is just the edge back to destination
            total_cost = cost_to_candidate + distance_matrix[candidate, destination_node]
        else:
            # Estimate the cost of connecting remaining nodes
            # Use MST weight of remaining as a lower bound on the path cost
            mst_w = _compute_mst_weight(distance_matrix, remaining)
            
            # Also account for connecting from candidate to the first node in remaining,
            # and from last node in remaining to destination
            # Approximate: min distance from candidate to any remaining node + MST + min distance from any remaining to destination
            min_from_candidate = np.min(distance_matrix[candidate, remaining])
            min_to_destination = np.min(distance_matrix[remaining, destination_node])
            
            total_cost = cost_to_candidate + min_from_candidate + mst_w + min_to_destination
        
        if total_cost < best_score:
            best_score = total_cost
            best_node = int(candidate)
    
    return best_node

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node in the TSP tour using an MST-aware greedy heuristic.
    """
    if len(unvisited_nodes) == 0:
        return -1  # Should not happen
    
    if len(unvisited_nodes) == 1:
        return int(unvisited_nodes[0])
    
    # Check deadline
    if _time_exceeded():
        # Fallback to nearest neighbor
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    # Use MST-aware heuristic
    try:
        return _mst_reduction_score(current_node, destination_node, unvisited_nodes, distance_matrix)
    except Exception:
        # Fallback to nearest neighbor on any error
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
# EVOLVE-BLOCK-END
