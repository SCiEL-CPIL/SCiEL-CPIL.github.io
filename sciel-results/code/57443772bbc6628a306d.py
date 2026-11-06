"""Improved TSP constructive heuristic with multi-candidate lookahead."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations

# Module-level state for time management
_start_time = None

def _get_start_time():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return _start_time

def _time_exceeded(limit=140.0):
    """Return True if we've exceeded the time limit (with safety margin)."""
    return (time.time() - _get_start_time()) > limit

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Select the next node in the TSP tour using a multi-candidate lookahead strategy.
    
    Parameters:
    - current_node: The node we're currently at
    - destination_node: The final destination (where we must return)
    - unvisited_nodes: Array of unvisited node indices
    - distance_matrix: NxN matrix of distances
    
    Returns:
    - The index of the next node to visit
    """
    # Check time budget
    if _time_exceeded(140.0):
        # Fallback to nearest neighbor if time is running out
        if len(unvisited_nodes) == 0:
            return int(current_node)
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return int(current_node)
    
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    # For small sets, we can do exact optimization
    if n_unvisited <= 8:
        return _exact_lookahead(current_node, destination_node, unvisited_nodes, distance_matrix)
    
    # Select top K candidates based on immediate distance
    K = min(12, n_unvisited)
    
    # Get distances from current node to all unvisited nodes
    dists_from_current = distance_matrix[current_node, unvisited_nodes]
    
    # Get top K candidates
    top_k_indices = np.argsort(dists_from_current)[:K]
    top_k_nodes = unvisited_nodes[top_k_indices]
    
    # For each top candidate, estimate the cost of completing the tour
    # using a greedy completion with some lookahead
    best_candidate = top_k_nodes[0]
    best_score = float('inf')
    
    # We'll use a 2-step lookahead for each candidate
    for i, cand in enumerate(top_k_nodes):
        if _time_exceeded(140.0):
            break
        
        # Cost to go from current to candidate
        cost_to_cand = distance_matrix[current_node, cand]
        
        # Estimate remaining cost: from candidate, greedily visit remaining unvisited
        # and return to destination
        remaining = np.setdiff1d(unvisited_nodes, [cand])
        
        if len(remaining) == 0:
            total_est = cost_to_cand + distance_matrix[cand, destination_node]
        else:
            # Greedy estimate: from cand, visit nearest unvisited, etc.
            # But to be faster, use a 2-step lookahead
            # Step 1: nearest from cand
            dists_from_cand = distance_matrix[cand, remaining]
            nearest_idx = np.argmin(dists_from_cand)
            next_node = remaining[nearest_idx]
            
            # Cost: current -> cand -> next_node -> (greedy completion estimate)
            cost_step1 = cost_to_cand + distance_matrix[cand, next_node]
            
            # For the rest, use average distance heuristic
            remaining_after_next = np.setdiff1d(remaining, [next_node])
            if len(remaining_after_next) == 0:
                cost_rest = distance_matrix[next_node, destination_node]
            else:
                # Estimate: average distance from next_node to remaining + return to destination
                avg_dist = np.mean(distance_matrix[next_node, remaining_after_next])
                cost_rest = len(remaining_after_next) * avg_dist + distance_matrix[next_node, destination_node]
            
            total_est = cost_step1 + cost_rest
        
        if total_est < best_score:
            best_score = total_est
            best_candidate = cand
    
    return int(best_candidate)

def _exact_lookahead(current_node, destination_node, unvisited_nodes, distance_matrix):
    """
    Exact optimization for small sets of unvisited nodes (<= 8).
    Uses dynamic programming to find the optimal next node.
    """
    n = len(unvisited_nodes)
    
    if n == 0:
        return int(current_node)
    
    if n == 1:
        return int(unvisited_nodes[0])
    
    # Map unvisited nodes to indices 0..n-1
    node_map = {node: i for i, node in enumerate(unvisited_nodes)}
    
    # Build sub distance matrix
    sub_dm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sub_dm[i, j] = distance_matrix[unvisited_nodes[i], unvisited_nodes[j]]
    
    # DP: dp[mask][i] = minimum cost to visit all nodes in mask, ending at node i
    # mask is a bitmask of visited nodes among the n unvisited nodes
    # We want to find which first node to visit minimizes total cost
    
    # Initialize DP
    size = 1 << n
    dp = np.full((size, n), np.inf)
    
    # Base case: only one node visited
    for i in range(n):
        dp[1 << i][i] = distance_matrix[current_node, unvisited_nodes[i]]
    
    # Fill DP table
    for mask in range(1, size):
        for i in range(n):
            if not (mask & (1 << i)):
                continue
            if dp[mask][i] == np.inf:
                continue
            # Try extending to node j not in mask
            for j in range(n):
                if mask & (1 << j):
                    continue
                new_mask = mask | (1 << j)
                new_cost = dp[mask][i] + sub_dm[i, j]
                if new_cost < dp[new_mask][j]:
                    dp[new_mask][j] = new_cost
    
    # Find the best first node: the one that leads to minimum total tour cost
    full_mask = size - 1
    best_first = 0
    best_total = np.inf
    
    for i in range(n):
        # Cost: current -> i -> (complete tour visiting all others) -> destination
        if dp[full_mask][i] == np.inf:
            continue
        total = dp[full_mask][i] + distance_matrix[unvisited_nodes[i], destination_node]
        if total < best_total:
            best_total = total
            best_first = i
    
    return int(unvisited_nodes[best_first])
# EVOLVE-BLOCK-END
