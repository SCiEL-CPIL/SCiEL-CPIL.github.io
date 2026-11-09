"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

_start_time = None

def _get_time_remaining():
    """Return seconds remaining before the 150s limit (with safety margin)."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    elapsed = time.time() - _start_time
    return max(0.0, 145.0 - elapsed)

def _compute_mst_leaf_savings(candidates, distance_matrix):
    """
    Compute MST over the candidate nodes.
    Identify leaves in the MST.
    For each leaf, calculate the 'savings' which is the reduction in MST cost 
    if that leaf were removed (i.e., the cost of its only MST edge).
    Returns a dictionary: {node_id: savings_value}
    """
    n = len(candidates)
    if n <= 1:
        return {}
    
    # Build a dense subgraph for the candidates
    # We need the MST of the graph induced by 'candidates'
    # Using Prim's algorithm on the subset
    
    # Map candidate indices to 0..n-1 for easier internal handling
    # candidates is an array of node IDs from the original graph
    
    # Initialize MST structure
    # We will use a simple O(n^2) Prim's algorithm since n is at most 50
    # This is fast enough for 50 nodes (2500 ops per call, 50 calls = 125k ops)
    
    visited = [False] * n
    min_edge_weight = [np.inf] * n
    parent = [-1] * n
    
    # Start from node 0 in the candidate list
    min_edge_weight[0] = 0
    
    mst_edges = []  # List of (u, v, weight)
    
    for _ in range(n):
        # Find the unvisited node with the minimum edge weight
        v = -1
        min_w = np.inf
        for i in range(n):
            if not visited[i] and min_edge_weight[i] < min_w:
                min_w = min_edge_weight[i]
                v = i
        
        if v == -1:
            break # Should not happen for connected graph
            
        visited[v] = True
        
        if parent[v] != -1:
            mst_edges.append((parent[v], v, min_w))
            
        # Update min_edge_weight for neighbors
        for u in range(n):
            if not visited[u]:
                # Get original node IDs to look up distance
                node_u_orig = candidates[u]
                node_v_orig = candidates[v]
                w = distance_matrix[node_u_orig, node_v_orig]
                if w < min_edge_weight[u]:
                    min_edge_weight[u] = w
                    parent[u] = v
                    
    # Now we have the MST edges. 
    # Determine degrees in the MST to find leaves.
    degree = [0] * n
    for u, v, w in mst_edges:
        degree[u] += 1
        degree[v] += 1
        
    # Leaves are nodes with degree 1 (or 0 if n==1, but we handled n<=1)
    # For each leaf, the 'savings' is the weight of its incident MST edge.
    # Why? If we remove a leaf from the MST, the cost of the new MST (on remaining nodes)
    # is exactly old_MST_cost - edge_weight. 
    # The heuristic argues that visiting a leaf early allows us to 'prune' it from the 
    # remaining structure, potentially aligning the tour with the MST structure.
    # Actually, a better proxy for "global skeleton" alignment is:
    # Minimize: d(current, leaf) - alpha * savings
    # Where savings is the edge weight.
    
    leaf_savings = {}
    for i in range(n):
        if degree[i] == 1:
            # Find the weight of the edge connecting this leaf to the rest
            # It's stored in min_edge_weight[i] from the Prim's run? 
            # No, min_edge_weight[i] holds the weight of the edge used to connect i to the tree.
            # If i is a leaf, it was connected by exactly one edge.
            # So min_edge_weight[i] is the correct savings value.
            leaf_savings[i] = min_edge_weight[i]
            
    return leaf_savings

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Time guard: if we're running low on time, fall back to simple nearest neighbor
    if _get_time_remaining() < 1.0:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    candidates = unvisited_nodes
    n = len(candidates)
    
    if n == 0:
        return destination_node
    
    if n == 1:
        return int(candidates[0])
        
    # Compute distances from current to all candidates
    dist_to_candidate = distance_matrix[current_node, candidates]
    
    # Also compute distance from candidates to destination to handle the end-game
    dist_candidate_to_dest = distance_matrix[candidates, destination_node]
    
    if n <= 5:
        # For very small sets, simple 2-step lookahead is reliable and fast
        # Minimize d(current, next) + d(next, dest)
        scores = dist_to_candidate + dist_candidate_to_dest
        return int(candidates[np.argmin(scores)])
        
    # Compute MST leaf savings for the remaining candidates
    # This captures the global structure of the remaining nodes
    leaf_savings_dict = _compute_mst_leaf_savings(candidates, distance_matrix)
    
    # Construct the score for each candidate
    # Score = d(current, candidate) - alpha * savings + beta * d(candidate, dest)
    # 
    # Logic:
    # 1. d(current, candidate): Standard proximity cost.
    # 2. - alpha * savings: We prefer candidates that are leaves of the MST of the remaining nodes.
    #    Removing a leaf from the MST reduces the total connection cost of the remaining graph.
    #    This encourages the tour to "peel off" the periphery of the remaining cluster,
    #    which is consistent with how optimal tours often traverse the convex hull / MST leaves.
    # 3. beta * d(candidate, dest): A bias towards the destination to ensure we don't get
    #    too far away, especially as n decreases.
    
    alpha = 0.5  # Weight for MST savings
    # Beta increases as we get closer to the end to ensure we head home
    beta = max(0.0, (n - 10) / 40.0) * 0.5  # 0 for n<=10, up to 0.5 for n=50
    
    scores = np.zeros(n)
    for i in range(n):
        node_id = candidates[i]
        base_cost = dist_to_candidate[i]
        savings = leaf_savings_dict.get(i, 0.0)
        dest_cost = dist_candidate_to_dest[i]
        
        scores[i] = base_cost - alpha * savings + beta * dest_cost
        
    return int(candidates[np.argmin(scores)])
# EVOLVE-BLOCK-END
