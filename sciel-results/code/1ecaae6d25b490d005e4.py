import numpy as np
import time

_start_time = time.time()
_LIMIT = 115.0

# State for beam search or just greedy
# We will use the MST heuristic greedy approach.
# No complex state needed, just the current call parameters.

def _mst_weight(nodes, distance_matrix):
    """
    Calculate the weight of the Minimum Spanning Tree for the given set of nodes.
    Uses Prim's algorithm.
    nodes: list of node indices
    distance_matrix: 2D numpy array
    """
    if len(nodes) <= 1:
        return 0.0
    
    n = len(nodes)
    # Map node indices to 0..n-1 for faster indexing?
    # Or just use the indices directly.
    
    # Prim's
    in_mst = [False] * len(nodes)
    min_edge = [np.inf] * len(nodes)
    
    # Start with first node
    start_node = nodes[0]
    min_edge[0] = 0.0
    in_mst[0] = True
    
    total_weight = 0.0
    
    for _ in range(n - 1):
        # Find the node with the smallest edge weight not in MST
        min_val = np.inf
        min_idx = -1
        for i in range(n):
            if not in_mst[i] and min_edge[i] < min_val:
                min_val = min_edge[i]
                min_idx = i
        
        if min_idx == -1:
            # Disconnected? Should not happen with complete graph
            break
            
        in_mst[min_idx] = True
        total_weight += min_val
        
        # Update min_edge for neighbors
        node = nodes[min_idx]
        for j in range(n):
            if not in_mst[j]:
                neighbor = nodes[j]
                dist = distance_matrix[node, neighbor]
                if dist < min_edge[j]:
                    min_edge[j] = dist
                    
    return total_weight

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _start_time, _LIMIT
    
    if (time.time() - _start_time) > _LIMIT:
        # Fallback: Nearest neighbor
        if len(unvisited_nodes) == 0:
            return -1
        if len(unvisited_nodes) == 1:
            return int(unvisited_nodes[0])
        
        dists = distance_matrix[current_node, unvisited_nodes]
        return int(unvisited_nodes[np.argmin(dists)])
        
    n_unvisited = len(unvisited_nodes)
    
    if n_unvisited == 0:
        return -1
        
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
        
    if n_unvisited == 2:
        # Pick the one that minimizes d(cur, c) + d(c, dest)
        # Actually, we must visit both.
        # Path: cur -> c1 -> c2 -> dest OR cur -> c2 -> c1 -> dest
        # Since we pick one now, the other is next.
        # Score(c) = d(cur, c) + d(c, other) + d(other, dest)
        # This is exact for 2 nodes.
        c1 = int(unvisited_nodes[0])
        c2 = int(unvisited_nodes[1])
        
        cost1 = distance_matrix[current_node, c1] + distance_matrix[c1, c2] + distance_matrix[c2, destination_node]
        cost2 = distance_matrix[current_node, c2] + distance_matrix[c2, c1] + distance_matrix[c1, destination_node]
        
        if cost1 <= cost2:
            return c1
        else:
            return c2

    # General case: MST Heuristic
    unvisited_arr = np.asarray(unvisited_nodes)
    best_score = np.inf
    best_node = -1
    
    # Pre-compute distances from current to all unvisited
    d_cur = distance_matrix[current_node, unvisited_arr]
    
    for i, c in enumerate(unvisited_arr):
        c = int(c)
        
        # Remaining nodes after picking c
        # We need MST of (unvisited \ {c}) U {dest}
        
        # Create list of nodes for MST
        # unvisited_arr has shape (M,)
        # We want to exclude index i
        
        if i == 0:
            rem_nodes = unvisited_arr[1:]
        elif i == len(unvisited_arr) - 1:
            rem_nodes = unvisited_arr[:-1]
        else:
            rem_nodes = np.concatenate([unvisited_arr[:i], unvisited_arr[i+1:]])
        
        # Add destination
        mst_nodes = np.concatenate([rem_nodes, [destination_node]])
        
        # Calculate MST weight
        mst_w = _mst_weight(mst_nodes, distance_matrix)
        
        score = d_cur[i] + mst_w
        
        if score < best_score:
            best_score = score
            best_node = c
            
    return best_node
