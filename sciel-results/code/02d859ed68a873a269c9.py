"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Cache for MST-based scores to avoid recomputation
_mst_cache = {}
_start_time = time.time()


def _mst_weight(nodes, dm):
    """Compute MST weight of the induced subgraph on `nodes` using Kruskal's."""
    n = len(nodes)
    if n <= 1:
        return 0.0
    # Build edge list
    edges = []
    node_list = list(nodes)
    for i in range(n):
        for j in range(i + 1, n):
            w = dm[node_list[i], node_list[j]]
            edges.append((w, i, j))
    edges.sort(key=lambda x: x[0])
    
    # Union-Find
    parent = list(range(n))
    rank = [0] * n
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1
        return True
    
    total = 0.0
    count = 0
    for w, i, j in edges:
        if union(i, j):
            total += w
            count += 1
            if count == n - 1:
                break
    return total


def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    # Deadline guard
    if time.time() - _start_time > 140:
        # Fallback: nearest neighbor
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    if n_unvisited == 0:
        return current_node
    
    # Compute scores for each candidate
    # Score = alpha * dist_to_current + beta * dist_to_dest + gamma * connectivity_cost
    # connectivity_cost approximates how expensive it is to keep this node in the remaining MST
    
    dists_to_current = distance_matrix[current_node, unvisited_nodes]
    dists_to_dest = distance_matrix[destination_node, unvisited_nodes]
    
    # Connectivity cost: for each node, sum of two smallest distances to other unvisited nodes
    # This is a proxy for MST edge contribution
    sub_dm = distance_matrix[np.ix_(unvisited_nodes, unvisited_nodes)]
    np.fill_diagonal(sub_dm, np.inf)
    
    # For each row, get the two smallest values
    # Use partition for efficiency
    if n_unvisited > 2:
        # Get indices of two smallest per row
        idx1 = np.argpartition(sub_dm, 1, axis=1)[:, :2]
        two_smallest = np.take_along_axis(sub_dm, idx1, axis=1)
        connectivity_cost = np.sum(two_smallest, axis=1)
    else:
        connectivity_cost = np.sum(sub_dm, axis=1)
    
    # Normalize components
    max_dc = np.max(dists_to_current) if np.max(dists_to_current) > 0 else 1.0
    max_dd = np.max(dists_to_dest) if np.max(dists_to_dest) > 0 else 1.0
    max_cc = np.max(connectivity_cost) if np.max(connectivity_cost) > 0 else 1.0
    
    dc_norm = dists_to_current / max_dc
    dd_norm = dists_to_dest / max_dd
    cc_norm = connectivity_cost / max_cc
    
    # Weighted blend: emphasize connectivity (MST-like) while keeping local awareness
    alpha = 0.3  # distance to current
    beta = 0.2   # distance to destination  
    gamma = 0.5  # connectivity cost (MST proxy)
    
    scores = alpha * dc_norm + beta * dd_norm + gamma * cc_norm
    
    return int(unvisited_nodes[np.argmin(scores)])
# EVOLVE-BLOCK-END
