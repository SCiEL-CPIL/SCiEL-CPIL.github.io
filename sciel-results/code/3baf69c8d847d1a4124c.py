"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_mst_heuristic(distance_matrix):
    """
    Compute edge attractiveness based on MST of the complete graph.
    Edges in the MST get high attractiveness; others get lower values.
    This encourages ants to follow the global skeleton of the optimal tour.
    """
    n = distance_matrix.shape[0]
    
    # Kruskal's algorithm for MST
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((distance_matrix[i, j], i, j))
    
    edges.sort(key=lambda x: x[0])
    
    # Union-Find
    parent = list(range(n))
    
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
            return True
        return False
    
    mst_edge_set = set()
    mst_count = 0
    for dist, i, j in edges:
        if union(i, j):
            mst_edge_set.add((i, j))
            mst_edge_set.add((j, i))
            mst_count += 1
            if mst_count == n - 1:
                break
    
    # Create heuristic matrix: MST edges get high value, others get lower
    heuristic = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                heuristic[i, j] = 0.0
            elif (i, j) in mst_edge_set:
                heuristic[i, j] = 1.0 / (distance_matrix[i, j] + 1e-9)
            else:
                # Non-MST edges: use inverse distance but scaled down
                heuristic[i, j] = 0.3 / (distance_matrix[i, j] + 1e-9)
    
    return heuristic


def heuristics(distance_matrix, coordinates, demands, capacity):
    """
    Compute an edge-attractiveness matrix for ant colony optimization.
    Uses MST-based heuristic to encourage ants to follow the global
    skeleton of the optimal tour.
    """
    start_time = time.time()
    deadline = 325.0  # Safety margin before 330s limit
    
    try:
        heuristic = _compute_mst_heuristic(distance_matrix)
    except Exception:
        # Fallback to simple inverse distance
        heuristic = 1.0 / (distance_matrix + 1e-9)
    
    # Ensure all values are positive and finite
    heuristic[~np.isfinite(heuristic)] = 1e-9
    heuristic[heuristic <= 0] = 1e-9
    
    return heuristic
# EVOLVE-BLOCK-END
