"""Rewritten program for Maximum Independent Set greedy priority scoring."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    # Ensure adjacency is a numpy array
    adj = np.asarray(adjacency, dtype=np.int8)
    n = adj.shape[0]

    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    # Compute degree
    degree = adj.sum(axis=1).astype(float)

    # Compute conflict radius: number of edges among neighbors of each vertex
    # For vertex i, this is the number of edges in the subgraph induced by N(i)
    # A[i,j] = 1 if i and j are adjacent
    # For vertex i, neighbors are N(i) = {j : A[i,j] = 1}
    # Edges among N(i) = sum over pairs (j,k) in N(i) of A[j,k] / 2
    # More efficiently: for each i, count = (A[i] @ A) @ A[i] / 2? No.
    # Let's think: for vertex i, the number of edges among its neighbors is:
    # sum_{j in N(i)} sum_{k in N(i), k > j} A[j,k]
    # = (1/2) * sum_{j in N(i)} (A[j] @ A[i])  ... not quite
    # Actually: for each neighbor j of i, count edges from j to other neighbors of i
    # = for each j in N(i): number of k in N(i) such that A[j,k]=1 and k != i
    # Since A[i,j]=1, we need A[j,k]=1 for k in N(i), k != i
    # This is (A[j] @ A[i]) - A[j,i] = (A[j] @ A[i]) - 1 (since A[j,i]=1)
    # So total edges among N(i) = (1/2) * sum_{j in N(i)} ((A[j] @ A[i]) - 1)
    # = (1/2) * (sum_{j in N(i)} (A[j] @ A[i]) - degree[i])
    # sum_{j in N(i)} (A[j] @ A[i]) = (A @ (A @ A[i]))[i]? No.
    # Let's use matrix form: Let B = A @ A. Then B[i,j] = number of common neighbors of i and j.
    # For vertex i, the number of edges among N(i) is:
    # For each pair (j,k) in N(i) with j<k, A[j,k] = 1
    # = (1/2) * sum_{j in N(i)} sum_{k in N(i)} A[j,k] - (1/2)*degree[i]  [subtract self-loops, but there are none]
    # Wait, sum_{j in N(i)} sum_{k in N(i)} A[j,k] counts each edge twice (once as (j,k) and once as (k,j))
    # So edges among N(i) = (1/2) * sum_{j in N(i)} sum_{k in N(i)} A[j,k]
    # = (1/2) * (A[i] @ (A @ A[i]))? No.
    # Let's define: for each i, let v_i = A[i] (row i of A, indicating neighbors)
    # Then sum_{j in N(i)} sum_{k in N(i)} A[j,k] = v_i @ (A @ v_i)
    # Because (A @ v_i)[j] = sum_k A[j,k] * v_i[k] = sum_{k in N(i)} A[j,k]
    # So v_i @ (A @ v_i) = sum_j v_i[j] * (A @ v_i)[j] = sum_{j in N(i)} sum_{k in N(i)} A[j,k]
    # So conflict_radius[i] = (1/2) * v_i @ (A @ A @ v_i)
    # In matrix form: conflict_radius = 0.5 * (A @ (A @ (A * 1)))? No.
    # Let's compute: AA = A @ A, then for each i: conflict[i] = 0.5 * A[i] @ AA[i]
    # In vectorized form: conflict = 0.5 * np.einsum('ij,ij->i', A, A @ A)
    
    try:
        if time.time() < deadline:
            AA = adj @ adj
            conflict = 0.5 * (adj * AA).sum(axis=1).astype(float)
        else:
            conflict = degree.copy()
    except Exception:
        conflict = degree.copy()

    # Additional features
    # 1. Degree: lower is better for picking
    # 2. Conflict radius: lower is better (fewer edges among neighbors means less interference)
    # 3. Neighbor degree sum: higher is better (removes more "capacity" from neighbors)
    neighbor_degree_sum = adj @ degree

    # 4. Second-order: number of distance-2 neighbors (A^2 diagonal gives common neighbors, but we want unique distance-2)
    # Actually, A^2[i,i] = degree[i], not useful. 
    # Let's use: number of vertices at distance exactly 2
    # This is harder. Let's skip and use other features.
    
    # 5. Local clustering coefficient approximation
    # For vertex i, clustering = edges among N(i) / (degree[i] * (degree[i]-1) / 2)
    # We already have edges among N(i) = conflict[i]
    # So clustering[i] = conflict[i] / (degree[i] * (degree[i]-1) / 2) for degree[i] >= 2
    with np.errstate(divide='ignore', invalid='ignore'):
        max_possible_edges = degree * (degree - 1) / 2.0
        clustering = np.where(max_possible_edges > 0, conflict / max_possible_edges, 0.0)

    # Normalize all features to [0, 1]
    def normalize(x):
        x = np.asarray(x, dtype=float)
        x_min = x.min()
        x_max = x.max()
        if x_max > x_min:
            return (x - x_min) / (x_max - x_min)
        return np.zeros_like(x)

    degree_norm = normalize(degree)
    conflict_norm = normalize(conflict)
    nds_norm = normalize(neighbor_degree_sum)
    clustering_norm = normalize(clustering)

    # Composite score:
    # We want to pick vertices that:
    # 1. Have low degree (negative weight on degree)
    # 2. Have low conflict radius (negative weight on conflict) - this is the key new feature
    # 3. Have high neighbor degree sum (positive weight) - removes high-degree neighbors
    # 4. Have low clustering (negative weight) - neighbors are not densely connected
    
    # The key insight: picking a vertex with low conflict radius means its neighbors
    # are not well-connected to each other, so removing the vertex doesn't "waste"
    # many potential independent set members.
    
    w_deg = 1.0
    w_conflict = 1.5  # High weight on conflict radius - this is the main structural change
    w_nds = 0.8
    w_clustering = 0.5

    score = (-w_deg * degree_norm 
             - w_conflict * conflict_norm 
             + w_nds * nds_norm 
             - w_clustering * clustering_norm)

    return score
# EVOLVE-BLOCK-END
