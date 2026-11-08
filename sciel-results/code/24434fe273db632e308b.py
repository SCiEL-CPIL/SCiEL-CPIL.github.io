import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 140.0
    
    n = adjacency.shape[0]
    if n == 0:
        return np.array([])
        
    # Ensure float for operations
    adj = adjacency.astype(float)
    degrees = adj.sum(axis=1)
    
    # Precompute basic features
    # Average neighbor degree
    sum_neighbor_degrees = adj.dot(degrees)
    avg_neighbor_degree = np.zeros(n)
    nonzero_mask = degrees > 0
    avg_neighbor_degree[nonzero_mask] = sum_neighbor_degrees[nonzero_mask] / degrees[nonzero_mask]
    
    # Local edge density (clustering coefficient)
    # Also compute "triangle count" or similar higher-order stats if time permits
    clustering = np.zeros(n)
    triangle_count = np.zeros(n)
    
    # Compute clustering and triangle count
    # For n=200, O(n * d^2) is fast enough. d ~ 5-10 for ER graphs.
    for i in range(n):
        if time.time() > deadline - 20:
            break
        d = int(degrees[i])
        if d > 1:
            neighbors = np.where(adj[i] > 0)[0]
            sub = adj[np.ix_(neighbors, neighbors)]
            edges = sub.sum() / 2.0
            possible = d * (d - 1) / 2.0
            clustering[i] = edges / possible
            triangle_count[i] = edges  # Number of triangles involving i
    
    # Feature: "Waste" - how many edges are among neighbors.
    # Selecting a vertex v removes all its neighbors.
    # If neighbors are well-connected, they would have blocked each other anyway?
    # Actually, for MIS, we want vertices whose neighbors are NOT well connected?
    # Or we want vertices with low degree?
    # Let's stick to the empirical success of avg_nd - deg + w*clus.
    
    # Additional feature: Degree of the degree (second moment)
    # sum of squares of neighbor degrees
    sum_sq_neighbor_degrees = adj.dot(degrees**2)
    avg_sq_neighbor_degree = np.zeros(n)
    avg_sq_neighbor_degree[nonzero_mask] = sum_sq_neighbor_degrees[nonzero_mask] / degrees[nonzero_mask]
    
    # Variance of neighbor degrees
    var_neighbor_degree = np.zeros(n)
    var_neighbor_degree[nonzero_mask] = avg_sq_neighbor_degree[nonzero_mask] - avg_neighbor_degree[nonzero_mask]**2

    # List of scoring functions
    # Based on previous successes: avg_nd - deg + w * clus
    # Let's explore a wider range of w and combinations.
    
    def score_min_degree(adj, deg, avg_nd, clus, tri, var_nd, avg_sq_nd):
        return -deg

    def score_avg_nd_minus_deg(adj, deg, avg_nd, clus, tri, var_nd, avg_sq_nd):
        return avg_nd - deg

    def score_w_clus_05(adj, deg, avg_nd, clus, tri, var_nd, avg_sq_nd):
        return avg_nd - deg + 0.5 * clus

    def score_w_clus_1(adj, deg, avg_nd, clus, tri, var_nd, avg_sq_nd):
        return avg_nd - deg + 1.0 * clus

    def score_w_clus_2(adj, deg, avg_nd, clus, tri, var_nd, avg_sq_nd):
        return avg_nd - deg + 2.0 * clus

    def score_w_clus_5(adj, deg, avg_nd, clus, tri, var_nd, avg_sq_nd):
        return avg_nd - deg + 5.0 * clus

    def score_w_clus_10(adj, deg, avg_nd, clus, tri, var_nd, avg_sq_nd):
        return avg_nd - deg + 10.0 * clus

    def score_w_clus_neg1(adj, deg, avg_nd, clus, tri, var_nd, avg_sq_nd):
        return avg_nd - deg - 1.0 * clus

    def score_w_clus_neg5(adj, deg, avg_nd, clus, tri, var_nd, avg_sq_nd):
        return avg_nd - deg - 5.0 * clus

    def score_tri_1(adj, deg, avg_nd, clus, tri, var_nd, avg_sq_nd):
        return avg_nd - deg + 1.0 * tri

    def score_tri_01(adj, deg, avg_nd, clus, tri, var_nd, avg_sq_nd):
        return avg_nd - deg + 0.1 * tri

    def score_tri_neg1(adj, deg, avg_nd, clus, tri, var_nd, avg_sq_nd):
        return avg_nd - deg - 1.0 * tri

    def score_var_nd_1(adj, deg, avg_nd, clus, tri, var_nd, avg_sq_nd):
        return avg_nd - deg + 1.0 * var_nd

    def score_var_nd_neg1(adj, deg, avg_nd, clus, tri, var_nd, avg_sq_nd):
        return avg_nd - deg - 1.0 * var_nd

    def score_avg_sq_nd_01(adj, deg, avg_nd, clus, tri, var_nd, avg_sq_nd):
        return avg_nd - deg + 0.1 * avg_sq_nd

    def score_avg_sq_nd_neg01(adj, deg, avg_nd, clus, tri, var_nd, avg_sq_nd):
        return avg_nd - deg - 0.1 * avg_sq_nd

    def score_ratio(adj, deg, avg_nd, clus, tri, var_nd, avg_sq_nd):
        safe_deg = deg + 1.0
        return avg_nd / safe_deg

    def score_inv_deg(adj, deg, avg_nd, clus, tri, var_nd, avg_sq_nd):
        return 1.0 / (deg + 1.0)
        
    def score_avg_nd_only(adj, deg, avg_nd, clus, tri, var_nd, avg_sq_nd):
        return avg_nd

    def score_mix_05(adj, deg, avg_nd, clus, tri, var_nd, avg_sq_nd):
        return -0.5 * deg + 0.5 * avg_nd

    def score_mix_2_1(adj, deg, avg_nd, clus, tri, var_nd, avg_sq_nd):
        return -2.0 * deg + 1.0 * avg_nd

    def score_mix_1_2(adj, deg, avg_nd, clus, tri, var_nd, avg_sq_nd):
        return -1.0 * deg + 2.0 * avg_nd

    scorers = [
        score_min_degree,
        score_avg_nd_minus_deg,
        score_w_clus_05,
        score_w_clus_1,
        score_w_clus_2,
        score_w_clus_5,
        score_w_clus_10,
        score_w_clus_neg1,
        score_w_clus_neg5,
        score_tri_1,
        score_tri_01,
        score_tri_neg1,
        score_var_nd_1,
        score_var_nd_neg1,
        score_avg_sq_nd_01,
        score_avg_sq_nd_neg01,
        score_ratio,
        score_inv_deg,
        score_avg_nd_only,
        score_mix_05,
        score_mix_2_1,
        score_mix_1_2
    ]
    
    best_score_val = -1
    best_scores = None
    
    # Helper to evaluate a scoring function
    def evaluate_scorer(scorer):
        scores = scorer(adj, degrees, avg_neighbor_degree, clustering, triangle_count, var_neighbor_degree, avg_sq_neighbor_degree)
        
        # Simulate greedy decoder
        order = np.argsort(-scores)
        
        selected = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if selected[v]:
                continue
            selected[v] = True
            count += 1
            # Mark neighbors as selected (blocked)
            # Using boolean indexing is fast
            selected[adj[v] > 0] = True
            
        return count, scores

    for scorer in scorers:
        if time.time() > deadline - 5:
            break
            
        count, scores = evaluate_scorer(scorer)
        
        if count > best_score_val:
            best_score_val = count
            best_scores = scores

    if best_scores is None:
        return -degrees

    return best_scores
