"""Guided local search penalty for TSP; DIRECT REJECTION approach: penalize high-regret edges directly."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_regrets(D, tour):
    """
    For each edge (tour[i], tour[i+1]) in the current tour, compute the best possible
    2-opt improvement if we were to break this edge and reconnect with some other edge.
    This is an approximation of the 'regret' of keeping this edge.
    
    Returns a list of regrets corresponding to each edge in the tour.
    """
    n = len(tour)
    if n < 4:
        return [0.0] * n
    
    # Precompute indices for speed
    # Edge i is (tour[i], tour[(i+1)%n])
    # A 2-opt move breaks edge i and edge j.
    # Improvement = D[tour[i], tour[j+1]] + D[tour[i+1], tour[j]] - D[tour[i], tour[i+1]] - D[tour[j], tour[j+1]]
    # We want to find, for each edge i, the max improvement over all j != i, i-1, i+1.
    
    regrets = np.zeros(n)
    # We can vectorize this partially or just loop. N=100, so N^2 = 10,000 ops is trivial.
    for i in range(n):
        u = tour[i]
        v = tour[(i + 1) % n]
        cost_uv = D[u, v]
        max_improvement = 0.0
        
        for j in range(n):
            # Skip adjacent edges (would result in same tour or trivial move)
            if j == i or j == (i - 1) % n or j == (i + 1) % n:
                continue
            
            w = tour[j]
            x = tour[(j + 1) % n]
            cost_wx = D[w, x]
            
            # Two possible reconnections:
            # 1. u-x and v-w
            imp1 = D[u, x] + D[v, w] - cost_uv - cost_wx
            # 2. u-w and v-x (This is the same as 1 if we swap roles, but let's be precise)
            # Actually, 2-opt reverses the segment.
            # If we reverse segment i+1..j, the new edges are (u, x) and (v, w).
            # So imp1 is the correct formula for the move that breaks i and j.
            
            if imp1 > max_improvement:
                max_improvement = imp1
        
        regrets[i] = max_improvement
    
    return regrets

def update_edge_distance(edge_distance, local_opt_tour, edge_n_used):
    D = np.asarray(edge_distance, dtype=np.float64)
    tour0 = np.asarray(local_opt_tour, dtype=int)
    used = np.asarray(edge_n_used, dtype=np.float64)
    n = D.shape[0]

    t_start = time.time()
    T_LIMIT = 250.0 # Safety margin for 300s limit
    
    # 1. Calculate regrets for all edges in the current tour
    regrets = _compute_regrets(D, tour0)
    
    # 2. Identify the "worst" edges to penalize.
    # We want to penalize edges that have high regret (breaking them yields big gain)
    # OR edges that have been used a lot (classic GLS).
    # Let's combine them. High regret edges are "bad" because they prevent improvement.
    # High usage edges are "bad" because they trap the search.
    
    # Normalize regrets to [0, 1]
    max_regret = np.max(regrets) if np.max(regrets) > 1e-10 else 1.0
    norm_regrets = regrets / max_regret
    
    # Normalize usage counts to [0, 1]
    max_used = np.max(used) if np.max(used) > 1e-10 else 1.0
    norm_used = used / max_used
    
    # Create a penalty matrix
    # We will penalize edges in the current tour.
    # Penalty score for edge i = alpha * norm_regret[i] + beta * norm_used[edge_i]
    
    # Let's try a few combinations of alpha and beta, but pick the best one?
    # No, the directive says avoid micro-tuning. Let's pick a robust heuristic.
    # High regret is a strong signal for 2-opt.
    # Let's penalize the top 5% of edges by regret, and also add a small penalty for usage.
    
    # Threshold for regret: top 10%
    threshold = np.percentile(norm_regrets, 90)
    
    M = D.copy()
    
    # Apply penalties
    for i in range(n):
        u = tour0[i]
        v = tour0[(i + 1) % n]
        
        # Base penalty from regret
        if norm_regrets[i] > threshold:
            # Strong penalty to force 2-opt to break this edge
            penalty = 0.5 * D[u, v] * (norm_regrets[i] + 0.5) # Scale with edge length? No, absolute is better for 2-opt delta?
            # Actually, adding to D changes the delta calculation.
            # delta = D_new[u,x] + D_new[v,w] - D_new[u,v] - D_new[w,x]
            # If we add P to D[u,v], delta decreases by P.
            # We want delta < 0. So we need P > (D[u,v]+D[w,x]) - (D[u,x]+D[v,w]) = -imp1.
            # Since imp1 is positive for improving moves, we just need P to be large enough.
            # Let's add a penalty proportional to the regret itself.
            M[u, v] += norm_regrets[i] * max_regret * 0.5
            M[v, u] += norm_regrets[i] * max_regret * 0.5
        
        # Add small usage penalty to all tour edges to encourage diversity
        # This helps if we are stuck in a cycle of same tours
        usage_pen = 0.1 * norm_used[u, v] * max_regret
        M[u, v] += usage_pen
        M[v, u] += usage_pen

    # Ensure symmetry and finiteness
    M = (M + M.T) * 0.5
    np.fill_diagonal(M, 0.0)
    M = np.where(np.isfinite(M), M, D)
    
    return M
# EVOLVE-BLOCK-END
