"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit
    
    # Convert to boolean matrix for efficient operations
    adj = np.asarray(adjacency, dtype=bool)
    n = adj.shape[0]
    
    if n == 0:
        return np.array([])
    
    # Compute degrees
    degrees = adj.sum(axis=1).astype(float)
    
    # Compute average neighbor degree
    avg_neigh_deg = np.zeros(n)
    mask = degrees > 0
    adj_float = adj.astype(np.float64)
    avg_neigh_deg[mask] = (adj_float @ degrees)[mask] / degrees[mask]
    
    # Compute clustering coefficient
    try:
        A2 = adj_float @ adj_float
        A3_diag = np.sum(adj_float * A2, axis=1)
        num_triangles = A3_diag / 2.0
        clustering = np.zeros(n)
        denom = degrees * (degrees - 1)
        mask_c = denom > 0
        clustering[mask_c] = num_triangles[mask_c] / denom[mask_c]
    except Exception:
        clustering = np.zeros(n)
    
    # Normalize features
    def normalize(f):
        mn, mx = np.min(f), np.max(f)
        if mx > mn:
            return (f - mn) / (mx - mn)
        return f
    
    deg_norm = normalize(degrees)
    avg_deg_norm = normalize(avg_neigh_deg)
    clus_norm = normalize(clustering)
    
    # Feature matrix
    X = np.column_stack([deg_norm, avg_deg_norm, clus_norm])
    
    # Fast greedy MIS evaluation
    def greedy_size_v2(weights):
        scores = X @ weights
        order = np.argsort(-scores)
        selected = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v]:
                count += 1
                selected[v] = True
                selected[adj[v]] = True
        return count
    
    # Portfolio of structural heuristics
    # Each heuristic provides a different ordering
    # We'll use a deterministic portfolio approach:
    # 1. Low degree first
    # 2. Low average neighbor degree first
    # 3. Low clustering first
    # 4. Combined score
    
    # Define several candidate orderings
    orderings = []
    
    # Ordering 1: Low degree
    orderings.append(np.argsort(degrees))
    
    # Ordering 2: Low average neighbor degree
    orderings.append(np.argsort(avg_neigh_deg))
    
    # Ordering 3: Low clustering
    orderings.append(np.argsort(clustering))
    
    # Ordering 4: Combined low degree + low clustering
    combined1 = 0.5 * deg_norm + 0.5 * clus_norm
    orderings.append(np.argsort(combined1))
    
    # Ordering 5: Combined low degree + low avg neigh deg
    combined2 = 0.5 * deg_norm + 0.5 * avg_deg_norm
    orderings.append(np.argsort(combined2))
    
    # Ordering 6: All three combined
    combined3 = (deg_norm + avg_deg_norm + clus_norm) / 3.0
    orderings.append(np.argsort(combined3))
    
    # Ordering 7: Inverse - high degree first (sometimes works)
    orderings.append(np.argsort(-degrees))
    
    # Ordering 8: High clustering first
    orderings.append(np.argsort(-clustering))
    
    # Evaluate each ordering
    best_order = None
    best_size = 0
    
    for order in orderings:
        selected = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v]:
                count += 1
                selected[v] = True
                selected[adj[v]] = True
        if count > best_size:
            best_size = count
            best_order = order
    
    # Now, use local search to improve the best ordering
    # We'll do hill climbing on the score vector, but initialize from the best ordering
    # Convert best_order to initial scores
    initial_scores = np.zeros(n)
    for rank, v in enumerate(best_order):
        initial_scores[v] = n - rank
    
    # Normalize to [0, 1]
    initial_scores = (initial_scores - np.min(initial_scores)) / (np.max(initial_scores) - np.min(initial_scores) + 1e-10)
    
    # Use the feature-based scoring to refine
    # Find weights that approximate the best_order
    # We'll do a simple local search on weights
    
    # Initialize weights with a reasonable guess
    # Since low degree is generally good, start with negative weight for degree
    current_w = np.array([-1.0, -0.5, -0.3])
    current_score = greedy_size_v2(current_w)
    
    best_w = current_w.copy()
    best_score = current_score
    
    # Hill climbing with random restarts
    T = 0.1
    max_steps = 3000
    
    for step in range(max_steps):
        if time.time() > deadline - 5.0:
            break
        
        # Perturb weights
        noise = np.random.normal(0, T, size=3)
        new_w = current_w + noise
        
        # Evaluate
        new_score = greedy_size_v2(new_w)
        
        # Accept if better or with some probability (simulated annealing)
        delta = new_score - current_score
        if delta > 0 or np.random.random() < np.exp(delta / T):
            current_w = new_w
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_w = current_w.copy()
        
        T = max(0.001, T * 0.995)
    
    # Final polish
    T = 0.01
    for step in range(1000):
        if time.time() > deadline - 2.0:
            break
        noise = np.random.normal(0, T, size=3)
        new_w = best_w + noise
        new_score = greedy_size_v2(new_w)
        if new_score > best_score:
            best_score = new_score
            best_w = new_w.copy()
        T = max(0.001, T * 0.99)
    
    # Compute final scores using best weights
    final_scores = X @ best_w
    
    return final_scores.astype(float)
# EVOLVE-BLOCK-END
