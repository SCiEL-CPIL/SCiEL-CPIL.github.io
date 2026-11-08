"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    deadline = start_time + 143.0  # 7s safety margin for 150s limit

    adj = np.asarray(adjacency, dtype=np.float64)
    n = adj.shape[0]
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])

    # Compute local structural features
    degrees = adj.sum(axis=1)

    # Average neighbor degree
    avg_neigh_deg = np.zeros(n)
    mask = degrees > 0
    avg_neigh_deg[mask] = (adj @ degrees)[mask] / degrees[mask]

    # Clustering coefficient
    try:
        A2 = adj @ adj
        A3_diag = np.sum(adj * A2, axis=1)
        num_triangles = A3_diag / 2.0
        clustering = np.zeros(n)
        denom = degrees * (degrees - 1)
        mask_c = denom > 0
        clustering[mask_c] = num_triangles[mask_c] / denom[mask_c]
    except Exception:
        clustering = np.zeros(n)

    # PageRank-like global score (captures long-range centrality)
    # For MIS, we want to pick vertices that are "peripheral" (low centrality)
    # so PageRank score should be used inversely
    pr_damping = 0.85
    pr = np.ones(n) / n
    for _ in range(50):
        if time.time() > deadline - 10.0:
            break
        new_pr = (1 - pr_damping) / n
        # For undirected graph: contribution from neighbor j to i is pr[j] / deg[j]
        # We can compute: sum over neighbors j of pr[j] / deg[j] for each i
        # This is adj @ (pr / degrees) but need to handle zero degrees
        safe_deg = np.where(degrees > 0, degrees, 1.0)
        pr_contrib = pr / safe_deg
        new_pr += pr_damping * (adj @ pr_contrib)
        pr = new_pr
    # Invert PageRank: low centrality is better for MIS
    pr_inv = 1.0 - pr / (pr.max() + 1e-10)

    # Second-order features: 
    # - Degree of neighbors' neighbors (how "expensive" is the neighborhood?)
    # - Eigenvector centrality (similar to PageRank but power iteration on A)
    try:
        ev = np.ones(n) / n
        for _ in range(30):
            if time.time() > deadline - 10.0:
                break
            ev = adj @ ev
            ev /= (ev.max() + 1e-10)
        ev_inv = 1.0 - ev
    except Exception:
        ev_inv = np.ones(n)

    def normalize(f):
        mn, mx = np.min(f), np.max(f)
        if mx > mn:
            return (f - mn) / (mx - mn)
        return f

    deg_norm = normalize(degrees)
    avg_deg_norm = normalize(avg_neigh_deg)
    clus_norm = normalize(clustering)
    pr_inv_norm = normalize(pr_inv)
    ev_inv_norm = normalize(ev_inv)

    # Build feature matrix
    X = np.column_stack([
        deg_norm,       # 0: degree (lower is better)
        avg_deg_norm,   # 1: avg neighbor degree
        clus_norm,      # 2: clustering coefficient
        pr_inv_norm,    # 3: inverted PageRank (higher = more peripheral)
        ev_inv_norm     # 4: inverted eigenvector centrality
    ])
    n_features = X.shape[1]

    adj_bool = adj > 0.5

    # Fast greedy evaluation
    def greedy_size(weights):
        scores = X @ weights
        order = np.argsort(-scores)
        selected = np.zeros(n, dtype=bool)
        count = 0
        for v in order:
            if not selected[v]:
                count += 1
                selected[v] = True
                selected[adj_bool[v]] = True
        return count

    # Initial heuristic: low degree, high peripheral-ness
    best_w = np.array([-1.0, -0.5, -0.3, 0.8, 0.7])
    best_score = greedy_size(best_w)
    best_w_list = [best_w.copy()]

    # Multiple restarts with different initializations
    np.random.seed(42)
    
    # Simulated annealing with multiple restarts
    n_restarts = 5
    max_steps_per_restart = 3000
    
    current_w = best_w.copy()
    current_score = best_score
    T = 0.1
    T_min = 0.0005
    cooling = 0.995
    
    for restart in range(n_restarts):
        if time.time() > deadline - 15.0:
            break
        
        if restart > 0:
            # New random initialization
            current_w = np.random.uniform(-2, 2, size=n_features)
            current_score = greedy_size(current_w)
            T = 0.15
        
        for step in range(max_steps_per_restart):
            if time.time() > deadline - 10.0:
                break
            
            noise = np.random.normal(0, T, size=n_features)
            new_w = current_w + noise
            new_score = greedy_size(new_w)
            
            delta = new_score - current_score
            if delta > 0 or (T > T_min and np.random.random() < np.exp(delta / max(T, 1e-10))):
                current_w = new_w
                current_score = new_score
                if current_score > best_score:
                    best_score = current_score
                    best_w = current_w.copy()
                    if best_w not in best_w_list:
                        best_w_list.append(best_w.copy())
            
            T = max(T_min, T * cooling)
    
    # Fine-tune around the best solution found
    T = 0.02
    for step in range(1500):
        if time.time() > deadline - 5.0:
            break
        noise = np.random.normal(0, T, size=n_features)
        new_w = best_w + noise
        new_score = greedy_size(new_w)
        if new_score > best_score:
            best_score = new_score
            best_w = new_w.copy()
        T = max(0.001, T * 0.995)
    
    # Also try a few alternative feature subsets in case some features hurt
    # Try with just degree + PageRank
    for alt_w in [
        np.array([-1.0, 0.0, 0.0, 1.0, 0.0]),
        np.array([-1.0, 0.0, 0.0, 0.0, 1.0]),
        np.array([-0.5, -0.3, 0.0, 0.5, 0.3]),
    ]:
        if time.time() > deadline - 3.0:
            break
        alt_score = greedy_size(alt_w)
        if alt_score > best_score:
            best_score = alt_score
            best_w = alt_w.copy()
    
    final_scores = X @ best_w
    return final_scores.astype(float)
# EVOLVE-BLOCK-END
