import numpy as np
import time
from scipy.sparse.csgraph import minimum_spanning_tree

def _simulate_aco(d, eta, n_ants=20, n_iterations=30):
    """
    Simulate ACO to estimate the quality of a heuristic.
    """
    n = d.shape[0]
    tau = np.ones((n, n))
    best_len = float('inf')
    
    for _ in range(n_iterations):
        all_tours = []
        all_lens = []
        
        for _ in range(n_ants):
            visited = [False] * n
            current = 0
            visited[0] = True
            tour_len = 0.0
            tour = [0]
            
            for _ in range(n - 1):
                # Calculate probabilities
                # prob[j] = tau[current, j] * eta[current, j]
                # Mask visited and self
                probs = tau[current, :] * eta[current, :]
                probs[visited] = 0.0
                probs[current] = 0.0
                
                total = np.sum(probs)
                if total == 0:
                    # Fallback to random unvisited
                    unvisited = [j for j in range(n) if not visited[j]]
                    next_node = np.random.choice(unvisited)
                else:
                    probs = probs / total
                    # Sample
                    next_node = np.random.choice(n, p=probs)
                
                tour_len += d[current, next_node]
                tour.append(next_node)
                current = next_node
                visited[current] = True
            
            # Close tour
            tour_len += d[current, 0]
            all_tours.append(tour)
            all_lens.append(tour_len)
            
            if tour_len < best_len:
                best_len = tour_len
        
        # Update pheromones
        tau *= 0.9
        for i, tour in enumerate(all_tours):
            L = all_lens[i]
            if L == 0: continue
            delta = 1.0 / L
            for k in range(len(tour)):
                u = tour[k]
                v = tour[(k + 1) % len(tour)]
                tau[u, v] += delta
                
    return best_len

def heuristics(distance_matrix):
    d = np.asarray(distance_matrix, dtype=float)
    n = d.shape[0]
    
    # Compute MST
    # d has 0 on diagonal. minimum_spanning_tree handles this.
    mst_sparse = minimum_spanning_tree(d)
    mst_dense = mst_sparse.toarray()
    # Symmetrize mask
    mst_mask = (mst_dense > 0) | (mst_dense.T > 0)
    
    start_time = time.time()
    time_limit = 280.0 # Safety margin
    
    best_eta = None
    best_score = float('inf')
    best_p = 1.0
    best_gamma = 0.0
    
    # Coarse search
    p_vals = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    gamma_vals = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    # Precompute base_eta for each p to avoid recalculating d**p
    base_etas = {}
    for p in p_vals:
        d_safe = d.copy()
        np.fill_diagonal(d_safe, 1.0)
        base_etas[p] = 1.0 / (d_safe ** p)
        
    for p in p_vals:
        for gamma in gamma_vals:
            if time.time() - start_time > time_limit:
                break
                
            base = base_etas[p]
            boost_factor = 1.0 + gamma * mst_mask
            eta = base * boost_factor
            np.fill_diagonal(eta, 1e-9)
            
            score = _simulate_aco(d, eta)
            
            if score < best_score:
                best_score = score
                best_eta = eta.copy()
                best_p = p
                best_gamma = gamma
        
        if time.time() - start_time > time_limit:
            break

    # Fine search around best
    if time.time() - start_time < time_limit - 60: # Leave 60s buffer
        p_range = np.arange(max(0.1, best_p - 0.5), best_p + 0.5 + 0.1, 0.2)
        gamma_range = np.arange(max(0.0, best_gamma - 2.0), best_gamma + 2.0 + 0.1, 1.0)
        
        # Recompute base_etas for new p values if needed, or just compute on fly
        # Since p changes, we need new base_etas
        for p in p_range:
            if time.time() - start_time > time_limit - 30:
                break
            d_safe = d.copy()
            np.fill_diagonal(d_safe, 1.0)
            base = 1.0 / (d_safe ** p)
            
            for gamma in gamma_range:
                if time.time() - start_time > time_limit - 30:
                    break
                boost_factor = 1.0 + gamma * mst_mask
                eta = base * boost_factor
                np.fill_diagonal(eta, 1e-9)
                
                score = _simulate_aco(d, eta)
                
                if score < best_score:
                    best_score = score
                    best_eta = eta.copy()
                    best_p = p
                    best_gamma = gamma

    if best_eta is None:
        # Default fallback
        d_safe = d.copy()
        np.fill_diagonal(d_safe, 1.0)
        best_eta = 1.0 / d_safe
        np.fill_diagonal(best_eta, 1e-9)
        
    return best_eta
