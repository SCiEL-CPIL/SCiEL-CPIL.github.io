import numpy as np
import time

def _run_aco_sim(dist, eta, n_ants=20, n_iters=30, seed=42):
    """
    Optimized ACO simulation.
    """
    n = len(dist)
    tau = np.ones((n, n), dtype=np.float64)
    rng = np.random.default_rng(seed)
    
    best_len = float('inf')
    
    for _ in range(n_iters):
        tours_list = []
        
        for _ in range(n_ants):
            tour = [0]
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            curr = 0
            
            for _ in range(n - 1):
                # Vectorized probability calculation
                probs = tau[curr, :] * eta[curr, :]
                probs[visited] = 0.0
                
                total_prob = np.sum(probs)
                if total_prob <= 0:
                    unvisited = np.where(~visited)[0]
                    next_node = rng.choice(unvisited)
                else:
                    probs /= total_prob
                    next_node = rng.choice(n, p=probs)
                
                tour.append(next_node)
                visited[next_node] = True
                curr = next_node
            
            tour.append(0)
            tour_arr = np.array(tour, dtype=np.int32)
            
            # Calculate length
            length = np.sum(dist[tour_arr[:-1], tour_arr[1:]])
            
            if length < best_len:
                best_len = length
            
            tours_list.append((length, tour_arr))
        
        # Update Pheromones
        tau *= 0.9
        
        for length, tour_arr in tours_list:
            reward = 1.0 / length
            u = tour_arr[:-1]
            v = tour_arr[1:]
            tau[u, v] += reward

    return best_len

def heuristics(distance_matrix):
    start_time = time.time()
    timeout = 320  # Leave margin for 330s limit
    
    n = len(distance_matrix)
    safe_dist = distance_matrix.copy()
    np.fill_diagonal(safe_dist, 1.0)
    
    best_score = float('inf')
    best_eta = None
    
    # Search for best alpha
    # Pure distance heuristic: eta = 1 / dist^alpha
    alphas = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]
    
    for alpha in alphas:
        if time.time() - start_time > timeout:
            break
        
        eta = 1.0 / (safe_dist ** alpha)
        np.fill_diagonal(eta, 1e-9)
        
        # Evaluate with a few seeds
        score_sum = 0
        valid = 0
        for seed in [1, 2, 3]:
            if time.time() - start_time > timeout:
                break
            s = _run_aco_sim(distance_matrix, eta, n_ants=20, n_iters=30, seed=seed)
            score_sum += s
            valid += 1
        
        if valid > 0:
            avg_score = score_sum / valid
            if avg_score < best_score:
                best_score = avg_score
                best_eta = eta.copy()
    
    if best_eta is None:
        # Fallback
        eta = 1.0 / safe_dist
        np.fill_diagonal(eta, 1e-9)
        best_eta = eta

    return best_eta
