import numpy as np
import time
import sys

def _aco_simulate(distance_matrix, heuristic, n_ants=20, n_iterations=30, seed=42):
    """
    Fast internal ACO simulation matching the contract's parameters.
    Used to evaluate heuristic quality during the heuristic construction phase.
    """
    np.random.seed(seed)
    n = distance_matrix.shape[0]
    tau = np.ones((n, n))
    best_tour_len = float('inf')
    
    for it in range(n_iterations):
        tour_lengths = np.zeros(n_ants)
        
        for ant in range(n_ants):
            visited = np.zeros(n, dtype=bool)
            visited[0] = True
            current = 0
            tour_len = 0.0
            tour_path = [0]
            
            for _ in range(n - 1):
                probs = tau[current, :] * heuristic[current, :]
                probs[visited] = 0.0
                
                total_prob = probs.sum()
                if total_prob < 1e-10:
                    unvisited_indices = np.where(~visited)[0]
                    next_node = unvisited_indices[np.random.randint(len(unvisited_indices))]
                else:
                    probs = probs / total_prob
                    next_node = np.random.choice(n, p=probs)
                
                tour_len += distance_matrix[current, next_node]
                visited[next_node] = True
                tour_path.append(next_node)
                current = next_node
            
            tour_len += distance_matrix[current, 0]
            tour_lengths[ant] = tour_len
            
            deposit = 1.0 / tour_len
            for i in range(n):
                u = tour_path[i]
                v = tour_path[(i + 1) % n]
                tau[u, v] += deposit
        
        tau *= 0.9
        
        current_best = np.min(tour_lengths)
        if current_best < best_tour_len:
            best_tour_len = current_best
            
    return best_tour_len

def _two_opt_improve(tour, dist, max_passes=100):
    """
    Apply 2-opt improvement to a tour.
    """
    n = len(tour)
    if n < 4:
        return tour, sum(dist[tour[i], tour[(i+1)%n]] for i in range(n))
    
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                if j == i + 1:
                    continue
                u1 = tour[(i - 1) % n]
                v1 = tour[i]
                u2 = tour[j]
                v2 = tour[(j + 1) % n]
                
                delta = dist[u1, u2] + dist[v1, v2] - dist[u1, v1] - dist[u2, v2]
                if delta < -1e-10:
                    tour[i:j+1] = tour[i:j+1][::-1]
                    improved = True
                    
    total_len = 0.0
    for i in range(n):
        total_len += dist[tour[i], tour[(i + 1) % n]]
    return tour, total_len

def _nearest_neighbor_tour(dist, start=0):
    """
    Generate a tour using Nearest Neighbor heuristic.
    """
    n = len(dist)
    visited = np.zeros(n, dtype=bool)
    visited[start] = True
    current = start
    tour = [start]
    
    for _ in range(n - 1):
        dists = dist[current, :].copy()
        dists[visited] = np.inf
        next_node = np.argmin(dists)
        tour.append(next_node)
        visited[next_node] = True
        current = next_node
        
    return tour

def _random_insertion_tour(dist, seed):
    """
    Generate a tour using Random Insertion heuristic.
    """
    np.random.seed(seed)
    n = len(dist)
    tour = [0, np.random.randint(1, n)]
    unvisited = list(range(1, n))
    unvisited.remove(tour[1])
    np.random.shuffle(unvisited)
    
    for node in unvisited:
        best_pos = 0
        best_cost = float('inf')
        for i in range(len(tour)):
            u = tour[i]
            v = tour[(i + 1) % len(tour)]
            cost = dist[u, node] + dist[node, v] - dist[u, v]
            if cost < best_cost:
                best_cost = cost
                best_pos = i + 1
        tour.insert(best_pos, node)
        
    return tour

def _cheapest_insertion_tour(dist):
    """
    Generate a tour using Cheapest Insertion heuristic.
    """
    n = len(dist)
    tour = [0]
    unvisited = list(range(1, n))
    
    while unvisited:
        best_node = -1
        best_pos = 0
        best_cost = float('inf')
        
        for node in unvisited:
            for i in range(len(tour)):
                u = tour[i]
                v = tour[(i + 1) % len(tour)]
                cost = dist[u, node] + dist[node, v] - dist[u, v]
                if cost < best_cost:
                    best_cost = cost
                    best_node = node
                    best_pos = i + 1
        
        if best_node == -1:
            break
            
        tour.insert(best_pos, best_node)
        unvisited.remove(best_node)
        
    return tour

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin
    
    n = distance_matrix.shape[0]
    if n == 0:
        return np.array([])
        
    safe_dist = distance_matrix.copy()
    np.fill_diagonal(safe_dist, 1.0)
    safe_dist[safe_dist < 1e-9] = 1e-9
    
    # Strategy: Multi-Source Structural Fusion with Enhanced Diversity
    # 1. Generate a diverse set of high-quality tours using different construction heuristics.
    # 2. Improve each with 2-opt.
    # 3. Aggregate edge statistics (frequency and quality) across these tours.
    # 4. Construct a heuristic matrix that rewards edges that are frequently part of good tours.
    # 5. Use ACO simulation to select the best scaling/blend.
    
    # Step 1: Generate diverse tours
    tours = []
    tour_lengths = []
    
    # Base seeds for diversity
    base_seeds = list(range(20))
    
    for seed in base_seeds:
        if time.time() > deadline - 60:
            break
            
        # Cycle through different construction heuristics
        heuristic_type = seed % 3
        
        if heuristic_type == 0:
            # Standard NN
            tour = _nearest_neighbor_tour(safe_dist, start=0)
        elif heuristic_type == 1:
            # Random Insertion
            tour = _random_insertion_tour(safe_dist, seed)
        else:
            # Cheapest Insertion
            tour = _cheapest_insertion_tour(safe_dist)
        
        # Improve with 2-opt
        tour, length = _two_opt_improve(tour, safe_dist)
        tours.append(tour)
        tour_lengths.append(length)
        
    if not tours:
        # Fallback if no time
        return 1.0 / safe_dist

    # Step 2: Aggregate edge statistics
    edge_weight = np.zeros((n, n))
    
    min_len = min(tour_lengths)
    max_len = max(tour_lengths)
    
    for tour, length in zip(tours, tour_lengths):
        # Weight this tour's edges by its quality
        # Use exponential weighting: better tours get significantly higher weight
        if max_len > min_len:
            norm_len = (length - min_len) / (max_len - min_len)
            w = np.exp(-2.0 * norm_len)
        else:
            w = 1.0
        
        for i in range(n):
            u = tour[i]
            v = tour[(i + 1) % n]
            # Symmetric accumulation
            edge_weight[u, v] += w
            edge_weight[v, u] += w

    # Step 3: Construct Heuristic
    base_heur = 1.0 / safe_dist
    
    max_w = np.max(edge_weight)
    if max_w > 0:
        # Scale weights so they have a significant impact
        # Try a few different scaling factors
        scaled_w = edge_weight / max_w
    else:
        scaled_w = np.zeros((n, n))
        
    # Initial candidate
    final_heur = (1.0 + scaled_w * 10.0) * base_heur
    final_heur[~np.isfinite(final_heur)] = 1e-9
    final_heur[final_heur <= 0] = 1e-9
    
    # Step 4: Refine with ACO simulation
    best_heur = final_heur
    best_score = _aco_simulate(safe_dist, final_heur, seed=123)
    
    if time.time() < deadline - 30:
        # Try different scaling factors for the structural weight
        scale_factors = [5.0, 10.0, 20.0, 50.0, 100.0]
        
        for scale_factor in scale_factors:
            if time.time() > deadline - 30:
                break
                
            scaled_w_test = edge_weight / max_w * scale_factor if max_w > 0 else np.zeros((n, n))
            h_test = (1.0 + scaled_w_test) * base_heur
            h_test[~np.isfinite(h_test)] = 1e-9
            h_test[h_test <= 0] = 1e-9
            
            score = _aco_simulate(safe_dist, h_test, seed=123)
            if score < best_score:
                best_score = score
                best_heur = h_test
        
        # Try blending with pure inverse distance
        blend_alphas = [0.3, 0.5, 0.7]
        
        for alpha in blend_alphas:
            if time.time() > deadline - 30:
                break
                
            h_blend = (1.0 - alpha) * base_heur + alpha * best_heur
            h_blend[~np.isfinite(h_blend)] = 1e-9
            h_blend[h_blend <= 0] = 1e-9
            score = _aco_simulate(safe_dist, h_blend, seed=123)
            if score < best_score:
                best_score = score
                best_heur = h_blend
                
    return best_heur
