import numpy as np
import time
from scipy.spatial import Delaunay

def _recover_points(distance_matrix):
    """
    Recovers 2D coordinates from a distance matrix using Classical MDS.
    """
    n = distance_matrix.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (distance_matrix ** 2) @ J
    
    try:
        eigvals, eigvecs = np.linalg.eigh(B)
    except np.linalg.LinAlgError:
        return None
        
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    eigvals_top2 = np.maximum(eigvals[:2], 0)
    points = eigvecs[:, :2] * np.sqrt(eigvals_top2)
    
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    range_coords = max_coords - min_coords
    range_coords[range_coords < 1e-9] = 1.0
    points = (points - min_coords) / range_coords
    
    return points

def _compute_delaunay_heuristic(distance_matrix):
    """
    Computes a geometry-aware heuristic based on Delaunay triangulation.
    """
    n = distance_matrix.shape[0]
    points = _recover_points(distance_matrix)
    
    if points is None:
        safe_dist = distance_matrix.copy()
        np.fill_diagonal(safe_dist, 1.0)
        safe_dist[safe_dist < 1e-9] = 1e-9
        return 1.0 / safe_dist

    try:
        tri = Delaunay(points)
    except Exception:
        safe_dist = distance_matrix.copy()
        np.fill_diagonal(safe_dist, 1.0)
        safe_dist[safe_dist < 1e-9] = 1e-9
        return 1.0 / safe_dist

    simplices = tri.simplices
    edge_circumradii = {}
    
    def circumradius(p1, p2, p3):
        a = np.linalg.norm(p2 - p3)
        b = np.linalg.norm(p1 - p3)
        c = np.linalg.norm(p1 - p2)
        cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
        area = 0.5 * np.abs(cross)
        
        if area < 1e-9:
            return 100.0
        R = (a * b * c) / (4.0 * area)
        return R

    for simplex in simplices:
        p1 = points[simplex[0]]
        p2 = points[simplex[1]]
        p3 = points[simplex[2]]
        
        R = circumradius(p1, p2, p3)
        
        edges = [(simplex[0], simplex[1]), (simplex[1], simplex[2]), (simplex[2], simplex[0])]
        for u, v in edges:
            key = (min(u, v), max(u, v))
            if key in edge_circumradii:
                edge_circumradii[key].append(R)
            else:
                edge_circumradii[key] = [R]
                
    all_R = [np.mean(r) for r in edge_circumradii.values()]
    if all_R:
        R_global = np.median(all_R)
    else:
        R_global = 1.0
        
    geo_heur = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                geo_heur[i, j] = 1.0
                continue
            d_ij = distance_matrix[i, j]
            key = (min(i, j), max(i, j))
            if key in edge_circumradii:
                R = np.mean(edge_circumradii[key])
            else:
                R = R_global
            if R < 1e-9:
                R = 1e-9
            geo_heur[i, j] = np.exp(-d_ij / R)
            
    return geo_heur

def _fast_aco_simulate(distance_matrix, heuristic, n_ants=20, n_iterations=30, rho=0.9, seed=42):
    """
    Fast internal ACO simulation to evaluate heuristic quality.
    Implements directed pheromone updates as per contract.
    """
    rng = np.random.RandomState(seed)
    n = distance_matrix.shape[0]
    tau = np.ones((n, n))
    best_tour_len = float('inf')
    
    for it in range(n_iterations):
        tour_lengths = []
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
                    next_node = unvisited_indices[rng.randint(len(unvisited_indices))]
                else:
                    probs = probs / total_prob
                    next_node = rng.choice(n, p=probs)
                
                tour_len += distance_matrix[current, next_node]
                visited[next_node] = True
                tour_path.append(next_node)
                current = next_node
            
            tour_len += distance_matrix[current, 0]
            tour_lengths.append(tour_len)
            
            deposit = 1.0 / tour_len
            for i in range(n):
                u = tour_path[i]
                v = tour_path[(i + 1) % n]
                tau[u, v] += deposit
            
        tau *= rho
        
        if tour_lengths:
            current_best = min(tour_lengths)
            if current_best < best_tour_len:
                best_tour_len = current_best
                
    return best_tour_len

def heuristics(distance_matrix):
    start_time = time.time()
    deadline = start_time + 290.0
    
    n = distance_matrix.shape[0]
    
    # Generate geometry-aware heuristic
    base_heur = _compute_delaunay_heuristic(distance_matrix)
    
    # Ensure validity
    base_heur[base_heur <= 0] = 1e-9
    base_heur[~np.isfinite(base_heur)] = 1e-9
    
    def evaluate_beta(beta, seeds=(1, 2, 3)):
        h = base_heur ** beta
        h[h <= 0] = 1e-9
        h[~np.isfinite(h)] = 1e-9
        
        scores = []
        for s in seeds:
            # Check time
            if time.time() > deadline - 5.0:
                return float('inf')
            score = _fast_aco_simulate(distance_matrix, h, n_ants=20, n_iterations=30, seed=s)
            scores.append(score)
        
        return np.mean(scores)
    
    best_beta = 1.0
    best_score = float('inf')
    
    # Coarse search for beta
    betas = np.arange(0.5, 5.5, 0.5)
    
    for beta in betas:
        if time.time() > deadline - 60.0:
            break
        score = evaluate_beta(beta)
        if score < best_score:
            best_score = score
            best_beta = beta
            
    # Fine search
    if time.time() < deadline - 30.0:
        lower = max(0.1, best_beta - 0.5)
        upper = min(10.0, best_beta + 0.5)
        fine_betas = np.linspace(lower, upper, 20)
        
        for beta in fine_betas:
            if time.time() > deadline - 10.0:
                break
            score = evaluate_beta(beta, seeds=(1, 2, 3, 4, 5))
            if score < best_score:
                best_score = score
                best_beta = beta
                
    # Construct final heuristic
    final_heuristic = base_heur ** best_beta
    final_heuristic[final_heuristic <= 0] = 1e-9
    final_heuristic[~np.isfinite(final_heuristic)] = 1e-9
    
    return final_heuristic
