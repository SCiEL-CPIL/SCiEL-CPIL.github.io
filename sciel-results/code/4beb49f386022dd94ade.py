import numpy as np
import time
from scipy.spatial import Delaunay

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
            # Vectorized pheromone update for the closed tour
            u = np.array(tour_path)
            v = np.array(tour_path[1:] + [tour_path[0]])
            tau[u, v] += deposit
            
        tau *= rho
        
        if tour_lengths:
            current_best = min(tour_lengths)
            if current_best < best_tour_len:
                best_tour_len = current_best
                
    return best_tour_len

def _compute_delaunay_circumradii(distance_matrix):
    """
    Compute Delaunay triangulation and return a matrix of local circumradii for each edge.
    For edges in the Delaunay triangulation, the value is the circumradius of the triangle.
    For non-Delaunay edges, the value is infinity.
    """
    n = distance_matrix.shape[0]
    # Recover coordinates from distance matrix (up to rotation/translation)
    # We can use MDS or just assume the distance matrix comes from points.
    # Since we only have the distance matrix, we need to embed it.
    # However, for 50 points in 2D, we can use classical MDS.
    # Or, simpler: the task says "50 uniform random points in the unit square".
    # The distance matrix is derived from these points.
    # We can recover the points using MDS.
    
    # Classical MDS
    D = distance_matrix
    D_sq = D ** 2
    # Double centering
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D_sq @ J
    
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(B)
    # Sort in descending order
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Take top 2 components
    coords = eigenvectors[:, :2] * np.sqrt(np.maximum(eigenvalues[:2], 0))
    
    # Perform Delaunay triangulation
    try:
        tri = Delaunay(coords)
    except Exception:
        # Fallback if triangulation fails (e.g., degenerate points)
        return np.full((n, n), np.inf), coords
        
    circumradii = np.full((n, n), np.inf)
    
    for simplex in tri.simplices:
        p1 = coords[simplex[0]]
        p2 = coords[simplex[1]]
        p3 = coords[simplex[2]]
        
        # Calculate circumradius
        a = np.linalg.norm(p2 - p3)
        b = np.linalg.norm(p1 - p3)
        c = np.linalg.norm(p1 - p2)
        
        # Area using Heron's formula
        s = (a + b + c) / 2.0
        area_sq = s * (s - a) * (s - b) * (s - c)
        if area_sq <= 0:
            continue
        area = np.sqrt(area_sq)
        
        if area < 1e-10:
            continue
            
        R = (a * b * c) / (4.0 * area)
        
        # Update circumradii for the edges of this triangle
        # The circumradius is the same for all 3 edges of this triangle.
        # If an edge is shared by multiple triangles, we might want the minimum or average.
        # Minimum circumradius implies "tighter" local fit, which might be better.
        # Let's take the minimum circumradius for each edge (most local density).
        
        edges = [(simplex[0], simplex[1]), (simplex[1], simplex[2]), (simplex[2], simplex[0])]
        for u, v in edges:
            if R < circumradii[u, v]:
                circumradii[u, v] = R
            if R < circumradii[v, u]:
                circumradii[v, u] = R
                
    return circumradii, coords

def _compute_mst_edges(distance_matrix):
    """
    Compute the Minimum Spanning Tree edges using Prim's algorithm.
    Returns a boolean matrix where True indicates an MST edge.
    """
    n = distance_matrix.shape[0]
    mst_adj = np.zeros((n, n), dtype=bool)
    in_mst = np.zeros(n, dtype=bool)
    in_mst[0] = True
    min_dist = distance_matrix[0].copy()
    min_dist[0] = np.inf
    
    for _ in range(n - 1):
        # Find the vertex with the minimum distance to the MST
        # np.argmin returns the first occurrence of the minimum value
        u = np.argmin(min_dist)
        # If min_dist is inf, we have a disconnected graph (shouldn't happen for TSP)
        if min_dist[u] == np.inf:
            break
            
        in_mst[u] = True
        # Update min_dist for neighbors
        # We need to find which vertex in the MST is connected to u with min_dist[u]
        # Actually, min_dist[u] stores the min distance from u to ANY vertex in MST.
        # To find the specific parent, we need to search.
        # However, for just marking edges, we can just record that u is added.
        # But we need the edge (parent, u).
        
        # Let's find the parent
        parent = -1
        for v in range(n):
            if in_mst[v] and v != u:
                if distance_matrix[v, u] == min_dist[u]:
                    parent = v
                    break
        if parent != -1:
            mst_adj[parent, u] = True
            mst_adj[u, parent] = True
        
        # Update min_dist for all non-MST vertices
        for v in range(n):
            if not in_mst[v]:
                if distance_matrix[u, v] < min_dist[v]:
                    min_dist[v] = distance_matrix[u, v]
                    
    return mst_adj

def heuristics(distance_matrix):
    start_time = time.time()
    # Safety margin: evaluator kills at 330s, we should finish well before.
    # Let's aim to finish within 280s to be safe.
    deadline = start_time + 280.0
    
    n = distance_matrix.shape[0]
    
    # Base heuristic: inverse distance
    safe_dist = distance_matrix.copy()
    np.fill_diagonal(safe_dist, 1.0)
    safe_dist[safe_dist < 1e-9] = 1e-9
    
    base_heur = 1.0 / safe_dist
    
    # Compute Delaunay circumradii for geometric guidance
    circumradii, _ = _compute_delaunay_circumradii(distance_matrix)
    
    # Heuristic = (base_heur ** beta) * (geometric_factor)
    # geometric_factor = (local_circumradius / edge_length) ** gamma
    # If edge is not in Delaunay, factor is 1 (or small penalty)
    # Let's define:
    # ratio = circumradius / distance
    # For Delaunay edges, ratio is typically > 0.5 (since R >= a/2)
    # We want to reward edges that are "tight" locally? Or "loose"?
    # Actually, the directive says: "rewards edges fitting the local point density"
    # High density -> small circumradius.
    # So small circumradius is good.
    # factor = (1 / circumradius) ** gamma ?
    # Or factor = (distance / circumradius) ** gamma ?
    # Let's try: factor = (distance / circumradius) ** gamma
    # For a very tight triangle (small R), d/R is large.
    # For a flat triangle (large R), d/R is small.
    # This rewards edges in dense areas.
    
    # Prepare geometric factor matrix
    # dist is distance_matrix
    # circumradii is inf for non-Delaunay edges
    # We want to apply this only to Delaunay edges.
    
    # Let's search for beta and gamma
    def evaluate_params(beta, gamma):
        if time.time() > deadline - 5.0:
            return float('inf')
            
        h = base_heur ** beta
        
        # Apply geometric factor
        # valid_mask = ~np.isinf(circumradii)
        # ratio = distance_matrix / circumradii
        # factor = ratio ** gamma
        
        # To avoid inf/0 issues:
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = safe_dist / circumradii
            ratio[~np.isfinite(ratio)] = 1.0 # Default to 1 if not defined
            factor = ratio ** gamma
            factor[~np.isfinite(factor)] = 1.0
            factor[factor <= 0] = 1e-9
            
        h *= factor
        
        # Ensure validity
        h[h <= 0] = 1e-9
        h[~np.isfinite(h)] = 1e-9
        
        # Use a fixed seed for consistent comparison during search
        # Using 5 iterations is a good surrogate for heuristic quality and faster
        score = _fast_aco_simulate(distance_matrix, h, n_ants=20, n_iterations=5, seed=42)
        return score

    best_beta = 2.0
    best_gamma = 1.0
    best_score = float('inf')
    
    # Coarse search over beta and gamma
    # Beta range: 0.5 to 6.0
    # Gamma range: 0.0 to 3.0 (0 means no geometric effect)
    betas_coarse = np.linspace(0.5, 6.0, 12)
    gammas_coarse = np.linspace(0.0, 3.0, 8)
    
    for beta in betas_coarse:
        for gamma in gammas_coarse:
            if time.time() > deadline - 60.0:
                break
            score = evaluate_params(beta, gamma)
            if score < best_score:
                best_score = score
                best_beta = beta
                best_gamma = gamma
        if time.time() > deadline - 60.0:
            break
            
    # Fine search around the best coarse parameters
    if time.time() < deadline - 30.0:
        lower_b = max(0.1, best_beta - 1.5)
        upper_b = min(8.0, best_beta + 1.5)
        lower_g = max(0.0, best_gamma - 1.0)
        upper_g = min(4.0, best_gamma + 1.0)
        
        betas_fine = np.linspace(lower_b, upper_b, 15)
        gammas_fine = np.linspace(lower_g, upper_g, 10)
        
        for beta in betas_fine:
            for gamma in gammas_fine:
                if time.time() > deadline - 10.0:
                    break
                score = evaluate_params(beta, gamma)
                if score < best_score:
                    best_score = score
                    best_beta = beta
                    best_gamma = gamma
            if time.time() > deadline - 10.0:
                break
                
    # Final check: verify the best parameters with a longer run?
    if time.time() < deadline - 15.0:
        h_final = base_heur ** best_beta
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = safe_dist / circumradii
            ratio[~np.isfinite(ratio)] = 1.0
            factor = ratio ** best_gamma
            factor[~np.isfinite(factor)] = 1.0
            factor[factor <= 0] = 1e-9
        h_final *= factor
        h_final[h_final <= 0] = 1e-9
        h_final[~np.isfinite(h_final)] = 1e-9
        
        # Full simulation to confirm
        final_score = _fast_aco_simulate(distance_matrix, h_final, n_ants=20, n_iterations=30, seed=42)
        
    # Construct final heuristic
    final_heuristic = base_heur ** best_beta
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = safe_dist / circumradii
        ratio[~np.isfinite(ratio)] = 1.0
        factor = ratio ** best_gamma
        factor[~np.isfinite(factor)] = 1.0
        factor[factor <= 0] = 1e-9
    final_heuristic *= factor
    final_heuristic[final_heuristic <= 0] = 1e-9
    final_heuristic[~np.isfinite(final_heuristic)] = 1e-9
    
    return final_heuristic
