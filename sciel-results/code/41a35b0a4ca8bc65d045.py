"""Dynamic geometric curvature minimization for TSP tour construction."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global state
_DEADLINE = None
_START_TIME = None
_CACHE = {}

def _get_instance_id(distance_matrix):
    """Create a lightweight fingerprint for the distance matrix."""
    n = distance_matrix.shape[0]
    if n > 10:
        sample = distance_matrix[::max(1, n // 10), :].tobytes()
    else:
        sample = distance_matrix.tobytes()
    return hash(sample)

def _reconstruct_coords_from_distance_matrix(distance_matrix):
    """
    Reconstruct 2D coordinates from a Euclidean distance matrix using classical MDS.
    Returns an (n, 2) array of coordinates.
    """
    n = distance_matrix.shape[0]
    D = distance_matrix.copy()
    
    # Classical MDS
    # Double centering
    ones = np.ones((n, n)) / (2 * n)
    B = -0.5 * (np.ones((n, 1)) @ D**2 @ np.ones((1, n)) 
               - D**2 
               - np.ones((n, 1)) @ D**2 @ np.ones((1, n))
               + D**2 @ ones + ones @ D**2)
    
    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(B)
    
    # Sort in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Take top 2 dimensions
    # Clip negative eigenvalues to 0
    eigvals = np.maximum(eigvals, 0)
    
    coords = eigvecs[:, :2] * np.sqrt(eigvals[:2])
    
    return coords

def _curvature_cost(current_node, prev_node, candidate_node, coords, distance_matrix, alpha=0.5):
    """
    Compute the composite cost for selecting candidate_node:
    - Immediate Euclidean distance from current_node to candidate_node
    - Squared cross-product magnitude (turning angle penalty) between
      the vector (prev_node -> current_node) and (current_node -> candidate_node)
    
    alpha controls the weight of the curvature penalty relative to distance.
    """
    # Distance component
    d = distance_matrix[current_node, candidate_node]
    
    # Curvature component: cross product of incoming and outgoing edge vectors
    if prev_node is not None:
        v1 = coords[current_node] - coords[prev_node]      # incoming edge vector
        v2 = coords[candidate_node] - coords[current_node] # outgoing edge vector
        # 2D cross product magnitude (scalar)
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        curvature_penalty = cross * cross / (max(np.dot(v1, v1), 1e-10) * max(np.dot(v2, v2), 1e-10))
    else:
        curvature_penalty = 0.0
    
    return d + alpha * curvature_penalty

def _construct_curvature_tour(distance_matrix, coords, start, alpha=0.5, beta=0.3):
    """
    Construct a tour using dynamic geometric curvature minimization.
    
    At each step, score each unvisited candidate by:
    score = distance(current, candidate) 
          + alpha * curvature_penalty(prev, current, candidate)
          + beta * distance(candidate, destination)
    
    The destination term provides lookahead to avoid ending up far from the return point.
    """
    n = distance_matrix.shape[0]
    unvisited = set(range(n))
    unvisited.remove(start)
    tour = [start]
    curr = start
    prev = None
    
    while unvisited:
        if len(unvisited) == 1:
            next_node = list(unvisited)[0]
            tour.append(next_node)
            unvisited.remove(next_node)
            break
        
        unvisited_list = list(unvisited)
        
        # Compute scores for all candidates
        best_score = np.inf
        best_node = None
        
        for cand in unvisited_list:
            # Distance from current
            d_curr = distance_matrix[curr, cand]
            
            # Curvature penalty
            if prev is not None:
                v1 = coords[curr] - coords[prev]
                v2 = coords[cand] - coords[curr]
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                curvature_penalty = cross * cross / (max(np.dot(v1, v1), 1e-10) * max(np.dot(v2, v2), 1e-10))
            else:
                curvature_penalty = 0.0
            
            # Lookahead: distance to destination (return to start)
            d_dest = distance_matrix[cand, start]
            
            # Composite score
            score = d_curr + alpha * curvature_penalty + beta * d_dest
            
            if score < best_score:
                best_score = score
                best_node = cand
        
        tour.append(best_node)
        unvisited.remove(best_node)
        prev = curr
        curr = best_node
    
    return np.array(tour)

def _two_opt(tour, distance_matrix, deadline):
    """Apply 2-Opt improvement to a tour."""
    n = len(tour)
    if n <= 3:
        return tour
    
    tour = tour.copy()
    current_cost = np.sum(distance_matrix[tour[:-1], tour[1:]]) + distance_matrix[tour[-1], tour[0]]
    
    improved = True
    passes = 0
    max_passes = 500
    
    while improved and passes < max_passes:
        improved = False
        passes += 1
        
        if time.time() > deadline - 3.0:
            break
        
        for i in range(n - 1):
            if time.time() > deadline - 3.0:
                break
            
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                
                u = tour[i]
                v = tour[i + 1]
                x = tour[j]
                y = tour[(j + 1) % n]
                
                d_uv = distance_matrix[u, v]
                d_xy = distance_matrix[x, y]
                d_ux = distance_matrix[u, x]
                d_vy = distance_matrix[v, y]
                
                delta = d_ux + d_vy - d_uv - d_xy
                
                if delta < -1e-10:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                    current_cost += delta
                    improved = True
                    break
            
            if improved:
                break
    
    return tour

def _construct_best_tour(distance_matrix, deadline):
    """
    Generate multiple candidate tours using curvature minimization with different
    starting points and alpha values, refine with 2-Opt, and return the best one.
    """
    n = distance_matrix.shape[0]
    best_tour = None
    best_len = np.inf
    
    # Reconstruct coordinates via MDS
    coords = _reconstruct_coords_from_distance_matrix(distance_matrix)
    
    # Try different starting points and curvature weights
    alpha_values = [0.3, 0.5, 0.7, 1.0]
    beta_values = [0.2, 0.3, 0.4]
    
    # For N=50, try a subset of starts combined with different alpha/beta
    starts = list(range(min(10, n)))
    
    for start in starts:
        if time.time() > deadline - 5.0:
            break
        
        for alpha in alpha_values:
            if time.time() > deadline - 5.0:
                break
            
            for beta in beta_values:
                if time.time() > deadline - 5.0:
                    break
                
                tour = _construct_curvature_tour(distance_matrix, coords, start, alpha, beta)
                tour = _two_opt(tour, distance_matrix, deadline)
                
                tour_len = np.sum(distance_matrix[tour[:-1], tour[1:]]) + distance_matrix[tour[-1], tour[0]]
                if tour_len < best_len:
                    best_len = tour_len
                    best_tour = tour.copy()
    
    # Also try pure NN as a baseline
    if time.time() < deadline - 5.0:
        for start in range(min(5, n)):
            if time.time() > deadline - 5.0:
                break
            # Simple NN
            unvisited = set(range(n))
            unvisited.remove(start)
            tour = [start]
            curr = start
            while unvisited:
                unvisited_list = list(unvisited)
                dists = distance_matrix[curr, unvisited_list]
                min_idx = np.argmin(dists)
                next_node = unvisited_list[min_idx]
                tour.append(next_node)
                unvisited.remove(next_node)
                curr = next_node
            
            tour = np.array(tour)
            tour = _two_opt(tour, distance_matrix, deadline)
            
            tour_len = np.sum(distance_matrix[tour[:-1], tour[1:]]) + distance_matrix[tour[-1], tour[0]]
            if tour_len < best_len:
                best_len = tour_len
                best_tour = tour.copy()
    
    if best_tour is None:
        best_tour = np.arange(n)
    
    return best_tour

def select_next_node(current_node, destination_node, unvisited_nodes, distance_matrix):
    global _DEADLINE, _CACHE, _START_TIME
    
    if _START_TIME is None:
        _START_TIME = time.time()
        _DEADLINE = _START_TIME + 140.0  # 10s safety margin before 150s kill
    
    # Deadline guard: fall back to nearest neighbor if time is up
    if time.time() > _DEADLINE:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    n_unvisited = len(unvisited_nodes)
    if n_unvisited == 1:
        return int(unvisited_nodes[0])
    
    if n_unvisited == 2:
        d0 = distance_matrix[current_node, unvisited_nodes[0]]
        d1 = distance_matrix[current_node, unvisited_nodes[1]]
        d_dest_0 = distance_matrix[unvisited_nodes[0], destination_node]
        d_dest_1 = distance_matrix[unvisited_nodes[1], destination_node]
        d_cross = distance_matrix[unvisited_nodes[0], unvisited_nodes[1]]
        
        cost_pick_0 = d0 + d_cross + d_dest_1
        cost_pick_1 = d1 + d_cross + d_dest_0
        
        if cost_pick_0 <= cost_pick_1:
            return int(unvisited_nodes[0])
        else:
            return int(unvisited_nodes[1])
    
    instance_id = _get_instance_id(distance_matrix)
    if instance_id in _CACHE:
        full_tour = _CACHE[instance_id]
    else:
        full_tour = _construct_best_tour(distance_matrix, _DEADLINE)
        _CACHE[instance_id] = full_tour
    
    try:
        idx = np.where(full_tour == current_node)[0][0]
        next_node = full_tour[(idx + 1) % len(full_tour)]
    except IndexError:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])
    
    if next_node in unvisited_nodes:
        return int(next_node)
    else:
        return int(unvisited_nodes[np.argmin(distance_matrix[current_node, unvisited_nodes])])

# EVOLVE-BLOCK-END
