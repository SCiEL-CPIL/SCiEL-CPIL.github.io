import numpy as np
import scipy.spatial as spat
import time
import itertools

N = 14

def calc_score_and_min_tri(points):
    """
    Calculates score and the indices of the minimum area triangle.
    Returns (score, min_idx) where min_idx is (i, j, k).
    """
    try:
        hull = spat.ConvexHull(points)
        hull_area = hull.volume
        if hull_area < 1e-10:
            return 0.0, (0, 1, 2)
    except Exception:
        return 0.0, (0, 1, 2)
    
    n = len(points)
    # Generate all combinations of 3 indices
    # For N=14, C(14,3) = 364.
    # Vectorized calculation for speed and to easily find argmin
    
    # We can precompute the indices for all triples to avoid overhead in the loop
    # But for N=14, even a loop is fast. Let's stick to vectorized for robustness.
    # However, creating the index arrays every call is overhead.
    # Let's cache the indices? No, N is fixed.
    
    # Actually, for coordinate descent, we call this many times.
    # Let's optimize the index generation.
    i_idx, j_idx, k_idx = np.indices((n, n, n)).reshape(3, -1)
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    k_idx = k_idx[mask]
    
    pi = points[i_idx]
    pj = points[j_idx]
    pk = points[k_idx]
    
    v1 = pj - pi
    v2 = pk - pi
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    min_pos = np.argmin(areas)
    
    min_i = i_idx[min_pos]
    min_j = j_idx[min_pos]
    min_k = k_idx[min_pos]
    
    score = min_area / hull_area
    return score, (min_i, min_j, min_k)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s kill
    
    # Initial guess: Regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    points = np.column_stack((np.cos(angles), np.sin(angles)))
    
    best_points = points.copy()
    best_score, _ = calc_score_and_min_tri(points)
    
    current_points = best_points.copy()
    current_score = best_score
    
    rng = np.random.default_rng(42)
    
    # Coordinate Descent with Stochastic Hill Climbing
    # Strategy:
    # 1. Identify the triangle with the minimum area (the bottleneck).
    # 2. Try to move one of its vertices to increase its area (and hopefully not decrease others too much).
    # 3. Use a local random search around the current positions of the 3 vertices of the bottleneck triangle.
    # 4. If a better global score is found, accept.
    
    # To make it faster, we can pre-compute the indices of all triangles.
    # Let's do that once.
    n = N
    i_idx, j_idx, k_idx = np.indices((n, n, n)).reshape(3, -1)
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    tri_indices = np.column_stack((i_idx[mask], j_idx[mask], k_idx[mask]))
    num_tris = len(tri_indices)
    
    def fast_score(points):
        """
        Fast score calculation using pre-computed triangle indices.
        Returns score and min triangle indices.
        """
        try:
            hull = spat.ConvexHull(points)
            hull_area = hull.volume
            if hull_area < 1e-10:
                return 0.0, (0, 1, 2)
        except Exception:
            return 0.0, (0, 1, 2)
            
        pi = points[tri_indices[:, 0]]
        pj = points[tri_indices[:, 1]]
        pk = points[tri_indices[:, 2]]
        
        v1 = pj - pi
        v2 = pk - pi
        
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        min_area = np.min(areas)
        min_pos = np.argmin(areas)
        
        min_i = tri_indices[min_pos, 0]
        min_j = tri_indices[min_pos, 1]
        min_k = tri_indices[min_pos, 2]
        
        score = min_area / hull_area
        return score, (min_i, min_j, min_k)

    # Re-evaluate with the faster function
    current_score, min_tri = fast_score(current_points)
    best_score = current_score
    best_points = current_points.copy()
    
    # Parameters for the local search around the bottleneck triangle
    # We will try to move the vertices of the min triangle.
    # Step size will decay over time.
    
    base_step = 0.5
    decay = 0.999
    
    for step in range(50000): # High iteration count, but each is fast
        if time.time() > deadline:
            break
        
        # Identify current bottleneck triangle
        _, (i, j, k) = fast_score(current_points)
        
        # We will try to move one of these vertices
        # Let's pick a random vertex from the bottleneck triangle
        vertex_to_move = rng.choice([i, j, k])
        
        # Generate a perturbation for this vertex
        # The direction should ideally push the triangle area up.
        # But simple random walk in the neighborhood is a good heuristic.
        # Let's use a Gaussian step.
        step_size = base_step * (decay ** step)
        if step_size < 1e-4:
            step_size = 1e-4
            
        # Perturb the chosen vertex
        new_points = current_points.copy()
        noise = rng.normal(0, step_size, size=2)
        new_points[vertex_to_move] += noise
        
        new_score, _ = fast_score(new_points)
        
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            
            if new_score > best_score:
                best_score = new_score
                best_points = new_points.copy()
                # Occasionally reset step size if we found a new global best?
                # No, let it decay naturally, but maybe reheat slightly?
                # For simplicity, just continue.
        else:
            # Rejection: keep current_points
            # To help escape local minima, we can occasionally accept worse moves
            # But for Heilbronn, the landscape is tricky. 
            # Let's add a small probability of accepting worse moves to escape plateaus.
            if rng.random() < 0.05:
                # Accept with some probability if not too bad?
                # Or just do a larger jump?
                # Let's do a larger jump if stuck.
                pass

    return best_points
