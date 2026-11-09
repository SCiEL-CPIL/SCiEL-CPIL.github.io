# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex 20-gon with equidistant vertices using multi-restart L-BFGS-B optimization.
    
    The strategy is to parameterize the polygon by 19 edge angles (turn angles) that sum to 2*pi,
    then optimize to maximize the minimum vertex score. We use multiple restarts from diverse
    initial points to escape local optima.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600 second limit
    
    def compute_polygon_from_turn_angles(turn_angles):
        """
        Given 19 turn angles (summing to 2*pi - last_angle), compute the 20 vertices.
        We parameterize by 19 angles where the 20th is determined by the closure constraint.
        Actually, for a closed polygon, the sum of exterior angles must be 2*pi.
        We use 19 free parameters and the 20th is 2*pi - sum of first 19.
        """
        n = 20
        # Ensure turn angles are positive and sum to 2*pi
        # The last turn angle is determined
        last_angle = 2.0 * np.pi - np.sum(turn_angles[:19])
        all_angles = np.append(turn_angles[:19], last_angle)
        
        # Ensure all angles are positive
        if np.any(all_angles <= 0):
            return None
        
        # Compute edge directions
        edge_directions = np.cumsum(np.concatenate([[0.0], all_angles[:-1]]))
        
        # Compute unit edge vectors
        edge_vectors = np.column_stack([np.cos(edge_directions), np.sin(edge_directions)])
        
        # For a closed polygon with unit edges, we need to solve for edge lengths
        # such that the sum of edge_vectors * lengths = 0
        # This is a linear system: A * lengths = 0 where A is 2x20
        # We need non-negative lengths
        
        # Actually, let's use a different parameterization:
        # Use 19 angles and optimize the edge lengths too, or use a simpler approach.
        
        # Simpler: use angles to define directions, then find edge lengths for closure
        # The constraint is: sum_i (l_i * e_i) = 0 where e_i are unit vectors
        # This gives us 2 equations in 20 unknowns, so we have 18 degrees of freedom for lengths
        
        # For simplicity, let's use equal edge lengths and just check if it closes
        # If not, we can adjust. But for optimization, let's use a different parameterization.
        
        # Better parameterization: use 19 angles and 19 edge lengths, with the 20th 
        # determined by closure. But that's complex.
        
        # Alternative: use the fact that for a convex polygon, we can specify vertices
        # directly and enforce convexity. Let's go back to direct vertex optimization
        # but with a better initial guess and more restarts.
        
        return None
    
    def vertex_score(v, all_vertices):
        """Compute the score for a single vertex v given all other vertices."""
        n = len(all_vertices)
        # Compute distances from v to all other vertices
        diffs = all_vertices - v
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        
        # Remove the distance to itself (which is 0)
        # Find the index of v in all_vertices
        # Actually, v is one of all_vertices, so we need to exclude it
        # Let's compute distances to all vertices except v
        idx = np.where((all_vertices == v).all(axis=1))[0]
        if len(idx) == 0:
            return 0.0
        idx = idx[0]
        
        # Get distances to other 19 vertices
        other_dists = np.delete(dists, idx)
        other_dists = np.sort(other_dists)
        
        # Check if any distance is < 0.05
        if np.any(other_dists < 0.05):
            return 0.0
        
        # Find k: first index minimizing d_{k+1} - d_k
        diffs = np.diff(other_dists)
        k = np.argmin(diffs)
        D = (other_dists[k] + other_dists[k + 1]) / 2.0
        
        if D < 0.1:
            return 0.0
        
        # Take four distances
        if k == 0:
            selected = other_dists[0:4]
        elif k == 17:
            selected = other_dists[15:19]
        else:
            selected = other_dists[k-1:k+3]
        
        # raw = mean of max(d/D, D/d) for these four
        ratios = np.maximum(selected / D, D / selected)
        raw = np.mean(ratios)
        
        return min(1.0, 1.0 / raw)
    
    def min_vertex_score(vertices):
        """Compute the minimum vertex score over all 20 vertices."""
        scores = []
        for i in range(20):
            s = vertex_score(vertices[i], vertices)
            scores.append(s)
        return min(scores)
    
    def negative_min_score(vertices_flat):
        """Objective function to minimize (negative of min vertex score)."""
        vertices = vertices_flat.reshape(20, 2)
        
        # Check for repeated points
        for i in range(20):
            for j in range(i + 1, 20):
                if np.linalg.norm(vertices[i] - vertices[j]) < 1e-6:
                    return 1000.0  # Large penalty
        
        # Check convexity: compute cross products of consecutive edges
        cross_products = []
        for i in range(20):
            v0 = vertices[i]
            v1 = vertices[(i + 1) % 20]
            v2 = vertices[(i + 2) % 20]
            e1 = v1 - v0
            e2 = v2 - v1
            cross = e1[0] * e2[1] - e1[1] * e2[0]
            cross_products.append(cross)
        
        cross_products = np.array(cross_products)
        
        # All cross products should be nonzero and of the same sign
        if np.any(np.abs(cross_products) < 1e-8):
            return 1000.0
        
        if not (np.all(cross_products > 0) or np.all(cross_products < 0)):
            return 1000.0
        
        # Normalize: subtract centroid, divide by mean distance to centroid
        centroid = np.mean(vertices, axis=0)
        vertices_centered = vertices - centroid
        dists_to_centroid = np.sqrt(np.sum(vertices_centered ** 2, axis=1))
        mean_dist = np.mean(dists_to_centroid)
        
        if mean_dist <= 0.01:
            return 1000.0
        
        vertices_normalized = vertices_centered / mean_dist
        
        # Compute min vertex score
        score = min_vertex_score(vertices_normalized)
        
        return -score
    
    # Initial guess: regular 20-gon
    n = 20
    radius = 1.0
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    initial_vertices = np.stack([x, y], axis=1)
    
    # Center and normalize
    centroid = np.mean(initial_vertices, axis=0)
    initial_vertices_centered = initial_vertices - centroid
    dists_to_centroid = np.sqrt(np.sum(initial_vertices_centered ** 2, axis=1))
    mean_dist = np.mean(dists_to_centroid)
    initial_vertices_normalized = initial_vertices_centered / mean_dist
    
    best_vertices = initial_vertices_normalized.copy()
    best_score = min_vertex_score(best_vertices)
    
    # Multi-restart L-BFGS-B
    num_restarts = 50
    rng = np.random.default_rng(42)
    
    for restart in range(num_restarts):
        if time.time() > deadline - 60.0:
            break
        
        # Generate diverse initial point
        if restart == 0:
            init_flat = initial_vertices_normalized.flatten()
        else:
            # Perturb the initial point
            noise_scale = 0.1 + 0.5 * (restart / num_restarts)
            init_flat = initial_vertices_normalized.flatten() + rng.normal(0, noise_scale, 40)
        
        # Optimize
        try:
            result = minimize(
                negative_min_score,
                init_flat,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < -best_score:
                candidate_vertices = result.x.reshape(20, 2)
                
                # Check validity
                valid = True
                for i in range(20):
                    for j in range(i + 1, 20):
                        if np.linalg.norm(candidate_vertices[i] - candidate_vertices[j]) < 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    # Check convexity
                    cross_products = []
                    for i in range(20):
                        v0 = candidate_vertices[i]
                        v1 = candidate_vertices[(i + 1) % 20]
                        v2 = candidate_vertices[(i + 2) % 20]
                        e1 = v1 - v0
                        e2 = v2 - v1
                        cross = e1[0] * e2[1] - e1[1] * e2[0]
                        cross_products.append(cross)
                    
                    cross_products = np.array(cross_products)
                    if np.any(np.abs(cross_products) < 1e-8):
                        valid = False
                    elif not (np.all(cross_products > 0) or np.all(cross_products < 0)):
                        valid = False
                
                if valid:
                    # Normalize
                    centroid = np.mean(candidate_vertices, axis=0)
                    candidate_centered = candidate_vertices - centroid
                    dists_to_centroid = np.sqrt(np.sum(candidate_centered ** 2, axis=1))
                    mean_dist = np.mean(dists_to_centroid)
                    
                    if mean_dist > 0.01:
                        candidate_normalized = candidate_centered / mean_dist
                        candidate_score = min_vertex_score(candidate_normalized)
                        
                        if candidate_score > best_score:
                            best_score = candidate_score
                            best_vertices = candidate_normalized.copy()
        except Exception:
            continue
    
    return best_vertices


# EVOLVE-BLOCK-END
