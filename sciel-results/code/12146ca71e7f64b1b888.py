# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def _compute_score(vertices):
    """Compute the minimum vertex score for a given polygon."""
    n = len(vertices)
    if n != 20:
        return 0.0
    
    # Normalize: subtract centroid, divide by mean distance
    centroid = vertices.mean(axis=0)
    vertices = vertices - centroid
    dists_to_centroid = np.linalg.norm(vertices, axis=1)
    mean_dist = dists_to_centroid.mean()
    if mean_dist <= 0.01:
        return 0.0
    vertices = vertices / mean_dist
    
    # Check for repeated points
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(vertices[i] - vertices[j]) < 0.05:
                return 0.0
    
    # Check strict convexity: all turns same sign, total turning +-2*pi
    cross_products = []
    for i in range(n):
        p0 = vertices[i]
        p1 = vertices[(i+1) % n]
        p2 = vertices[(i+2) % n]
        edge1 = p1 - p0
        edge2 = p2 - p1
        cross = edge1[0] * edge2[1] - edge1[1] * edge2[0]
        cross_products.append(cross)
    
    cross_products = np.array(cross_products)
    if np.any(np.abs(cross_products) < 1e-10):
        return 0.0
    if not (np.all(cross_products > 0) or np.all(cross_products < 0)):
        return 0.0
    
    # Compute total turning
    angles = []
    for i in range(n):
        p0 = vertices[i]
        p1 = vertices[(i+1) % n]
        p2 = vertices[(i+2) % n]
        edge1 = p1 - p0
        edge2 = p2 - p1
        angle1 = np.arctan2(edge1[1], edge1[0])
        angle2 = np.arctan2(edge2[1], edge2[0])
        turn = angle2 - angle1
        if turn > np.pi:
            turn -= 2 * np.pi
        elif turn < -np.pi:
            turn += 2 * np.pi
        angles.append(turn)
    
    total_turning = sum(angles)
    if abs(abs(total_turning) - 2 * np.pi) > 1e-6:
        return 0.0
    
    # Compute vertex scores
    min_score = 1.0
    for i in range(n):
        v = vertices[i]
        # Distances to all other vertices
        d = np.linalg.norm(vertices - v, axis=1)
        d[i] = np.inf
        d_sorted = np.sort(d)
        
        # Check if any distance < 0.05
        if d_sorted[0] < 0.05:
            return 0.0
        
        # Find k: first index minimizing d_{k+1} - d_k
        diffs = d_sorted[1:] - d_sorted[:-1]
        k = np.argmin(diffs)
        D = (d_sorted[k] + d_sorted[k+1]) / 2
        if D < 0.1:
            return 0.0
        
        # Take four distances
        if k == 0:
            four_d = d_sorted[0:4]
        elif k == 17:
            four_d = d_sorted[15:19]
        else:
            four_d = d_sorted[k-1:k+3]
        
        raw = np.mean([max(d/D, D/d) for d in four_d])
        vertex_score = min(1.0, 1.0 / raw) if raw > 0 else 0.0
        min_score = min(min_score, vertex_score)
    
    return min_score


def _fourier_to_polygon(coeffs, n=20):
    """Convert Fourier coefficients to polygon vertices, ensuring convexity."""
    # coeffs: (m, 2) where m is number of harmonics (excluding DC)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = np.zeros(n)
    y = np.zeros(n)
    for k, (ak, bk) in enumerate(coeffs, start=1):
        x += ak * np.cos(k * angles)
        y += bk * np.sin(k * angles)
    return np.stack([x, y], axis=1)


def _polygon_to_fourier(polygon, n=20):
    """Convert polygon vertices to Fourier coefficients."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = polygon[:, 0]
    y = polygon[:, 1]
    coeffs = []
    for k in range(1, n//2 + 1):
        ak = (2/n) * np.sum(x * np.cos(k * angles))
        bk = (2/n) * np.sum(y * np.sin(k * angles))
        coeffs.append([ak, bk])
    return np.array(coeffs)


def _negative_score(fourier_coeffs):
    """Objective function for optimization (negate score for minimization)."""
    try:
        polygon = _fourier_to_polygon(fourier_coeffs)
        score = _compute_score(polygon)
        return -score
    except:
        return 1.0


def equidistant_points_polygon20() -> np.ndarray:
    """
    Build a convex polygon with n=20 vertices optimized for the equidistance property.
    
    Uses Fourier representation + differential evolution + local refinement.
    """
    start_time = time.time()
    deadline = start_time + 600  # 600s safety margin
    
    n = 20
    # Start with regular polygon as initial guess
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    regular_polygon = np.stack([x, y], axis=1)
    
    # Convert to Fourier coefficients
    best_fourier = _polygon_to_fourier(regular_polygon)
    best_score = _compute_score(regular_polygon)
    best_polygon = regular_polygon.copy()
    
    # Number of harmonics to use (keep it moderate for speed)
    n_harmonics = 4
    
    # Define bounds for Fourier coefficients
    # For convexity, we want the radius to be positive everywhere
    # Use moderate bounds
    bounds = [(-0.5, 0.5)] * (2 * n_harmonics)
    
    # Try differential evolution if time permits
    elapsed = time.time() - start_time
    if elapsed < 400:  # Leave 200s for local refinement
        try:
            def objective(flat_coeffs):
                coeffs = flat_coeffs.reshape(-1, 2)
                return _negative_score(coeffs)
            
            # Initial population around regular polygon
            x0 = best_fourier.flatten()
            
            result = differential_evolution(
                objective,
                bounds,
                maxiter=100,
                popsize=10,
                tol=1e-6,
                seed=42,
                polish=False,
                init='latinhypercube',
                updating='deferred',
                workers=1
            )
            
            elapsed = time.time() - start_time
            if elapsed < 500:
                candidate_fourier = result.x.reshape(-1, 2)
                candidate_polygon = _fourier_to_polygon(candidate_fourier)
                candidate_score = _compute_score(candidate_polygon)
                
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_polygon = candidate_polygon
                    best_fourier = candidate_fourier
        except Exception as e:
            pass
    
    # Local refinement with Nelder-Mead if time permits
    elapsed = time.time() - start_time
    if elapsed < 550 and best_score > 0:
        try:
            def objective_flat(flat_coeffs):
                coeffs = flat_coeffs.reshape(-1, 2)
                return _negative_score(coeffs)
            
            result = minimize(
                objective_flat,
                best_fourier.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            elapsed = time.time() - start_time
            if elapsed < 580:
                candidate_fourier = result.x.reshape(-1, 2)
                candidate_polygon = _fourier_to_polygon(candidate_fourier)
                candidate_score = _compute_score(candidate_polygon)
                
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_polygon = candidate_polygon
                    best_fourier = candidate_fourier
        except Exception as e:
            pass
    
    # Final coordinate descent polish if time permits
    elapsed = time.time() - start_time
    if elapsed < 590 and best_score > 0:
        try:
            current_fourier = best_fourier.copy()
            current_score = best_score
            
            for iteration in range(50):
                if time.time() - start_time > 590:
                    break
                
                improved = False
                for i in range(len(current_fourier)):
                    for j in range(2):
                        # Try small perturbations
                        for delta in [0.001, -0.001, 0.0005, -0.0005, 0.0001, -0.0001]:
                            test_fourier = current_fourier.copy()
                            test_fourier[i, j] += delta
                            test_polygon = _fourier_to_polygon(test_fourier)
                            test_score = _compute_score(test_polygon)
                            
                            if test_score > current_score + 1e-10:
                                current_fourier = test_fourier
                                current_score = test_score
                                improved = True
                                break
                
                if not improved:
                    break
            
            if current_score > best_score:
                best_score = current_score
                best_polygon = _fourier_to_polygon(current_fourier)
        except Exception as e:
            pass
    
    return best_polygon


# EVOLVE-BLOCK-END
