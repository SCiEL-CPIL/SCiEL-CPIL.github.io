import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools
import time

def heilbronn_convex13() -> np.ndarray:
    n = 13
    tri_indices = np.array(list(itertools.combinations(range(n), 3)))
    
    # Precompute indices for hull area calculation assuming order 0..n-1
    # x_i * y_{i+1} - x_{i+1} * y_i
    i_idx = np.arange(n)
    j_idx = (np.arange(n) + 1) % n
    
    def score(points):
        # points shape (13, 2)
        p1 = points[tri_indices[:, 0]]
        p2 = points[tri_indices[:, 1]]
        p3 = points[tri_indices[:, 2]]
        
        dx1 = p2[:, 0] - p1[:, 0]
        dy1 = p2[:, 1] - p1[:, 1]
        dx2 = p3[:, 0] - p1[:, 0]
        dy2 = p3[:, 1] - p1[:, 1]
        
        cross = dx1 * dy2 - dy1 * dx2
        areas = 0.5 * np.abs(cross)
        min_area = np.min(areas)
        
        # Check if points are in convex position with order 0..n-1
        # Calculate cross products of consecutive edges
        x = points[:, 0]
        y = points[:, 1]
        
        # Vectorized cross product for edges (i, i+1) and (i+1, i+2)
        # Or simply check the sign of the cross product of (P_{i+1}-P_i) and (P_{i+2}-P_{i+1})
        # Actually, easier: check if the polygon is convex.
        # A polygon is convex if all cross products of consecutive edges have the same sign.
        
        # Edge i: P_{i+1} - P_i
        # Edge i+1: P_{i+2} - P_{i+1}
        
        # Let's compute cross products of consecutive edges
        # e_i = P_{i+1} - P_i
        # e_{i+1} = P_{i+2} - P_{i+1}
        # cross(e_i, e_{i+1})
        
        # Indices for edges
        # i from 0 to n-1
        # P_i, P_{i+1}, P_{i+2} (mod n)
        
        idx0 = np.arange(n)
        idx1 = (np.arange(n) + 1) % n
        idx2 = (np.arange(n) + 2) % n
        
        p0 = points[idx0]
        p1v = points[idx1]
        p2v = points[idx2]
        
        e1 = p1v - p0
        e2 = p2v - p1v
        
        cross_e = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
        
        # If all cross_e have the same sign (and non-zero), it's convex
        if np.all(cross_e > 1e-9) or np.all(cross_e < -1e-9):
            # Convex, calculate area using shoelace formula
            # Area = 0.5 * | sum(x_i * y_{i+1} - x_{i+1} * y_i) |
            x_next = x[j_idx]
            y_next = y[j_idx]
            area_sum = np.sum(x * y_next - x_next * y)
            hull_area = 0.5 * np.abs(area_sum)
        else:
            # Not convex or degenerate, use ConvexHull
            try:
                hull = ConvexHull(points)
                hull_area = hull.volume
            except Exception:
                return 0.0
                
        if hull_area <= 0:
            return 0.0
            
        return min_area / hull_area

    def neg_score(points_flat):
        points = points_flat.reshape((n, 2))
        s = score(points)
        return -s

    # Initial guess: Regular 13-gon
    angles = 2 * np.pi * np.arange(n) / n
    x0 = np.column_stack((np.cos(angles), np.sin(angles))).flatten()
    
    start_time = time.time()
    deadline = start_time + 590.0
    
    best_points = np.column_stack((np.cos(angles), np.sin(angles)))
    best_score = score(best_points)
    
    try:
        # Nelder-Mead
        result = minimize(neg_score, x0, method='Nelder-Mead', 
                          options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        
        if result.x is not None:
            candidate_points = result.x.reshape((n, 2))
            candidate_score = score(candidate_points)
            
            if candidate_score > best_score:
                best_points = candidate_points
                best_score = candidate_score
                
    except Exception:
        pass

    # If time allows, try a second run with a slightly perturbed start?
    if time.time() < deadline - 30:
        # Perturb initial guess
        rng = np.random.default_rng(42)
        perturbation = rng.normal(0, 0.1, (n, 2))
        x0_perturbed = (best_points + perturbation).flatten()
        
        try:
            result2 = minimize(neg_score, x0_perturbed, method='Nelder-Mead', 
                               options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
            
            if result2.x is not None:
                candidate_points2 = result2.x.reshape((n, 2))
                candidate_score2 = score(candidate_points2)
                
                if candidate_score2 > best_score:
                    best_points = candidate_points2
                    best_score = candidate_score2
        except Exception:
            pass

    return best_points
