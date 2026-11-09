# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the area of the
    smallest triangle formed by these points relative to the convex hull area.
    """
    start_time = time.time()
    deadline = 630.0 # 630s limit, keep margin
    
    # 1. Generate seed points from triangular lattice
    # Lattice vectors
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3)/2])
    
    # Indices for the 13 points (i, j) in lattice coordinates
    # 1 center, 6 at norm^2=1, 6 at norm^2=3
    indices = [
        (0, 0),
        (1, 0), (0, 1), (1, -1), (-1, 0), (0, -1), (-1, 1),
        (1, 1), (-1, -1), (2, -1), (-2, 1), (1, -2), (-1, 2)
    ]
    
    points = np.array([i * v1 + j * v2 for i, j in indices])
    
    # Center and scale to unit area? 
    # The ratio is scale invariant, but numerical stability might benefit from scaling.
    # Let's center at origin.
    points -= np.mean(points, axis=0)
    
    # Objective function to minimize: -log(A_min) + log(A_hull)
    # Or minimize A_hull / A_min
    # Using log is safer.
    # Score = A_min / A_hull. Maximize Score.
    # Minimize -log(Score) = -log(A_min) + log(A_hull)
    
    def objective(params):
        if time.time() > deadline - 5.0:
            return 1e10 # Penalty if timeout
            
        p = params.reshape(13, 2)
        
        # Check for degeneracy
        if not np.all(np.isfinite(p)):
            return 1e10
            
        # Calculate areas of all triangles
        # 286 triangles
        # Vectorized calculation
        # p has shape (13, 2)
        # We need area of triangle (i, j, k)
        # Area = 0.5 * |det(p_j - p_i, p_k - p_i)|
        
        # To avoid O(N^3) python loops, we can use numpy broadcasting, 
        # but 13 is small, so a loop over indices might be fast enough or we can vectorize.
        
        # Let's vectorize the area calculation.
        # p_i, p_j, p_k
        # We can create arrays of indices
        idx = np.arange(13)
        i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
        # Filter i < j < k
        mask = (i_idx < j_idx) & (j_idx < k_idx)
        i_idx = i_idx[mask]
        j_idx = j_idx[mask]
        k_idx = k_idx[mask]
        
        pi = p[i_idx]
        pj = p[j_idx]
        pk = p[k_idx]
        
        # Vectors
        v1 = pj - pi
        v2 = pk - pi
        
        # Cross product in 2D: x1*y2 - y1*x2
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        areas = 0.5 * np.abs(cross)
        
        A_min = np.min(areas)
        
        if A_min < 1e-12:
            return 1e10
            
        # Convex Hull Area
        try:
            hull = ConvexHull(p)
            A_hull = hull.volume
        except Exception:
            return 1e10
            
        if A_hull < 1e-12:
            return 1e10
            
        # Return -log(A_min) + log(A_hull) = log(A_hull/A_min)
        return np.log(A_hull) - np.log(A_min)

    # Initial points flattened
    x0 = points.flatten()
    
    # Optimization
    best_score = 1e10
    best_points = points
    
    # Try Nelder-Mead
    # Maxiter needs to be reasonable. 
    # We have ~600s. 
    # Each eval is fast (vectorized area calc).
    # Let's estimate eval time. 13 points, 286 triangles. Vectorized ops are fast.
    # Maybe 0.1ms per eval? 
    # 10000 evals = 1s.
    # We can afford many iterations.
    
    try:
        # Use Nelder-Mead
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 5000, 'xatol': 1e-5, 'fatol': 1e-5})
        if res.fun < best_score:
            best_score = res.fun
            best_points = res.x.reshape(13, 2)
            
        # If time remains, try a few random restarts or different initializations?
        # The lattice solution is likely global optimum or very close.
        # But maybe a slight perturbation helps?
        
    except Exception as e:
        pass

    # Final check and return
    if time.time() > deadline - 10.0:
        # Return best found so far
        return best_points

    # Verify the best points
    # Re-calculate score to be sure
    p = best_points
    idx = np.arange(13)
    i_idx, j_idx, k_idx = np.meshgrid(idx, idx, idx, indexing='ij')
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_idx = i_idx[mask]
    j_idx = j_idx[mask]
    k_idx = k_idx[mask]
    
    pi = p[i_idx]
    pj = p[j_idx]
    pk = p[k_idx]
    
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    A_min = np.min(areas)
    
    try:
        hull = ConvexHull(p)
        A_hull = hull.volume
    except:
        # Fallback to seed if hull fails
        return points

    if A_min > 0 and A_hull > 0:
         return p
    else:
         return points

# EVOLVE-BLOCK-END
