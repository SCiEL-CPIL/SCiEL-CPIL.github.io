# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of 13 points to maximize the area of the
    smallest triangle formed by these points relative to the convex hull area.
    """
    n = 13
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    # Helper function to calculate the score (higher is better)
    def calculate_score(points):
        try:
            # Check for NaN/Inf
            if not np.all(np.isfinite(points)):
                return -np.inf
            # Check for collinearity / hull validity
            hull = ConvexHull(points)
            a_hull = hull.volume
            if a_hull <= 1e-10:
                return -np.inf
            
            # Calculate min triangle area
            # n=13, so 286 triangles. Vectorized approach for speed.
            # Get indices
            idx = np.indices((n, n, n))
            # We need i < j < k
            # This creates a 13x13x13 array, which is fine.
            # But filtering is easier with loops for n=13, or using comb.
            # Let's use a loop for clarity and sufficient speed, or vectorized comb.
            # Vectorized:
            # p1, p2, p3
            # Area = 0.5 * | (x2-x1)(y3-y1) - (x3-x1)(y2-y1) |
            
            min_area = np.inf
            # Since n is small, a simple loop is often faster than complex vectorization overhead in Python for this specific case?
            # Actually, 286 iterations is nothing.
            for i in range(n):
                for j in range(i+1, n):
                    for k in range(j+1, n):
                        p1 = points[i]
                        p2 = points[j]
                        p3 = points[k]
                        # Cross product
                        area = 0.5 * abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1]))
                        if area < min_area:
                            min_area = area
                            
            return min_area / a_hull
        except Exception:
            return -np.inf

    # Objective function for scipy.optimize.minimize (minimizes, so we negate score)
    def objective(x):
        points = x.reshape((n, 2))
        score = calculate_score(points)
        if score == -np.inf:
            return 1e6  # Large penalty
        return -score

    # Initialization Strategies
    def init_regular_polygon():
        angles = np.linspace(0, 2*np.pi, n, endpoint=False)
        radius = 1.0
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        return np.column_stack((x, y)).flatten()

    def init_triangular_lattice():
        # Create a triangular lattice of side 5 (15 points) and pick 13
        # Or side 4 (10 points) + 3?
        # Let's generate a grid of points in a triangle
        # Side length 4 has 10 points. Side length 5 has 15.
        # We'll use side 5 and remove 2 points that are likely to cause small triangles.
        # In a triangular lattice, the smallest triangles are the unit triangles.
        # Removing interior points might help?
        # Actually, let's just use the first 13 points of a sorted lattice?
        # Better: Use a specific known good configuration if possible, or random perturbation of lattice.
        
        # Generate lattice points
        pts = []
        for i in range(5): # 0 to 4
            for j in range(5 - i): # 0 to 4-i
                x = i + 0.5 * j
                y = (np.sqrt(3) / 2) * j
                pts.append([x, y])
        pts = np.array(pts)
        # This gives 15 points.
        # We need 13.
        # Remove the two points with the highest "local density" or just random 2?
        # Let's remove the two center-most points of the 15?
        # Center of mass:
        com = np.mean(pts, axis=0)
        dists = np.linalg.norm(pts - com, axis=1)
        # The points closest to the center are the ones most likely to form small triangles with neighbors?
        # Actually, in a Heilbronn problem, we want to *increase* min area.
        # The unit triangles have area A. If we remove points, we might remove vertices of small triangles,
        # but we also lose points.
        # Let's just take 13 random points from this lattice? No, that's not structured.
        # Let's remove the 2 points that are "redundant" in a convex hull sense? No, they are interior.
        # Let's remove the 2 points with the smallest distance to their nearest neighbor?
        # This is getting complicated.
        # Alternative: Use a regular 13-gon as the primary init, and a perturbed triangular grid as secondary.
        
        # For now, return a perturbed triangular lattice of 13 points.
        # We'll pick 13 points from the 15 by removing the 2 with the smallest "space" around them?
        # Simpler: Just return the first 13 points of the lattice?
        # The lattice is ordered by i then j.
        # i=0: 5 pts (top edge)
        # i=1: 4 pts
        # i=2: 3 pts
        # i=3: 2 pts
        # i=4: 1 pt (bottom vertex)
        # Total 15.
        # Removing the bottom vertex (i=4, j=0) and one from the top?
        # Let's just return the 13 points excluding the two "corners" of the lattice triangle?
        # The corners are (0,0), (4,0), (0,4).
        # Let's remove (4,0) and (0,4)? That breaks symmetry.
        
        # Let's try a different init: Regular 13-gon is safer and often close to optimal for small n.
        # We will rely on the optimizer to push it towards the triangular structure if beneficial.
        
        # For the second init, let's use a "nested" structure.
        # 1 center, 3 around it, 6 around that, 3 more?
        # This is hard to code generically.
        
        # Let's just return a random perturbation of the regular polygon for the second init.
        rng = np.random.default_rng(42)
        reg = init_regular_polygon().reshape(n, 2)
        reg += rng.normal(0, 0.1, (n, 2))
        return reg.flatten()

    # List of initial guesses
    inits = []
    inits.append(init_regular_polygon())
    
    # Add a few random perturbations of the regular polygon
    rng = np.random.default_rng(123)
    for _ in range(4):
        base = init_regular_polygon().reshape(n, 2)
        base += rng.normal(0, 0.2, (n, 2))
        inits.append(base.flatten())
        
    # Add a triangular lattice init (randomly selecting 13 from 15)
    # Generate lattice
    pts_lattice = []
    for i in range(5):
        for j in range(5 - i):
            x = i + 0.5 * j
            y = (np.sqrt(3) / 2) * j
            pts_lattice.append([x, y])
    pts_lattice = np.array(pts_lattice)
    # Select 13 points
    # We'll remove the 2 points that are "most central" to the convex hull of the lattice?
    # Or just remove the 2 with the smallest distance to the centroid?
    com = np.mean(pts_lattice, axis=0)
    dists = np.linalg.norm(pts_lattice - com, axis=1)
    # Remove the 2 with smallest distance (most central)
    indices_to_remove = np.argsort(dists)[:2]
    mask = np.ones(15, dtype=bool)
    mask[indices_to_remove] = False
    pts_lattice_13 = pts_lattice[mask]
    inits.append(pts_lattice_13.flatten())
    
    best_score = -np.inf
    best_points = None
    
    for i, init in enumerate(inits):
        if time.time() > deadline - 10:
            break
            
        try:
            # Nelder-Mead is robust for non-smooth functions
            res = minimize(objective, init, method='Nelder-Mead', 
                           options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True})
            
            if res.success or res.fun < -best_score:
                curr_score = -res.fun
                if curr_score > best_score:
                    best_score = curr_score
                    best_points = res.x.reshape(n, 2)
        except Exception as e:
            pass

    # If no optimization succeeded, return the best initial guess
    if best_points is None:
        # Fallback to regular polygon
        best_points = init_regular_polygon().reshape(n, 2)
        best_score = calculate_score(best_points)
        
    # Final verification and cleanup
    # Ensure points are finite
    if not np.all(np.isfinite(best_points)):
         best_points = init_regular_polygon().reshape(n, 2)
         
    return best_points

# EVOLVE-BLOCK-END
