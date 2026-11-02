"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

# Pre-compute triangle indices for speed
def get_triangle_indices(n):
    idx = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                idx.append((i, j, k))
    return np.array(idx)

TRI_IDX = get_triangle_indices(N)
NUM_TRI = len(TRI_IDX)

def min_triangle_area(points):
    """
    Calculate the minimum area of any triangle formed by the points.
    points: (n, 2) array
    """
    p0 = points[TRI_IDX[:, 0]]
    p1 = points[TRI_IDX[:, 1]]
    p2 = points[TRI_IDX[:, 2]]
    v1 = p1 - p0
    v2 = p2 - p0
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def convex_hull_area(points):
    """
    Calculate the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-10

def score(points):
    """
    Calculate the score: min_triangle_area / convex_hull_area
    """
    a_min = min_triangle_area(points)
    a_hull = convex_hull_area(points)
    if a_hull < 1e-10:
        return 0.0
    return a_min / a_hull

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin of 5s before 150s kill
    
    best_score = -1.0
    best_points = None
    
    # Strategy: Symmetric parameterization
    # 6 outer points forming a convex hexagon (regular or slightly perturbed)
    # 6 inner points on the symmetry axes (radii r_i)
    # 1 center point (optional, but we have 13 points: 6 outer + 6 inner + 1 center = 13)
    
    # The symmetry group D6 (hexagon) has 6 axes.
    # Outer vertices at angles: 0, 60, 120, 180, 240, 300 degrees.
    # Inner points at angles: 30, 90, 150, 210, 270, 330 degrees (bisecting the outer angles)
    # This ensures convexity of the hull (outer points) and symmetry.
    
    def generate_symmetric_points(params):
        """
        params: array of 7 floats
        params[0]: outer radius R
        params[1..6]: inner radii r0..r5
        """
        R = params[0]
        r_inners = params[1:7]
        
        # Outer points
        angles_outer = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        outer_pts = np.column_stack([R * np.cos(angles_outer), R * np.sin(angles_outer)])
        
        # Inner points (offset by 30 degrees = pi/6)
        angles_inner = angles_outer + np.pi / 6
        # Ensure radii are positive and less than R to stay inside
        r_inners = np.clip(r_inners, 0.0, R * 0.9)
        inner_pts = np.column_stack([r_inners * np.cos(angles_inner), r_inners * np.sin(angles_inner)])
        
        # Center point
        center = np.array([[0.0, 0.0]])
        
        # Combine: 6 outer + 6 inner + 1 center = 13 points
        # Order matters for index consistency, but score is invariant to permutation
        points = np.vstack([outer_pts, inner_pts, center])
        return points

    # Grid search over R and inner radii
    # R is the scale, but score is scale-invariant (area/area). 
    # However, the ratio of inner to outer radii matters.
    # Let's fix R=1.0 and vary the inner radii.
    
    # We have 6 inner radii to optimize.
    # Due to symmetry, we might expect them to be equal or follow a pattern.
    # Let's try a grid search over a few symmetric configurations first, then local polish.
    
    # Configuration 1: All inner radii equal (r)
    # Configuration 2: Alternating radii (r1, r2, r1, r2, r1, r2)
    # Configuration 3: General (r0..r5)
    
    # Let's do a coarse grid search over (r1, r2) for alternating, and (r) for equal.
    # And a general 6D grid is too big.
    
    # Instead, we will use the symmetric parameterization as a starting point for local optimization.
    # This is much better than random starts because the landscape is smoother and more constrained.
    
    # Initial guesses for the 6 inner radii (normalized to R=1)
    # Guess 1: All equal, r = 0.5
    # Guess 2: All equal, r = 0.3
    # Guess 3: All equal, r = 0.7
    # Guess 4: Alternating 0.3, 0.7
    # Guess 5: Alternating 0.5, 0.5 (same as 1)
    # Guess 6: Random within [0.1, 0.9]
    
    rng = np.random.default_rng(42)
    
    inits = []
    inits.append(np.array([1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]))
    inits.append(np.array([1.0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]))
    inits.append(np.array([1.0, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]))
    inits.append(np.array([1.0, 0.3, 0.7, 0.3, 0.7, 0.3, 0.7]))
    inits.append(np.array([1.0, 0.7, 0.3, 0.7, 0.3, 0.7, 0.3]))
    
    for _ in range(20):
        r_rand = rng.uniform(0.1, 0.9, 6)
        inits.append(np.concatenate([[1.0], r_rand]))
        
    def objective(params):
        pts = generate_symmetric_points(params)
        s = score(pts)
        return -s
    
    # Optimize each initialization
    for init in inits:
        if time.time() > deadline - 10.0:
            break
        try:
            # Use Nelder-Mead for the 7D parameter space
            # This is much faster and more reliable than 26D coordinate space
            res = minimize(objective, init, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8,
                                    'adaptive': True})
            best_params = res.x
            pts = generate_symmetric_points(best_params)
            s = score(pts)
            
            if s > best_score:
                best_score = s
                best_points = pts
        except Exception:
            pass

    # If we found a good point, try to break symmetry slightly?
    # No, the task allows any points. But the symmetric solution is a strong candidate.
    # Let's do a final polish in the full 26D space starting from the best symmetric point
    # to see if a slight asymmetry helps.
    
    if best_points is not None and time.time() < deadline - 5.0:
        def objective_full(x):
            pts = x.reshape(-1, 2)
            return -score(pts)
            
        try:
            x0 = best_points.flatten()
            res = minimize(objective_full, x0, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10,
                                    'adaptive': True})
            pts = res.x.reshape(-1, 2)
            s = score(pts)
            if s > best_score:
                best_score = s
                best_points = pts
        except Exception:
            pass

    # Fallback
    if best_points is None:
        # Regular polygon + center
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = score(best_points)
    
    return best_points

# EVOLVE-BLOCK-END
