import numpy as np
import time
from scipy.optimize import minimize

N = 14
D = 3

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s

    # Base structures
    # Cube vertices (side 2, centered at 0)
    cube_verts = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
    ], dtype=float)
    
    # Octahedron vertices (on axes)
    oct_verts = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ], dtype=float)

    def rotate_z(points, angle):
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        return points @ R.T

    def get_points(s, theta):
        # Scale octahedron by s
        oct_scaled = oct_verts * s
        # Rotate octahedron by theta
        oct_rot = rotate_z(oct_scaled, theta)
        # Concatenate
        return np.vstack([cube_verts, oct_rot])

    def objective(params):
        s, theta = params
        if s <= 0:
            return 1e10
        pts = get_points(s, theta)
        # Compute pairwise squared distances
        # Using broadcasting: (n, 1, 3) - (1, n, 3) -> (n, n, 3)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        
        # Exclude self-distances (diagonal)
        np.fill_diagonal(dist_sq, np.inf)
        
        min_d_sq = np.min(dist_sq)
        max_d_sq = np.max(dist_sq)
        
        if min_d_sq < 1e-12:
            return 1e10
            
        ratio = max_d_sq / min_d_sq
        return ratio

    # Optimization Strategy
    # 1. Grid search for initial guesses
    # 2. Local refinement using L-BFGS-B

    best_val = np.inf
    best_params = None

    # Grid search over s and theta
    # s is the distance from origin for octahedron vertices.
    # Cube vertices are at distance sqrt(3) ~ 1.732.
    # Min distance in cube is 2. Max is 2*sqrt(3) ~ 3.464.
    # If s is too small, octahedron points are close to each other (dist 2s) and to cube points.
    # If s is too large, octahedron points are far.
    # Let's try s in [0.5, 2.0] and theta in [0, pi/2]
    
    s_vals = np.linspace(0.5, 2.5, 20)
    theta_vals = np.linspace(0, np.pi/2, 10)

    for s in s_vals:
        for theta in theta_vals:
            val = objective([s, theta])
            if val < best_val:
                best_val = val
                best_params = [s, theta]

    # Local refinement from best grid point and a few others
    # Also try random starts if time permits
    starts = [best_params]
    
    # Add some random starts
    np.random.seed(42)
    for _ in range(5):
        s_rand = np.random.uniform(0.5, 2.5)
        theta_rand = np.random.uniform(0, np.pi/2)
        starts.append([s_rand, theta_rand])

    # Also try specific symmetric points
    # s = 1 (octahedron inscribed in cube? No, cube vertices at 1, oct at 1. Distance from (1,1,1) to (1,0,0) is sqrt(1+1+0)=sqrt(2).)
    starts.append([1.0, 0.0])
    starts.append([1.0, np.pi/4])
    starts.append([1.5, 0.0])
    starts.append([0.8, np.pi/4])

    for start in starts:
        if time.time() > deadline - 10:
            break
            
        try:
            res = minimize(objective, start, method='L-BFGS-B', bounds=[(0.1, 5.0), (0.0, np.pi/2)], options={'maxiter': 500})
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass

    # Final polish with tighter tolerance if time remains
    if time.time() < deadline - 5 and best_params is not None:
        try:
            res = minimize(objective, best_params, method='L-BFGS-B', bounds=[(0.1, 5.0), (0.0, np.pi/2)], options={'maxiter': 1000, 'gtol': 1e-8})
            if res.success and res.fun < best_val:
                best_val = res.fun
                best_params = res.x
        except Exception:
            pass

    if best_params is None:
        # Fallback to grid best or simple structure
        return np.array(get_points(1.0, 0.0))
        
    s, theta = best_params
    pts = get_points(s, theta)
    
    # Ensure distinct points (should be by construction unless s=0, but we bounded s>0.1)
    return pts
