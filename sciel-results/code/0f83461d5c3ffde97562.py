import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import differential_evolution, minimize

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the (x,y) coordinates.

    """
    n = 16
    d = 2
    start_time = time.time()
    deadline = start_time + 590  # Safety margin

    def get_score(points):
        """Calculate the score R = (d_min / d_max)^2"""
        if points.shape != (n, d):
            return 0.0
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        if d_max == 0:
            return 0.0
        return (d_min / d_max) ** 2

    def objective(flat_points):
        """Objective function for minimization (negative score)"""
        points = flat_points.reshape(n, d)
        score = get_score(points)
        return -score

    # 1. Generate initial candidates
    candidates = []

    # Candidate 1: Hexagonal lattice subset
    # Center at (0,0). Ring 1 (6 pts), Ring 2 (12 pts). We need 16 total.
    # 1 + 6 + 9 = 16.
    hex_pts = np.array([[0.0, 0.0]])
    for j in range(-2, 3):
        for i in range(-2, 3):
            if i == 0 and j == 0:
                continue
            # Hexagonal coordinates to Cartesian
            x = i * np.sqrt(3) + j * np.sqrt(3) / 2
            y = j * 1.5
            hex_pts = np.vstack([hex_pts, [x, y]])
    # Take first 16 points (center + 15 others)
    # Actually, let's just take the 16 closest to the center from a larger hex grid
    # Or just construct a specific 16-point hex cluster.
    # A better hex cluster: 1 center, 6 ring 1, 9 ring 2.
    # Let's just generate a dense hex grid and pick the 16 closest to origin.
    grid_hex = []
    for j in range(-3, 4):
        for i in range(-3, 4):
            x = i * np.sqrt(3) + j * np.sqrt(3) / 2
            y = j * 1.5
            grid_hex.append([x, y])
    grid_hex = np.array(grid_hex)
    dists_to_center = np.sum(grid_hex**2, axis=1)
    indices = np.argsort(dists_to_center)[:n]
    hex_candidate = grid_hex[indices]
    candidates.append(hex_candidate)

    # Candidate 2: Square grid 4x4
    sq = np.linspace(-1.5, 1.5, 4)
    xx, yy = np.meshgrid(sq, sq)
    sq_candidate = np.column_stack((xx.ravel(), yy.ravel()))
    candidates.append(sq_candidate)

    # Candidate 3: Random points on a circle (uniform angle)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    rad_candidate = np.column_stack((np.cos(angles), np.sin(angles)))
    candidates.append(rad_candidate)

    # Candidate 4: Random points
    np.random.seed(42)
    rand_candidate = np.random.randn(n, d) * 2
    candidates.append(rand_candidate)

    best_points = None
    best_score = -np.inf

    # Evaluate initial candidates
    for cand in candidates:
        score = get_score(cand)
        if score > best_score:
            best_score = score
            best_points = cand.copy()

    # 2. Global Optimization using Differential Evolution
    # Bounds: Let's assume points are within [-10, 10]
    bounds = [(-10, 10)] * (n * d)
    
    # Check time
    if time.time() < deadline - 100:
        try:
            # DE settings
            # popsize=20, maxiter=50 is a quick start. 
            # We can run multiple short DEs or one longer one.
            # Let's try one moderate DE run.
            res = differential_evolution(
                objective, 
                bounds, 
                maxiter=100, 
                popsize=20, 
                tol=0.01, 
                seed=42,
                polish=False # We will polish manually
            )
            
            if res.fun < best_score * -1: # res.fun is -score
                best_score = -res.fun
                best_points = res.x.reshape(n, d).copy()
        except Exception:
            pass

    # 3. Local Polish using Nelder-Mead
    # Nelder-Mead is robust for non-smooth functions.
    # We will polish the best point found so far.
    
    if time.time() < deadline - 10:
        try:
            # Use the best points as initial guess
            res_nm = minimize(
                objective, 
                best_points.ravel(), 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            
            score_nm = -res_nm.fun
            if score_nm > best_score:
                best_score = score_nm
                best_points = res_nm.x.reshape(n, d).copy()
        except Exception:
            pass

    # 4. Final Polish with Powell (often faster/cleaner for this)
    if time.time() < deadline - 5:
        try:
            res_pw = minimize(
                objective, 
                best_points.ravel(), 
                method='Powell', 
                options={'maxiter': 2000, 'xtol': 1e-10, 'ftol': 1e-12}
            )
            
            score_pw = -res_pw.fun
            if score_pw > best_score:
                best_score = score_pw
                best_points = res_pw.x.reshape(n, d).copy()
        except Exception:
            pass

    # Ensure the result is a valid numpy array of shape (16, 2)
    if best_points is None:
        # Fallback to random if everything failed
        best_points = np.random.randn(n, d)
        
    return best_points

# EVOLVE-BLOCK-END
