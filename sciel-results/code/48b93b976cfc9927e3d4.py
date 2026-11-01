import numpy as np
import time
from scipy.optimize import minimize

N = 14
D = 3

def _pairwise_dist_sq(x):
    # x is shape (N, D)
    # Compute squared distances
    # d^2 = ||x_i||^2 + ||x_j||^2 - 2 x_i . x_j
    sq_norms = np.sum(x**2, axis=1)
    dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * (x @ x.T)
    # Zero out diagonal
    np.fill_diagonal(dist_sq, 0)
    return dist_sq

def _objective(x):
    points = x.reshape(N, D)
    dist_sq = _pairwise_dist_sq(points)
    
    # Get off-diagonal elements
    mask = ~np.eye(N, dtype=bool)
    d_sq = dist_sq[mask]
    
    min_d_sq = np.min(d_sq)
    max_d_sq = np.max(d_sq)
    
    # Avoid division by zero
    if min_d_sq < 1e-12:
        return 1e12
        
    ratio = max_d_sq / min_d_sq
    
    # Penalty to keep points on unit sphere
    # This helps stabilize the optimization and prevents collapse/expansion
    norms = np.sum(points**2, axis=1)
    penalty = np.sum((norms - 1.0)**2) * 0.01
    
    return ratio + penalty

def _initial_icosahedron_plus_2():
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    
    # Normalize to unit sphere
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, None]
    
    # Add 2 points. 
    # The icosahedron has 5-fold symmetry axes through opposite vertices.
    # Let's pick the first vertex (0, 1, phi) and its opposite (0, -1, -phi).
    # We will place the new points along this axis, slightly inside or outside?
    # Or just place them at the poles of a rotated frame?
    # Let's place them at (0,0,1) and (0,0,-1) after rotating the icosahedron?
    # Actually, let's just take the first vertex v0 and its opposite -v0.
    # The "poles" of the icosahedron in this orientation are v0 and -v0.
    # If we add points at v0 and -v0, they coincide.
    # So we add points slightly displaced?
    # Better: Add points at the centers of the two faces adjacent to the "top" and "bottom"?
    # Simplest robust start: Random perturbation of the 12 verts + 2 random points on sphere.
    
    # Let's try placing the 2 extra points at the poles of the z-axis, 
    # and rotate the icosahedron such that it has symmetry around z-axis?
    # The icosahedron does not have a single z-axis of 5-fold symmetry in standard coords?
    # It has 6 axes of 5-fold symmetry.
    # Let's just use the 12 vertices and 2 points at (0,0,1) and (0,0,-1).
    # This breaks symmetry but provides a good starting point.
    
    p1 = np.array([0.0, 0.0, 1.0])
    p2 = np.array([0.0, 0.0, -1.0])
    
    all_points = np.vstack([verts, p1, p2])
    
    # Add small random noise to break symmetry and help escape local minima
    noise = np.random.normal(0, 0.01, all_points.shape)
    all_points = all_points + noise
    
    # Re-normalize to unit sphere
    norms = np.linalg.norm(all_points, axis=1)
    all_points = all_points / norms[:, None]
    
    return all_points.flatten()

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    best_score = np.inf
    best_points = None
    
    # Generate several initial guesses
    guesses = []
    
    # Guess 1: Icosahedron + 2 poles
    guesses.append(_initial_icosahedron_plus_2())
    
    # Guess 2-6: Random points on sphere
    for _ in range(5):
        pts = np.random.randn(N, D)
        pts = pts / np.linalg.norm(pts, axis=1)[:, None]
        guesses.append(pts.flatten())
        
    # Try a grid-like start but projected to sphere? No, just random.
    
    for i, guess in enumerate(guesses):
        if time.time() > deadline - 10.0:
            break
            
        try:
            res = minimize(
                _objective, 
                guess, 
                method='Nelder-Mead', 
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            
            if res.success or res.fun < best_score:
                # Check if distinct
                pts = res.x.reshape(N, D)
                dist_sq = _pairwise_dist_sq(pts)
                min_d = np.min(dist_sq[~np.eye(N, dtype=bool)])
                if min_d > 1e-8:
                    best_score = res.fun
                    best_points = res.x.copy()
                    
        except Exception:
            pass
            
    if best_points is None:
        # Fallback to seed
        side = int(np.ceil(N ** (1 / D)))
        import itertools
        grid = list(itertools.product(range(side), repeat=D))[:N]
        return np.array(grid, dtype=float)
        
    return best_points.reshape(N, D)
