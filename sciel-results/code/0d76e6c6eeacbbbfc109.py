# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time

def _score(points):
    """Compute (d_min / d_max)^2 for a set of points."""
    if len(points) < 2:
        return 0.0
    dists = pdist(points, metric='euclidean')
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0 or np.isnan(d_min) or np.isnan(d_max):
        return 0.0
    return (d_min / d_max) ** 2

def _generate_icosahedron_symmetry(params):
    """
    Generates 14 points in 3D with a specific symmetry structure.
    
    Structure:
    - 12 points forming a distorted icosahedron (or similar ring structure)
    - 2 points on the z-axis (poles)
    
    Parameters:
    p[0]: radius of the 12-point ring
    p[1]: z-offset of the 12-point ring (height)
    p[2]: angular twist or radial modulation amplitude
    p[3]: z-coordinate of the poles
    """
    r_ring = params[0]
    z_ring = params[1]
    twist = params[2]
    z_pole = params[3]
    
    points = []
    
    # 2 poles
    points.append([0.0, 0.0, z_pole])
    points.append([0.0, 0.0, -z_pole])
    
    # 12 points in a ring. 
    # To maintain high symmetry and avoid degenerate overlaps, we use a 
    # structure similar to an icosahedron's equatorial belt but modulated.
    # An icosahedron has 12 vertices. We can place 10 in two rings of 5 and 2 poles, 
    # or 12 in one complex ring? 
    # Actually, a better symmetric 14-point candidate is:
    # 8 vertices of a cube + 6 face centers (truncated octahedron vertices? No, that's 24).
    # Let's try: 12 points on a sphere + 2 poles.
    # The 12 points can be arranged as two pentagons (5 each) rotated by 36 degrees.
    
    # Pentagon 1 (z = z_ring)
    for i in range(5):
        theta = 2 * np.pi * i / 5.0
        x = r_ring * np.cos(theta)
        y = r_ring * np.sin(theta)
        points.append([x, y, z_ring])
        
    # Pentagon 2 (z = -z_ring), rotated by 36 degrees (pi/5)
    for i in range(5):
        theta = 2 * np.pi * i / 5.0 + np.pi / 5.0
        x = r_ring * np.cos(theta)
        y = r_ring * np.sin(theta)
        points.append([x, y, -z_ring])
        
    # We have 2 + 5 + 5 = 12 points. We need 14.
    # Let's add 2 more points? Or change the structure.
    
    # Alternative Structure: 
    # 14 points. 
    # Maybe: 2 poles + 4 in one ring + 4 in another ring + 4 in a third ring?
    # Or: 2 poles + 6 in one ring + 6 in another ring?
    
    # Let's try: 2 poles + 12 points arranged as 3 rings of 4 points each (square rings).
    # This has high symmetry (D4h).
    
    # Reset points
    points = []
    
    # 2 poles
    points.append([0.0, 0.0, z_pole])
    points.append([0.0, 0.0, -z_pole])
    
    # 3 rings of 4 points
    # Ring 1: z = h1, radius r1
    # Ring 2: z = 0, radius r2
    # Ring 3: z = -h1, radius r1
    # But we only have 4 params. Let's assume r1=r2=r_ring and h1 = z_ring.
    # And maybe twist the rings?
    
    # Let's use:
    # p[0]: r (radius for all rings)
    # p[1]: h (height of outer rings, middle is 0)
    # p[2]: twist (angle offset for middle ring relative to outer)
    # p[3]: z_pole
    
    r = r_ring
    h = z_ring
    twist = twist
    
    # Ring 1 (top): z = h
    for i in range(4):
        theta = np.pi / 4 + i * np.pi / 2  # Start at 45 deg for symmetry
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append([x, y, h])
        
    # Ring 2 (middle): z = 0, twisted by 'twist'
    for i in range(4):
        theta = np.pi / 4 + i * np.pi / 2 + twist
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append([x, y, 0.0])
        
    # Ring 3 (bottom): z = -h
    for i in range(4):
        theta = np.pi / 4 + i * np.pi / 2
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append([x, y, -h])
        
    pts = np.array(points)
    # Normalize to unit sphere to handle scale invariance in the objective
    norms = np.linalg.norm(pts, axis=1)
    norms[norms == 0] = 1.0
    pts = pts / norms[:, np.newaxis]
    
    return pts

def _objective_symmetry(params):
    """
    Objective function for the symmetry-constrained optimization.
    """
    points = _generate_icosahedron_symmetry(params)
    s = _score(points)
    return -s

def min_max_dist_dim3_14() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580  # Safety margin
    
    best_score = -np.inf
    best_params_sym = None
    best_points = None
    
    # Initial guesses for the symmetry parameters (r, h, twist, z_pole)
    # We search in the space of the 4 parameters.
    # Since the points are normalized, the scale of r, h, z_pole doesn't matter, 
    # only their relative ratios. We can fix one, e.g., z_pole = 1.0, 
    # or just let the optimizer find the relative values. 
    # However, normalization inside _generate_icosahedron_symmetry makes the objective 
    # invariant to scaling of all params. This can cause issues with gradient-based methods 
    # if the manifold is curved. 
    # Better: Fix z_pole = 1.0 and optimize r, h, twist.
    # Let's modify _generate_icosahedron_symmetry to take 3 params and fix z_pole=1.
    
    # Actually, let's just use 4 params and let SLSQP handle it. 
    # To avoid the scale invariance singularity, we can penalize large norms or 
    # simply start with reasonable values.
    
    initial_guesses_sym = []
    
    # Guess 1: Regular Icosahedron-like
    # In a regular icosahedron, the "rings" aren't squares. 
    # But let's try a cube-like structure first.
    # Cube: 8 corners + 6 centers. 
    # My symmetry generator produces: 2 poles + 3 squares.
    # This is a 14-point structure: 2 poles + 12 square ring points.
    # This is actually a "tubular" or "cylindrical" symmetry structure.
    
    # Let's try values for a cube-like projection.
    # If we want a cube, we might need a different symmetry.
    # But let's see if this 14-point structure can be good.
    
    # r ~ 1, h ~ 1, twist ~ 0, z_pole ~ 1
    initial_guesses_sym.append([1.0, 1.0, 0.0, 1.0])
    initial_guesses_sym.append([1.0, 0.5, 0.0, 1.0])
    initial_guesses_sym.append([1.0, 1.5, 0.0, 1.0])
    initial_guesses_sym.append([1.0, 1.0, np.pi/4, 1.0]) # Twisted
    initial_guesses_sym.append([0.8, 1.2, 0.0, 1.0])
    initial_guesses_sym.append([1.2, 0.8, 0.0, 1.0])
    
    # Random guesses
    np.random.seed(42)
    for _ in range(20):
        r = np.random.uniform(0.5, 1.5)
        h = np.random.uniform(0.0, 2.0)
        twist = np.random.uniform(0, np.pi/2)
        z_pole = np.random.uniform(0.5, 2.0)
        initial_guesses_sym.append([r, h, twist, z_pole])
        
    # Optimization Loop
    for i, guess in enumerate(initial_guesses_sym):
        if time.time() > deadline - 20:
            break
            
        try:
            # SLSQP is good for small dimensional problems with constraints.
            # We don't have explicit constraints other than bounds.
            # Bounds: r > 0, h >= 0, z_pole > 0. Twist is periodic.
            bounds = [
                (0.1, 5.0),   # r
                (0.0, 5.0),   # h
                (0.0, np.pi/2), # twist (symmetry allows 0 to pi/2)
                (0.1, 5.0)    # z_pole
            ]
            
            res = minimize(
                _objective_symmetry, 
                guess, 
                method='SLSQP', 
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            
            score = -res.fun
            if score > best_score:
                best_score = score
                best_params_sym = res.x.copy()
                best_points = _generate_icosahedron_symmetry(best_params_sym)
                
        except Exception:
            continue
            
    # If the symmetry-constrained search didn't find a good solution, 
    # or if we have time, we can try a generic local polish on the best point found
    # using the full 42D space, starting from the symmetric configuration.
    
    if best_points is not None and time.time() < deadline - 30:
        try:
            # Convert best_points to flat array for generic optimizer
            flat_params = best_points.flatten()
            
            def _objective_flat(params):
                points = params.reshape(14, 3)
                # Normalize
                norms = np.linalg.norm(points, axis=1)
                norms[norms == 0] = 1.0
                points = points / norms[:, np.newaxis]
                return -_score(points)
                
            # Use L-BFGS-B for the polish
            res_polish = minimize(
                _objective_flat, 
                flat_params, 
                method='L-BFGS-B', 
                options={'maxiter': 500, 'ftol': 1e-14}
            )
            
            score_polish = -res_polish.fun
            if score_polish > best_score:
                best_score = score_polish
                pts_polish = res_polish.x.reshape(14, 3)
                # Normalize final points
                norms = np.linalg.norm(pts_polish, axis=1)
                norms[norms == 0] = 1.0
                best_points = pts_polish / norms[:, np.newaxis]
        except Exception:
            pass

    # Fallback if everything failed
    if best_points is None:
        # Return a basic icosahedron + 2 poles
        phi = (1 + np.sqrt(5)) / 2
        ico = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ])
        ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
        extra = np.array([[0, 0, 1], [0, 0, -1]])
        best_points = np.vstack([ico, extra])
        best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
        
    # Ensure finite and correct shape
    best_points = np.nan_to_num(best_points, nan=0.0, posinf=0.0, neginf=0.0)
    if best_points.shape != (14, 3):
        # Emergency reset
        best_points = np.eye(14, 3) # Invalid shape? No, (14,3). 
        # Actually np.eye(14,3) is (14,3). But points are not on sphere.
        # Let's just return the icosahedron fallback again if shape is wrong.
        phi = (1 + np.sqrt(5)) / 2
        ico = np.array([
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
        ])
        ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
        extra = np.array([[0, 0, 1], [0, 0, -1]])
        best_points = np.vstack([ico, extra])
        best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)

    return best_points

# EVOLVE-BLOCK-END
