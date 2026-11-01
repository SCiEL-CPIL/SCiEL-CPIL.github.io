import time
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import differential_evolution, minimize

N = 14
D = 3
DEADLINE = 145.0
_t0 = None

def _true_score(pts):
    """Calculate the squared ratio of max to minimum pairwise distance."""
    d = cdist(pts, pts)
    np.fill_diagonal(d, np.inf)
    mn = d.min()
    mx = d.max()
    if mn < 1e-9:
        return np.inf
    return (mx / mn) ** 2

def _get_tetrahedron_vertices():
    """
    Return the 4 vertices of a regular tetrahedron centered at the origin.
    Using the standard embedding in 3D.
    """
    a = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1]
    ], dtype=float)
    a /= np.sqrt(3) # Normalize to unit sphere
    return a

def _generate_stella_octangula_config(params):
    """
    Generate 14 points from a parameter vector.
    
    Parameters (18 total):
    0: rotation angle around z-axis for the second tetrahedron
    1: rotation angle around x-axis for the second tetrahedron
    2: rotation angle around y-axis for the second tetrahedron
    3: scale of the first tetrahedron
    4: scale of the second tetrahedron
    5: x-translation of the first tetrahedron
    6: y-translation of the first tetrahedron
    7: z-translation of the first tetrahedron
    8: x-translation of the second tetrahedron
    9: y-translation of the second tetrahedron
    10: z-translation of the second tetrahedron
    11-16: coordinates of the 6 additional points (2x3)
    
    This creates 8 points from two tetrahedra and 6 free points.
    """
    theta_z = params[0]
    theta_x = params[1]
    theta_y = params[2]
    scale1 = params[3]
    scale2 = params[4]
    trans1 = params[5:8]
    trans2 = params[8:11]
    extra_pts = params[11:17].reshape(2, 3)
    
    # Base tetrahedron
    tet = _get_tetrahedron_vertices()
    
    # First tetrahedron
    tet1 = tet * scale1 + trans1
    
    # Rotation matrix for the second tetrahedron
    # Simple Euler angles Z-X-Y
    cz, sz = np.cos(theta_z), np.sin(theta_z)
    cx, sx = np.cos(theta_x), np.sin(theta_x)
    cy, sy = np.cos(theta_y), np.sin(theta_y)
    
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    
    R = Rz @ Rx @ Ry
    tet2 = (tet @ R.T) * scale2 + trans2
    
    # Combine
    pts = np.vstack([tet1, tet2, extra_pts])
    
    return pts

def _objective(params):
    """
    Objective function for differential evolution.
    Returns the squared ratio of max to min distance.
    """
    pts = _generate_stella_octangula_config(params)
    return _true_score(pts)

def construct():
    global _t0
    if _t0 is None:
        _t0 = time.time()
    
    best_pts = None
    best_score = np.inf
    
    # Define bounds for the parameters
    # Angles: -pi to pi
    # Scales: 0.1 to 2.0
    # Translations: -2.0 to 2.0
    # Extra points: -2.0 to 2.0
    
    bounds = [
        (-np.pi, np.pi),  # theta_z
        (-np.pi, np.pi),  # theta_x
        (-np.pi, np.pi),  # theta_y
        (0.1, 2.0),       # scale1
        (0.1, 2.0),       # scale2
        (-2.0, 2.0),      # trans1 x
        (-2.0, 2.0),      # trans1 y
        (-2.0, 2.0),      # trans1 z
        (-2.0, 2.0),      # trans2 x
        (-2.0, 2.0),      # trans2 y
        (-2.0, 2.0),      # trans2 z
    ]
    # 6 extra points, 3 coords each
    for _ in range(6):
        bounds.append((-2.0, 2.0))
    
    # Initial guess
    x0 = np.zeros(len(bounds))
    # Start with standard stella octangula: two interlocking tetrahedra
    # One rotated by 90 degrees around z? No, usually they share the same center.
    # Standard stella octangula: one tetrahedron and its inversion (rotation by 180 around axes? No, just the other 4 vertices of a cube).
    # The other 4 vertices of a cube are (-1, -1, -1), (-1, 1, 1), (1, -1, 1), (1, 1, -1) normalized.
    # This is equivalent to a rotation?
    # Let's just use a reasonable initial guess.
    x0[0] = 0
    x0[1] = 0
    x0[2] = 0
    x0[3] = 1.0
    x0[4] = 1.0
    # Translations 0
    
    # Run Differential Evolution
    # We have ~18 parameters.
    # Time budget: ~130 seconds
    start_time = time.time()
    
    try:
        # Use differential evolution
        # maxiter: how many generations
        # popsize: population size
        # We need to be careful with time.
        
        # Let's run a quick DE first
        de_result = differential_evolution(
            _objective,
            bounds,
            seed=42,
            maxiter=50,
            popsize=15,
            tol=1e-4,
            polish=True,
            init='sobol'
        )
        
        if de_result.success or de_result.fun < best_score:
            best_score = de_result.fun
            best_pts = _generate_stella_octangula_config(de_result.x)
            
    except Exception as e:
        pass
        
    # If we have time, do a local refinement using L-BFGS-B on the full 42 coordinates
    # starting from the best symmetric configuration found.
    if best_pts is not None and (time.time() - start_time) < DEADLINE - 30:
        try:
            def full_obj(x_flat):
                pts = x_flat.reshape(N, D)
                return _true_score(pts)
            
            # Try a few random restarts from the symmetric solution
            rng = np.random.default_rng(42)
            for i in range(5):
                if time.time() - start_time > DEADLINE - 20:
                    break
                
                # Perturb the symmetric solution
                x0_flat = best_pts.flatten() + rng.normal(0, 0.1, size=(N * D,))
                
                res = minimize(
                    full_obj,
                    x0_flat,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                
                if res.fun < best_score:
                    best_score = res.fun
                    best_pts = res.x.reshape(N, D)
                    
        except Exception:
            pass
            
    # If DE failed or didn't produce a good result, fallback to a known good configuration
    if best_pts is None or best_score > 10.0:
        # Fallback: Fibonacci sphere
        i = np.arange(N)
        phi = np.arccos(1 - 2 * (i + 0.5) / N)
        theta = np.pi * (1 + 5 ** 0.5) * i
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        best_pts = np.column_stack([x, y, z])
        best_score = _true_score(best_pts)
        
    # Ensure distinct points
    d = cdist(best_pts, best_pts)
    np.fill_diagonal(d, np.inf)
    if d.min() < 1e-6:
        best_pts += np.random.normal(0, 1e-4, size=(N, D))
        
    return best_pts
