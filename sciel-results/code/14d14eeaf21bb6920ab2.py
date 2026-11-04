import numpy as np
from scipy.optimize import minimize
import time

N = 11

def _hex_vertices(cx, cy, s, angle_deg):
    """Vertices of a regular hexagon with center (cx,cy), side s, rotation angle_deg."""
    a = np.deg2rad(angle_deg)
    angles = a + np.pi/3 * np.arange(6)
    return np.column_stack([cx + s*np.cos(angles), cy + s*np.sin(angles)])

def _min_enclosing_hex_side(inner_pts, outer_center, outer_angle_deg):
    """
    Compute the minimum side length of a regular hexagon with given center and rotation
    that contains all points in inner_pts (each point represents a vertex of an inner hex).
    """
    oc = np.array(outer_center)
    a = np.deg2rad(outer_angle_deg)
    # 6 outward normals of the outer hexagon
    normals = np.column_stack([np.cos(a + np.pi/3*np.arange(6)), 
                                np.sin(a + np.pi/3*np.arange(6))])
    # For each normal, the supporting line must be at distance s from center
    # such that all points are on the inside
    # point p is inside if dot(p - oc, n) <= s for all normals n
    # So s must be >= max over all points and all normals of dot(p-oc, n)
    # But also need dot(p-oc, -n) <= s, i.e., dot(oc-p, n) <= s
    # So s >= max over all points p and all normals n of |dot(p-oc, n)|
    rel = inner_pts - oc
    # Project onto each normal
    projs = rel @ normals.T  # shape (n_pts, 6)
    # s must be >= max over all entries of abs(projs)
    s = np.max(np.abs(projs))
    return s

def _compute_cost(params):
    """
    params: flat array
    First 3*N: inner hex positions and angles (x, y, angle_deg) for each of N hexagons
    Next 3: outer center (ox, oy) and outer angle_deg
    """
    N_inner = 11
    inner_params = params[:3*N_inner].reshape(N_inner, 3)
    outer_center = params[3*N_inner:3*N_inner+2]
    outer_angle = params[3*N_inner+2]
    
    # Get all vertices of all inner hexagons
    all_pts = []
    for i in range(N_inner):
        x, y, ang = inner_params[i]
        verts = _hex_vertices(x, y, 1.0, ang)
        all_pts.append(verts)
    all_pts = np.vstack(all_pts)  # shape (N*6, 2)
    
    return _min_enclosing_hex_side(all_pts, outer_center, outer_angle)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    best_cost = float('inf')
    best_params = None
    
    # Initial guess: honeycomb arrangement
    # Arrange 11 hexagons in a honeycomb pattern
    # Rows: 3, 4, 4 or similar
    # For unit hexagons in honeycomb, horizontal spacing = 2, vertical spacing = sqrt(3)
    # Offset rows by 1 horizontally
    
    def make_honeycomb():
        # 3 rows: 3, 4, 4
        positions = []
        # Row 0: 3 hexagons
        for j in range(3):
            positions.append((2.0*j, 0.0, 0.0))
        # Row 1: 4 hexagons, offset by 1 horizontally, up by sqrt(3)
        for j in range(4):
            positions.append((2.0*j + 1.0, np.sqrt(3), 0.0))
        # Row 2: 4 hexagons, offset by 1 horizontally, up by 2*sqrt(3)
        for j in range(4):
            positions.append((2.0*j + 1.0, 2*np.sqrt(3), 0.0))
        return positions
    
    honeycomb = make_honeycomb()
    
    # Center the arrangement
    xs = [p[0] for p in honeycomb]
    ys = [p[1] for p in honeycomb]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    
    init_params = []
    for x, y, ang in honeycomb:
        init_params.extend([x - cx, y - cy, ang])
    init_params.extend([0.0, 0.0, 0.0])
    init_params = np.array(init_params)
    
    # Multi-start optimization
    n_starts = 5
    rng = np.random.default_rng(42)
    
    starts = [init_params]
    
    # Add some random perturbations
    for _ in range(n_starts - 1):
        perturbed = init_params.copy()
        # Perturb inner positions slightly
        perturbed[:3*N] += rng.normal(0, 0.5, size=3*N)
        # Perturb outer angle
        perturbed[3*N+2] += rng.uniform(-30, 30)
        starts.append(perturbed)
    
    for start in starts:
        if time.time() > deadline - 5.0:
            break
        
        result = minimize(_compute_cost, start, method='Powell', 
                         options={'maxiter': 500, 'xtol': 1e-10, 'ftol': 1e-10})
        
        if result.fun < best_cost:
            best_cost = result.fun
            best_params = result.x
    
    if best_params is None:
        best_params = init_params
        best_cost = _compute_cost(best_params)
    
    # Extract results
    inner_params = best_params[:3*N].reshape(N, 3)
    outer_center = tuple(best_params[3*N:3*N+2])
    outer_angle = best_params[3*N+2]
    
    inner = [[float(inner_params[i,0]), float(inner_params[i,1]), float(inner_params[i,2])] for i in range(N)]
    
    return inner, outer_center, float(best_cost), float(outer_angle)

