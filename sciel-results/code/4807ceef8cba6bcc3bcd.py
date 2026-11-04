import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given angle in degrees."""
    a = np.radians(angle_deg)
    angles = a + np.pi * np.arange(6) / 3.0
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def _min_outer_side(inner_pts):
    """Given list of (x,y,angle) for inner hexagons, find min outer hexagon side length.
    Outer hexagon centered at origin, can rotate. Minimize over outer angle."""
    # Collect all vertices of all inner hexagons
    all_verts = []
    for x, y, ang in inner_pts:
        verts = _hex_vertices(x, y, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    def objective(outer_angle):
        # Outer hexagon with side S centered at origin, rotated by outer_angle
        # A point p is inside outer hexagon of side S (pointy-top, rotated) iff
        # for each of 3 axes, the projection is <= S
        # The 3 axis directions for a hexagon rotated by theta are at theta, theta+60, theta+120
        # Actually for a regular hexagon, the 3 pairs of parallel sides have normals at 
        # theta, theta+60, theta+120 (and opposites)
        # Distance from center to each side = S * cos(30°) = S * sqrt(3)/2 ... 
        # No wait. For a regular hexagon with side length S, the apothem (inradius) = S*sqrt(3)/2
        # The circumradius = S
        # A point is inside iff its projection onto each of the 3 normal directions <= apothem
        # Normal directions are at angles: outer_angle + 30°, outer_angle + 90°, outer_angle + 150°
        # (perpendicular to the 3 pairs of sides)
        
        normals = []
        for k in range(3):
            ang = np.radians(outer_angle) + np.pi/6 + k * np.pi/3
            normals.append(np.array([np.cos(ang), np.sin(ang)]))
        normals = np.array(normals)
        
        # Project all vertices onto each normal
        projections = all_verts @ normals.T  # shape (n_verts, 3)
        max_proj = np.max(np.abs(projections), axis=0)  # max abs projection per normal
        
        # For the point to be inside, we need max_proj[k] <= S * sqrt(3)/2 for all k
        # So S >= max_k(max_proj[k]) / (sqrt(3)/2)
        return np.max(max_proj) / (np.sqrt(3) / 2)
    
    # Try several initial angles for outer hexagon
    best_S = float('inf')
    best_ao = 0.0
    for ao_init in np.linspace(0, 180, 12):
        res = minimize(objective, [ao_init], method='Nelder-Mead', 
                      options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
        S = res.fun
        if S < best_S:
            best_S = S
            best_ao = res.x[0]
    
    return best_S, best_ao

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    # Initial configuration: honeycomb-like arrangement
    # Place 12 hexagons in a pattern
    # A hexagonal number pattern: center + 6 around + 5 more
    # Or a 4x3 rectangular-ish grid with hexagonal spacing
    
    # Hexagonal lattice spacing: adjacent hexagons center distance = 2 * sqrt(3)/2 = sqrt(3) for flat sides touching
    # Actually for two unit hexagons to be tangent (sharing an edge), center distance = 2 * apothem = sqrt(3)
    # For flat-top hexagons, horizontal spacing = 3/2 * side = 1.5, vertical = sqrt(3)
    
    # Let's use a hexagonal packing arrangement
    # 12 hexagons: could be a 4-3-4-... or a hexagon of radius 2 (which has 19, too many)
    # Hexagon of radius 1 has 7, radius 2 has 19
    # So 12 is between. Let's do a compact arrangement.
    
    # Try: a roughly rectangular grid with hexagonal offset
    # Row 0: 4 hexagons
    # Row 1: 4 hexagons (offset)
    # Row 2: 4 hexagons
    # That's 12
    
    # For flat-top hexagons:
    # Horizontal spacing between adjacent in same row: 1.5 (3/2 * side)
    # Vertical spacing between rows: sqrt(3)
    # Offset for alternating rows: 0.75
    
    dx = 1.5
    dy = np.sqrt(3)
    
    initial_inner = []
    positions = []
    for row in range(3):
        for col in range(4):
            x = col * dx
            y = row * dy
            if row % 2 == 1:
                x += dx / 2
            positions.append((x, y, 0.0))
    
    # Center the configuration
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (max(xs) + min(xs)) / 2
    cy = (max(ys) + min(ys)) / 2
    for i, (x, y, a) in enumerate(positions):
        positions[i] = (x - cx, y - cy, a)
    
    initial_inner = positions
    
    # Optimize: variables are x, y, angle for each hexagon (36 vars)
    # Objective: minimum outer hexagon side length
    
    def full_objective(params):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5.0:
            return 1e10  # signal timeout
        
        inner = []
        for i in range(N):
            x = params[3*i]
            y = params[3*i+1]
            ang = params[3*i+2]
            inner.append((x, y, ang))
        
        S, _ = _min_outer_side(inner)
        return S
    
    # Initial params
    x0 = []
    for x, y, a in initial_inner:
        x0.extend([x, y, a])
    x0 = np.array(x0)
    
    best_S, best_params = full_objective(x0), x0.copy()
    
    # Try Nelder-Mead optimization
    try:
        res = minimize(full_objective, x0, method='Nelder-Mead',
                      options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8,
                               'adaptive': True})
        if res.fun < best_S:
            best_S = res.fun
            best_params = res.x.copy()
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 10.0:
        # Return best found
        inner = []
        for i in range(N):
            x = best_params[3*i]
            y = best_params[3*i+1]
            ang = best_params[3*i+2]
            inner.append([x, y, ang])
        return inner, (0.0, 0.0), best_S, 0.0
    
    # Try L-BFGS-B as well
    try:
        res2 = minimize(full_objective, x0, method='L-BFGS-B',
                       options={'maxiter': 300, 'ftol': 1e-8})
        if res2.fun < best_S:
            best_S = res2.fun
            best_params = res2.x.copy()
    except Exception:
        pass
    
    # Build final answer
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        ang = best_params[3*i+2]
        inner.append([x, y, ang])
    
    # Get the optimal outer angle
    _, best_ao = _min_outer_side(inner)
    
    return inner, (0.0, 0.0), best_S, best_ao
# EVOLVE-BLOCK-END
