"""Packing 11 unit hexagons in a regular hexagon - minimize outer side length."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 11
DEADLINE = time.time() + 140.0  # Safety margin before 150s kill

def hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy), side length, rotation in degrees."""
    angle_rad = np.radians(angle_deg)
    r = side  # circumradius = side for regular hexagon
    angles = angle_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + r * np.cos(angles), cy + r * np.sin(angles)])

def all_vertices(inner):
    """Get all vertices of all inner hexagons."""
    verts = []
    for x, y, ang in inner:
        verts.append(hex_vertices(x, y, 1.0, ang))
    return np.vstack(verts)

def enclosing_hexagon_size(pts, theta):
    """Given points and rotation theta (radians), find min side length of enclosing regular hexagon.
    Returns (side_length, center)."""
    # 6 unit normals
    normals = np.array([[np.cos(theta + k * np.pi / 3), np.sin(theta + k * np.pi / 3)] for k in range(6)])
    
    def objective(c):
        # c = [cx, cy]
        diffs = pts - c
        projections = diffs @ normals.T  # shape (n, 6)
        max_proj = np.max(projections, axis=0)  # shape (6,)
        # S * sqrt(3)/2 >= max_proj[k] for all k
        # S >= 2/sqrt(3) * max_proj[k]
        return np.max(max_proj) * 2.0 / np.sqrt(3.0)
    
    # Try multiple starting points for center
    best_s = float('inf')
    best_c = None
    x0_candidates = [np.array([np.mean(pts[:, 0]), np.mean(pts[:, 1])])]
    # Add a few more
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            x0_candidates.append(np.array([np.mean(pts[:, 0]) + dx, np.mean(pts[:, 1]) + dy]))
    
    for x0 in x0_candidates:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
        if res.success and res.fun < best_s:
            best_s = res.fun
            best_c = res.x
    
    return best_s, best_c

def evaluate_config(inner):
    """Evaluate a configuration: find minimal enclosing hexagon over all rotations."""
    pts = all_vertices(inner)
    if time.time() > DEADLINE:
        return float('inf'), None, None
    
    # Search over rotation angle
    best_s = float('inf')
    best_theta = 0
    best_c = None
    
    # Coarse search over theta
    for theta_deg in range(0, 60, 5):  # 0 to 55 degrees, 12 samples
        if time.time() > DEADLINE:
            break
        theta = np.radians(theta_deg)
        s, c = enclosing_hexagon_size(pts, theta)
        if s < best_s:
            best_s = s
            best_theta = theta
            best_c = c
    
    # Fine search around best
    for theta_deg in np.linspace(best_theta - 5, best_theta + 5, 11):
        if time.time() > DEADLINE:
            break
        theta = np.radians(theta_deg)
        s, c = enclosing_hexagon_size(pts, theta)
        if s < best_s:
            best_s = s
            best_theta = theta
            best_c = c
    
    return best_s, best_c, best_theta

def hexagonal_lattice_config():
    """Generate a hexagonal lattice arrangement of 11 hexagons."""
    # Flat-top hexagons: neighbors at angles 0, 60, 120, 180, 240, 300 with distance sqrt(3)
    d = np.sqrt(3)
    centers = [(0.0, 0.0)]
    # First ring: 6 hexagons
    for k in range(6):
        angle = np.radians(60 * k)
        centers.append((d * np.cos(angle), d * np.sin(angle)))
    # Second ring: need 4 more to make 11
    # Positions at distance 2d in directions 0, 60, 120, 180, 240, 300
    # And positions at distance d*sqrt(3) in directions 30, 90, 150, 210, 270, 330
    # Let's add 4 from the second ring
    added = 0
    for k in range(6):
        if added >= 4:
            break
        angle = np.radians(60 * k)
        cx = 2 * d * np.cos(angle)
        cy = 2 * d * np.sin(angle)
        centers.append((cx, cy))
        added += 1
    
    inner = [[x, y, 0.0] for x, y in centers[:11]]
    return inner

def construct():
    if time.time() > DEADLINE:
        # Fallback: just return something
        inner = [[0.0, 0.0, 0.0]] * N
        return inner, (0.0, 0.0), 5.0, 0.0
    
    best_inner = None
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Start with hexagonal lattice
    inner0 = hexagonal_lattice_config()
    s, c, theta = evaluate_config(inner0)
    if s < best_side:
        best_side = s
        best_inner = inner0
        best_center = c
        best_angle = np.degrees(theta)
    
    # Try some perturbations
    np.random.seed(42)
    for trial in range(20):
        if time.time() > DEADLINE - 5:
            break
        
        inner = hexagonal_lattice_config()
        # Add small random perturbations
        for i in range(len(inner)):
            inner[i][0] += np.random.uniform(-0.1, 0.1)
            inner[i][1] += np.random.uniform(-0.1, 0.1)
            inner[i][2] = np.random.uniform(0, 60)
        
        s, c, theta = evaluate_config(inner)
        if s < best_side:
            best_side = s
            best_inner = inner
            best_center = c
            best_angle = np.degrees(theta)
    
    # Try different second-ring selections
    d = np.sqrt(3)
    for second_ring_pattern in range(6):
        if time.time() > DEADLINE - 5:
            break
        centers = [(0.0, 0.0)]
        for k in range(6):
            angle = np.radians(60 * k)
            centers.append((d * np.cos(angle), d * np.sin(angle)))
        # Add 4 from second ring with offset
        for k in range(second_ring_pattern, second_ring_pattern + 4):
            angle = np.radians(60 * k)
            cx = 2 * d * np.cos(angle)
            cy = 2 * d * np.sin(angle)
            centers.append((cx, cy))
        
        inner = [[x, y, 0.0] for x, y in centers[:11]]
        s, c, theta = evaluate_config(inner)
        if s < best_side:
            best_side = s
            best_inner = inner
            best_center = c
            best_angle = np.degrees(theta)
    
    if best_inner is None:
        inner0 = hexagonal_lattice_config()
        return inner0, (0.0, 0.0), 5.0, 0.0
    
    return best_inner, best_center, best_side, best_angle
