import math
import numpy as np
import time
from scipy.optimize import linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 8.0
EPS = 1e-9

def hex_vertices(cx, cy, angle_deg):
    """Return 6 vertices of unit regular hexagon centered at (cx,cy) rotated by angle_deg."""
    a = math.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = a + k * math.pi / 3.0
        verts.append((cx + math.cos(theta), cy + math.sin(theta)))
    return np.array(verts)

def hexagon_overlap(v1, v2, eps=1e-9):
    """Check if two hexagons (arrays of 6 vertices) overlap (interior intersection).
    Returns True if they overlap (interiors intersect)."""
    # Check if any vertex of v1 is strictly inside v2, or vice versa
    # A point is inside a convex polygon if it's on the same side of all edges
    def point_in_convex_poly(pt, poly):
        n = len(poly)
        for i in range(n):
            p0 = poly[i]
            p1 = poly[(i+1) % n]
            edge = p1 - p0
            normal = np.array([-edge[1], edge[0]])
            if np.dot(pt - p0, normal) < -eps:
                return False
        return True
    
    # Check v1 vertices in v2
    for pt in v1:
        if point_in_convex_poly(pt, v2):
            return True
    # Check v2 vertices in v1
    for pt in v2:
        if point_in_convex_poly(pt, v1):
            return True
    return False

def enclosing_hexagon_side(pts, center, beta):
    """Compute the minimal enclosing regular hexagon side length for points with center and rotation beta."""
    # The 6 supporting lines of a regular hexagon with side S, centered at c, rotated by beta
    # have normals at angles beta + pi/6 + k*pi/3
    # The apothem (distance from center to each side) = S * sqrt(3)/2
    # For a point p, the projection onto normal n is (p-c).n
    # We need max over all points and normals of |(p-c).n| <= apothem
    normals = np.array([
        (math.cos(beta + math.pi/6.0 + k*math.pi/3.0),
         math.sin(beta + math.pi/6.0 + k*math.pi/3.0))
        for k in range(6)
    ])
    centered = pts - np.array(center)
    projs = centered @ normals.T  # shape (n_pts, 6)
    D = float(np.max(np.abs(projs)))
    S = (D + APOTHEM) / APOTHEM
    return S

def lp_enclosing_hexagon(pts, beta):
    """Use LP to find optimal center for enclosing hexagon with fixed rotation beta.
    Returns (S, center, D)."""
    normals = np.array([
        (math.cos(beta + math.pi/6.0 + k*math.pi/3.0),
         math.sin(beta + math.pi/6.0 + k*math.pi/3.0))
        for k in range(6)
    ])
    # Variables: cx, cy, D
    # Minimize D
    # Constraints: for each point p and each normal n:
    #   (px - cx)*nx + (py - cy)*ny <= D  =>  -nx*cx - ny*cy - D <= -px*nx - py*ny
    #   (px - cx)*nx + (py - cy)*ny >= -D =>   nx*cx + ny*cy - D <=  px*nx + py*ny
    c_obj = np.array([0.0, 0.0, 1.0])
    n_pts = len(pts)
    n_cons = 2 * 6 * n_pts
    A_ub = np.zeros((n_cons, 3))
    b_ub = np.zeros(n_cons)
    row = 0
    for k in range(6):
        nx, ny = normals[k]
        for i in range(n_pts):
            px, py = pts[i]
            # (px-cx)*nx + (py-cy)*ny <= D
            A_ub[row, 0] = -nx
            A_ub[row, 1] = -ny
            A_ub[row, 2] = -1.0
            b_ub[row] = -(px*nx + py*ny)
            row += 1
            # (px-cx)*nx + (py-cy)*ny >= -D  =>  nx*cx + ny*cy - D <= px*nx + py*ny
            A_ub[row, 0] = nx
            A_ub[row, 1] = ny
            A_ub[row, 2] = -1.0
            b_ub[row] = px*nx + py*ny
            row += 1
    bounds = [(None, None), (None, None), (0.0, None)]
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            cx, cy, D = float(res.x[0]), float(res.x[1]), float(res.x[2])
            S = (D + APOTHEM) / APOTHEM
            return S, (cx, cy), D
    except Exception:
        pass
    return float('inf'), None, float('inf')

def objective_for_positions(positions, angles, center, beta):
    """Compute enclosing hexagon side length for given positions and angles."""
    pts = np.array(positions)
    return enclosing_hexagon_side(pts, center, beta)

def simulate_annealing(n_iters, time_budget):
    """Simulated annealing to find good positions for 11 unit hexagons."""
    # Initial: axial lattice configuration (compact)
    axial_init = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    positions = []
    for q, r in axial_init:
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        positions.append([x, y])
    positions = np.array(positions, dtype=float)
    angles = np.zeros(N)  # all aligned initially
    
    # Precompute vertices for initial config
    best_positions = positions.copy()
    best_angles = angles.copy()
    best_S = float('inf')
    
    # Initial enclosing hexagon
    center = (float(np.mean(positions[:, 0])), float(np.mean(positions[:, 1])))
    # Try a few rotations
    for beta_deg in [0, 30, 60, 90, 120, 150]:
        beta = math.radians(beta_deg)
        S = enclosing_hexagon_side(positions, center, beta)
        if S < best_S:
            best_S = S
            best_positions = positions.copy()
            best_angles = angles.copy()
    
    T = 0.5  # initial temperature
    T_min = 0.001
    cooling = 0.995
    
    t0 = time.time()
    for it in range(n_iters):
        if time.time() - t0 > time_budget:
            break
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        T = max(T_min, T * cooling)
        
        # Pick a random hexagon to move
        i = np.random.randint(N)
        old_x, old_y = positions[i]
        old_angle = angles[i]
        
        # Propose new position
        step = T * 0.5
        new_x = old_x + np.random.randn() * step
        new_y = old_y + np.random.randn() * step
        new_angle = (old_angle + np.random.randn() * step * 0.3) % 180.0  # degrees
        
        # Check collision with all other hexagons
        new_verts = hex_vertices(new_x, new_y, new_angle)
        collision = False
        for j in range(N):
            if j == i:
                continue
            other_verts = hex_vertices(positions[j, 0], positions[j, 1], angles[j])
            if hexagon_overlap(new_verts, other_verts):
                collision = True
                break
        
        if collision:
            # Reject
            continue
        
        # Compute new objective
        positions[i, 0] = new_x
        positions[i, 1] = new_y
        angles[i] = new_angle
        
        # Quick objective: use centroid as center, try best rotation
        center = (float(np.mean(positions[:, 0])), float(np.mean(positions[:, 1])))
        S_new = float('inf')
        for beta_deg in [0, 30, 60, 90, 120, 150]:
            beta = math.radians(beta_deg)
            S = enclosing_hexagon_side(positions, center, beta)
            if S < S_new:
                S_new = S
        
        # Metropolis acceptance
        delta_S = S_new - best_S
        if delta_S < 0 or np.random.rand() < math.exp(-delta_S / T):
            best_positions = positions.copy()
            best_angles = angles.copy()
            best_S = S_new
        else:
            # Revert
            positions[i, 0] = old_x
            positions[i, 1] = old_y
            angles[i] = old_angle
    
    return best_positions, best_angles, best_S

def polish_solution(positions, angles, time_budget):
    """Polish the solution: optimize center and rotation for the given positions."""
    pts = np.array(positions)
    best_S = float('inf')
    best_center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
    best_beta = 0.0
    
    t0 = time.time()
    
    # Grid search over rotation
    n_grid = 120
    for i in range(n_grid):
        if time.time() - t0 > time_budget:
            break
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        beta = i * (math.pi / 6.0) / (n_grid - 1)
        S, center, D = lp_enclosing_hexagon(pts, beta)
        if S < best_S:
            best_S = S
            best_center = center
            best_beta = beta
    
    # Refine around best
    if best_center is not None:
        lo = max(0.0, best_beta - math.pi / 60)
        hi = min(math.pi / 6.0, best_beta + math.pi / 60)
        # Fine grid
        for i in range(60):
            if time.time() - t0 > time_budget:
                break
            beta = lo + i * (hi - lo) / 59.0
            S, center, D = lp_enclosing_hexagon(pts, beta)
            if S < best_S:
                best_S = S
                best_center = center
                best_beta = beta
    
    # Final: use LP result directly (it's already exact for fixed beta)
    # The LP gives the optimal center for that beta
    return best_S, best_center, best_beta

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    # Phase 1: Simulated annealing
    sa_time = min(80.0, TIME_LIMIT - START_TIME - 30.0)
    if sa_time > 5.0:
        positions, angles, S_sa = simulate_annealing(200000, sa_time)
    else:
        # Fallback: use axial lattice
        axial_init = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (1, 1), (-1, 2), (-2, 1), (2, -1)
        ]
        positions = []
        for q, r in axial_init:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            positions.append([x, y])
        positions = np.array(positions, dtype=float)
        angles = np.zeros(N)
        S_sa = float('inf')
    
    # Phase 2: Polish
    polish_time = TIME_LIMIT - START_TIME - SAFETY_MARGIN - 5.0
    if polish_time < 5.0:
        polish_time = 5.0
    S_final, center, beta = polish_solution(positions, angles, polish_time)
    
    if S_final == float('inf') or center is None:
        # Fallback
        center = (float(np.mean(positions[:, 0])), float(np.mean(positions[:, 1])))
        beta = 0.0
        S_final = enclosing_hexagon_side(positions, center, beta)
    
    # Build output
    inner = []
    for i in range(N):
        inner.append([float(positions[i, 0]), float(positions[i, 1]), float(angles[i])])
    
    cx, cy = center
    angle_deg = math.degrees(beta)
    
    return inner, [float(cx), float(cy)], float(S_final), float(angle_deg)
