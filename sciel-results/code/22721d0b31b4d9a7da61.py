import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar, differential_evolution

N = 12
DEADLINE = 115.0  # Safety margin

# Precompute unit hexagon vertices (flat-top orientation)
_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))

# Precompute edge normals for SAT
_EDGE_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i+1)%6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i+1)%6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = math.sqrt(_nx*_nx + _ny*_ny)
    _EDGE_NORMALS[_i] = [_nx / _norm, _ny / _norm]

# For a regular hexagon with side length s, the inradius is s * sqrt(3)/2
# The circumradius is s
# The width across flats is s * sqrt(3)
# The width across corners is 2*s
# For the containing hexagon with side L, inradius is L * sqrt(3)/2
# The condition for a hexagon of side 1 to fit inside is:
# distance from center to the farthest vertex <= L * sqrt(3)/2  (if aligned)
# More generally, for any orientation, the containing hexagon must cover the Minkowski sum
# of the inner hexagons.

def construct():
    t0 = time.time()
    
    # Fast overlap check using SAT
    def hex_overlap(x1, y1, a1, x2, y2, a2):
        # If centers are far apart, no overlap
        dx = x1 - x2
        dy = y1 - y2
        if dx*dx + dy*dy > 4.0:  # max distance for overlap is 2 (sum of circumradii)
            return False
        
        c1, s1 = math.cos(a1), math.sin(a1)
        c2, s2 = math.cos(a2), math.sin(a2)
        
        # Project all vertices of hex 1 onto normals of hex 2 and vice versa
        # Also need to check normals of hex 1 onto vertices of hex 2
        # Actually, SAT requires checking all normals from both hexagons.
        
        # Hex 1 vertices
        v1x = x1 + _UNIT_HEX_VERTS[:, 0] * c1 - _UNIT_HEX_VERTS[:, 1] * s1
        v1y = y1 + _UNIT_HEX_VERTS[:, 0] * s1 + _UNIT_HEX_VERTS[:, 1] * c1
        # Hex 2 vertices
        v2x = x2 + _UNIT_HEX_VERTS[:, 0] * c2 - _UNIT_HEX_VERTS[:, 1] * s2
        v2y = y2 + _UNIT_HEX_VERTS[:, 0] * s2 + _UNIT_HEX_VERTS[:, 1] * c2
        
        # Normals for hex 1 (rotated)
        n1x = _EDGE_NORMALS[:, 0] * c1 - _EDGE_NORMALS[:, 1] * s1
        n1y = _EDGE_NORMALS[:, 0] * s1 + _EDGE_NORMALS[:, 1] * c1
        # Normals for hex 2 (rotated)
        n2x = _EDGE_NORMALS[:, 0] * c2 - _EDGE_NORMALS[:, 1] * s2
        n2y = _EDGE_NORMALS[:, 0] * s2 + _EDGE_NORMALS[:, 1] * c2
        
        # Check all 12 axes
        for k in range(6):
            # Axis from hex 1 normal k
            ax, ay = n1x[k], n1y[k]
            p1 = v1x * ax + v1y * ay
            p2 = v2x * ax + v2y * ay
            if p1.max() < p2.min() or p2.max() < p1.min():
                return False
            # Axis from hex 2 normal k
            ax, ay = n2x[k], n2y[k]
            p1 = v1x * ax + v1y * ay
            p2 = v2x * ax + v2y * ay
            if p1.max() < p2.min() or p2.max() < p1.min():
                return False
        return True

    def check_all_overlap(params):
        # params: [x1, y1, a1, x2, y2, a2, ...]
        # returns True if any overlap
        for i in range(N):
            for j in range(i+1, N):
                x1, y1, a1 = params[3*i], params[3*i+1], params[3*i+2]
                x2, y2, a2 = params[3*j], params[3*j+1], params[3*j+2]
                if hex_overlap(x1, y1, a1, x2, y2, a2):
                    return True
        return False

    def objective(params):
        # Minimize the side length of the containing hexagon
        # Center is the centroid of all hexagon centers
        xs = params[0::3]
        ys = params[1::3]
        angles = params[2::3]
        
        cx = np.mean(xs)
        cy = np.mean(ys)
        
        # Get all vertices relative to center
        all_verts_x = []
        all_verts_y = []
        for i in range(N):
            c, s = math.cos(angles[i]), math.sin(angles[i])
            vx = xs[i] + _UNIT_HEX_VERTS[:, 0] * c - _UNIT_HEX_VERTS[:, 1] * s
            vy = ys[i] + _UNIT_HEX_VERTS[:, 0] * s + _UNIT_HEX_VERTS[:, 1] * c
            all_verts_x.extend(vx)
            all_verts_y.extend(vy)
        
        vx = np.array(all_verts_x) - cx
        vy = np.array(all_verts_y) - cy
        
        # The containing hexagon is regular, side L, centered at (cx, cy), angle 0 (WLOG)
        # Actually, the containing hexagon can be rotated. But due to symmetry, we can fix its angle.
        # Wait, the problem says "outer hexagon center (x, y), side length and rotation".
        # So we need to minimize over the outer hexagon's rotation too.
        # But the score is just the side length.
        # For a fixed set of points, the minimum side length of a containing regular hexagon
        # is the minimum over all rotations of the hexagon of the maximum "radius" in that direction.
        
        # Let's iterate over the outer hexagon's rotation.
        # The condition for a point (x, y) to be inside a regular hexagon of side L, centered at origin, 
        # with angle phi is:
        # max_i | (x * nx_i + y * ny_i) | <= L * sqrt(3) / 2
        # where (nx_i, ny_i) are the outward normals of the hexagon edges.
        
        # For a regular hexagon, the normals are at angles phi + 0, 60, 120, 180, 240, 300 degrees.
        # The max projection on these normals must be <= L * sqrt(3) / 2.
        
        # So L >= (2 / sqrt(3)) * max_i | dot(p, n_i) |
        
        # We need to minimize this over phi.
        # This is a 1D minimization problem for phi.
        
        def outer_side(phi):
            # phi in radians
            normals = phi + np.arange(6) * np.pi / 3
            nx = np.cos(normals)
            ny = np.sin(normals)
            projs = np.abs(vx * nx + vy * ny)
            max_proj = np.max(projs)
            return (2.0 / math.sqrt(3)) * max_proj
        
        # Find the minimum over phi
        best_L = float('inf')
        # Coarse scan
        for phi_deg in range(0, 60, 1):
            L = outer_side(np.radians(phi_deg))
            if L < best_L:
                best_L = L
        
        # Refine
        # Since the function is periodic and smooth, we can use minimize_scalar
        # But we need to know the initial guess. Let's just do a finer scan.
        for phi_deg in np.linspace(0, 60, 61):
            L = outer_side(np.radians(phi_deg))
            if L < best_L:
                best_L = L
        
        return best_L

    # Initial guess: honeycomb lattice
    sqrt3 = math.sqrt(3)
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3/2.0, 1.5])
    lattice_coords = [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, -1), (-1, 1), (2, -1), (-2, 1),
        (1, 1), (-1, -1), (2, 0),
    ]
    
    initial_params = np.zeros(3 * N)
    for i, (u, v) in enumerate(lattice_coords):
        pos = u * a1 + v * a2
        initial_params[3*i] = pos[0]
        initial_params[3*i+1] = pos[1]
        initial_params[3*i+2] = 0.0

    # Check initial validity
    if check_all_overlap(initial_params):
        # If initial is invalid, try a different arrangement
        # For now, just perturb
        pass

    best_side = objective(initial_params)
    best_params = initial_params.copy()
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling_rate = 0.999
    step_size = 0.5
    steps = 0
    max_steps = 20000
    accept_count = 0
    total_count = 0
    
    current_params = initial_params.copy()
    current_side = best_side
    
    while time.time() - t0 < DEADLINE - 10:
        steps += 1
        total_count += 1
        
        idx = random.randint(0, N-1)
        new_params = current_params.copy()
        
        # Perturb position and angle
        dx = random.gauss(0, step_size)
        dy = random.gauss(0, step_size)
        da = random.gauss(0, 0.1)
        
        new_params[3*idx] += dx
        new_params[3*idx+1] += dy
        new_params[3*idx+2] += da
        
        # Check overlap
        if check_all_overlap(new_params):
            # Rejection
            T *= cooling_rate
            if T < T_min:
                T = T_min
            continue
        
        new_side = objective(new_params)
        
        if new_side < current_side:
            current_params = new_params
            current_side = new_side
            accept_count += 1
            if current_side < best_side:
                best_side = current_side
                best_params = current_params.copy()
        else:
            delta = new_side - current_side
            if random.random() < math.exp(-delta / T):
                current_params = new_params
                current_side = new_side
                accept_count += 1
        
        # Adaptive temperature and step size
        if total_count % 500 == 0:
            acc_rate = accept_count / total_count
            if acc_rate < 0.2:
                step_size *= 0.9
            elif acc_rate > 0.5:
                step_size *= 1.1
            accept_count = 0
            total_count = 0
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    # Final local optimization using Nelder-Mead
    if time.time() - t0 < DEADLINE - 5:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead', 
                           options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_side and not check_all_overlap(res.x):
                best_side = res.fun
                best_params = res.x
        except Exception:
            pass

    # Extract results
    xs = best_params[0::3]
    ys = best_params[1::3]
    angles_deg = np.degrees(best_params[2::3])
    
    cx = np.mean(xs)
    cy = np.mean(ys)
    
    inner = []
    for i in range(N):
        inner.append([float(xs[i]), float(ys[i]), float(angles_deg[i])])
    
    center = (float(cx), float(cy))
    side = float(best_side)
    angle = 0.0  # The outer hexagon's optimal angle is implicitly minimized in the objective,
                 # but we need to return it. Let's find it.
    
    # Find the optimal angle for the outer hexagon
    vx = []
    vy = []
    for i in range(N):
        c, s = math.cos(best_params[3*i+2]), math.sin(best_params[3*i+2])
        vx.extend(xs[i] + _UNIT_HEX_VERTS[:, 0] * c - _UNIT_HEX_VERTS[:, 1] * s - cx)
        vy.extend(ys[i] + _UNIT_HEX_VERTS[:, 0] * s + _UNIT_HEX_VERTS[:, 1] * c - cy)
    vx = np.array(vx)
    vy = np.array(vy)
    
    best_angle = 0.0
    best_L = float('inf')
    for phi_deg in np.linspace(0, 60, 121):
        phi = np.radians(phi_deg)
        normals = phi + np.arange(6) * np.pi / 3
        nx = np.cos(normals)
        ny = np.sin(normals)
        projs = np.abs(vx * nx + vy * ny)
        L = (2.0 / math.sqrt(3)) * np.max(projs)
        if L < best_L:
            best_L = L
            best_angle = phi_deg
    
    angle = float(best_angle)
    
    return inner, center, side, angle
