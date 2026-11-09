"""Packing 12 unit hexagons in minimal regular hexagon using L-BFGS-B optimization."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

def hex_vertices(cx, cy, angle_deg, side=1.0):
    """Return 6 vertices of a regular hexagon with given center, angle, side length."""
    angle_rad = np.deg2rad(angle_deg)
    verts = []
    for k in range(6):
        theta = angle_rad + k * np.pi / 3.0
        verts.append([cx + side * np.cos(theta), cy + side * np.sin(theta)])
    return np.array(verts)

def point_in_hex(px, py, cx, cy, angle_deg, side=1.0):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with given angle and side."""
    # Transform to hexagon's local frame
    dx = px - cx
    dy = py - cy
    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    lx = cos_a * dx - sin_a * dy
    ly = sin_a * dx + cos_a * dy
    # Regular hexagon with flat top: vertices at angles 0, 60, 120, 180, 240, 300
    # Inradius = side * sqrt(3)/2
    # The hexagon in local coords: |lx| <= side, and |ly| <= sqrt(3)*side*(1 - |lx|/side) for |lx|<=side
    # Actually for a regular hexagon with vertices at (cos(k*60), sin(k*60))*side:
    # The constraint is: max over 3 directions of |dot(v, d_i)| <= side*sqrt(3)/2
    # Directions: 0, 60, 120 degrees
    inradius = side * np.sqrt(3) / 2.0
    # Three normal directions
    d0 = np.array([1.0, 0.0])
    d1 = np.array([0.5, np.sqrt(3)/2.0])
    d2 = np.array([-0.5, np.sqrt(3)/2.0])
    p = np.array([lx, ly])
    h0 = abs(np.dot(p, d0))
    h1 = abs(np.dot(p, d1))
    h2 = abs(np.dot(p, d2))
    return h0 <= inradius + 1e-9 and h1 <= inradius + 1e-9 and h2 <= inradius + 1e-9

def hex_min_distance(c1, a1, c2, a2, side=1.0):
    """Compute minimum distance between two unit hexagons. Returns 0 if overlapping, positive if separated."""
    # Use the support function approach: min distance = max over directions of (support1 - support2 - dot(d, c2-c1))
    # Actually, two convex sets A and B are disjoint iff for all directions d, h_A(d) + h_B(-d) < dot(d, c2-c1)
    # The minimum distance is the minimum over all d of (dot(d, c2-c1) - h_A(d) - h_B(-d))
    # For regular hexagons, we only need to check directions that are normals to edges
    # Edge normals of hexagon 1: 6 directions at angle a1 + k*60
    # Edge normals of hexagon 2: 6 directions at angle a2 + k*60
    # We need to check all 12 edge normal directions (and their negatives)
    
    dx = c2[0] - c1[0]
    dy = c2[1] - c1[1]
    
    inradius = side * np.sqrt(3) / 2.0
    
    min_dist = float('inf')
    
    # Generate all edge normal directions from both hexagons
    for base_angle in [a1, a2]:
        for k in range(6):
            theta = np.deg2rad(base_angle) + k * np.pi / 3.0
            # Normal direction pointing outward from edge
            nx = np.cos(theta)
            ny = np.sin(theta)
            
            # Support of hex1 in direction (nx, ny): the max dot product of vertices with (nx,ny)
            # For a regular hexagon centered at c1 with inradius r, support in direction d is r * (1 + 2*|cos(angle between d and nearest normal)|)
            # Actually, for a regular hexagon, h(d) = inradius * (1 + 2*max(0, |cos(delta)|)) where delta is angle to nearest normal
            # Simpler: just compute directly from vertices
            
            v1 = hex_vertices(c1[0], c1[1], a1, side)
            v2 = hex_vertices(c2[0], c2[1], a2, side)
            
            h1 = np.max(v1[:, 0] * nx + v1[:, 1] * ny)
            h2_neg = np.max(v2[:, 0] * (-nx) + v2[:, 1] * (-ny))
            
            # The gap in this direction
            gap = dx * nx + dy * ny - h1 - h2_neg
            if gap < min_dist:
                min_dist = gap
    
    return min_dist

def objective_and_constraints(params, L):
    """
    params: array of length 3*N + 3 = [x1,y1,a1, x2,y2,a2, ..., xN,yN,aN, ox,oy,oa]
    Actually let's fix outer hexagon center at origin and angle at 0 for simplicity.
    params: 3*N = [x1,y1,a1, ..., xN,yN,aN]
    L is the outer side length (variable to minimize)
    """
    # This won't work directly since L is separate. Let me restructure.
    pass

def objective(params, L):
    """
    params: 3*N array of [x1,y1,a1, ..., xN,yN,aN]
    L: outer hexagon side length
    Returns: penalty function (we want to minimize L + penalties)
    """
    penalty = 0.0
    inradius_outer = L * np.sqrt(3) / 2.0
    
    for i in range(N):
        xi = params[3*i]
        yi = params[3*i+1]
        ai = params[3*i+2]
        
        # Check all 6 vertices of inner hexagon i are inside outer hexagon
        vi = hex_vertices(xi, yi, ai, 1.0)
        for vx, vy in vi:
            # Transform to outer hexagon frame (centered at origin, angle 0)
            # Outer hexagon: inradius = inradius_outer
            # Check using the 3 normal directions
            d0 = np.array([1.0, 0.0])
            d1 = np.array([0.5, np.sqrt(3)/2.0])
            d2 = np.array([-0.5, np.sqrt(3)/2.0])
            p = np.array([vx, vy])
            h0 = abs(np.dot(p, d0))
            h1 = abs(np.dot(p, d1))
            h2 = abs(np.dot(p, d2))
            if h0 > inradius_outer:
                penalty += (h0 - inradius_outer) ** 2
            if h1 > inradius_outer:
                penalty += (h1 - inradius_outer) ** 2
            if h2 > inradius_outer:
                penalty += (h2 - inradius_outer) ** 2
    
    # Check non-overlap between all pairs
    for i in range(N):
        for j in range(i+1, N):
            ci = np.array([params[3*i], params[3*i+1]])
            cj = np.array([params[3*j], params[3*j+1]])
            ai = params[3*i+2]
            aj = params[3*j+2]
            
            dist = hex_min_distance(ci, ai, cj, aj)
            if dist < 0:
                penalty += dist ** 2
    
    return penalty

def full_objective(params):
    """
    params: 3*N + 1 array = [x1,y1,a1, ..., xN,yN,aN, L]
    """
    L = params[-1]
    inner_params = params[:-1]
    return L + 100.0 * objective(inner_params, L)

def gradient(params):
    """Numerical gradient for full_objective."""
    eps = 1e-7
    grad = np.zeros_like(params)
    f0 = full_objective(params)
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        f_plus = full_objective(params_plus)
        grad[i] = (f_plus - f0) / eps
    return grad

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120 second limit
    
    best_L = 40.0
    best_params = None
    
    # Initial guesses: various configurations
    initial_guesses = []
    
    # 1. Grid-like arrangement
    for offset in [0.0, 0.5, 1.0]:
        params = np.zeros(3*N + 1)
        for i in range(N):
            row = i // 4
            col = i % 4
            params[3*i] = (col - 1.5) * 2.2 + offset
            params[3*i+1] = (row - 1.5) * 1.9
            params[3*i+2] = 0.0
        params[-1] = 8.0
        initial_guesses.append(params)
    
    # 2. Ring arrangement
    for radius in [2.0, 3.0, 4.0]:
        params = np.zeros(3*N + 1)
        for i in range(N):
            angle = 2 * np.pi * i / N
            params[3*i] = radius * np.cos(angle)
            params[3*i+1] = radius * np.sin(angle)
            params[3*i+2] = angle * 180 / np.pi
        params[-1] = 8.0
        initial_guesses.append(params)
    
    # 3. Random configurations
    np.random.seed(42)
    for _ in range(10):
        params = np.zeros(3*N + 1)
        for i in range(N):
            params[3*i] = np.random.uniform(-4, 4)
            params[3*i+1] = np.random.uniform(-4, 4)
            params[3*i+2] = np.random.uniform(0, 60)
        params[-1] = 10.0
        initial_guesses.append(params)
    
    # 4. Clustered arrangement
    for center_offset in [(-2, 0), (2, 0), (0, 2), (0, -2)]:
        params = np.zeros(3*N + 1)
        for i in range(N):
            group = i // 3
            pos_in_group = i % 3
            gx = center_offset[0] + (pos_in_group - 1) * 2.2
            gy = center_offset[1] + (group % 2 - 0.5) * 1.9
            params[3*i] = gx
            params[3*i+1] = gy
            params[3*i+2] = 0.0
        params[-1] = 10.0
        initial_guesses.append(params)
    
    # Optimize from each initial guess
    for idx, x0 in enumerate(initial_guesses):
        if time.time() > deadline - 5.0:
            break
        
        try:
            # Use L-BFGS-B with numerical gradient
            result = minimize(
                full_objective,
                x0,
                method='L-BFGS-B',
                jac=gradient,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < best_L:
                L_candidate = result.x[-1]
                # Verify the solution
                inner_params = result.x[:-1]
                penalty = objective(inner_params, L_candidate)
                if penalty < 0.01:  # Valid solution
                    if L_candidate < best_L:
                        best_L = L_candidate
                        best_params = result.x.copy()
        except Exception:
            continue
    
    # Final polish with tighter tolerances if time permits
    if best_params is not None and time.time() < deadline - 10.0:
        try:
            result = minimize(
                full_objective,
                best_params,
                method='L-BFGS-B',
                jac=gradient,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            L_candidate = result.x[-1]
            inner_params = result.x[:-1]
            penalty = objective(inner_params, L_candidate)
            if penalty < 0.01 and L_candidate < best_L:
                best_L = L_candidate
                best_params = result.x.copy()
        except Exception:
            pass
    
    # Extract final answer
    if best_params is not None:
        inner = []
        for i in range(N):
            x = best_params[3*i]
            y = best_params[3*i+1]
            a = best_params[3*i+2]
            inner.append([x, y, a])
        center = (0.0, 0.0)
        side = best_L
        angle = 0.0
    else:
        # Fallback: linear arrangement
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        center = (0.0, 0.0)
        side = 40.0
        angle = 0.0
    
    return inner, center, side, angle

# EVOLVE-BLOCK-START
# (The construct function is defined above)
# EVOLVE-BLOCK-END
