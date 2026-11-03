import numpy as np
from scipy.optimize import minimize

N = 12

def hex_vertices(center, side, angle):
    """
    Generate vertices of a regular hexagon.
    center: (x, y)
    side: length
    angle: rotation in radians
    """
    cx, cy = center
    angles = np.linspace(0, 2*np.pi, 7)[:-1] + angle
    return np.array([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ]).T

def point_in_hexagon(point, center, side, angle):
    """
    Check if a point is inside a regular hexagon.
    Returns True if inside.
    """
    px, py = point
    cx, cy = center
    # Translate point to hexagon center
    dx = px - cx
    dy = py - cy
    # Rotate point by -angle to align hexagon with axes
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    rx = dx * cos_a - dy * sin_a
    ry = dx * sin_a + dy * cos_a
    
    # Inradius r = side * sqrt(3) / 2
    r = side * np.sqrt(3) / 2
    
    # Check against 6 sides
    # The hexagon is defined by |x| <= r, |y| <= r, |x+y| <= r*sqrt(3)? No.
    # For a hexagon with flat top/bottom (angle 0):
    # Vertices at (s, 0), (s/2, s*sqrt(3)/2), ...
    # The constraints are:
    # 1. |x| <= r  (Left/Right sides? No, vertices are at s. Sides are vertical? No.)
    # Let's use the normal form.
    # Normals are at 0, 60, 120, 180, 240, 300 degrees.
    # Distance from center to each side is r.
    # So dot product of (rx, ry) with normal must be <= r.
    
    normals = np.array([
        [1, 0],
        [0.5, np.sqrt(3)/2],
        [-0.5, np.sqrt(3)/2],
        [-1, 0],
        [-0.5, -np.sqrt(3)/2],
        [0.5, -np.sqrt(3)/2]
    ])
    
    dots = normals.dot([rx, ry])
    return np.all(dots <= r + 1e-9)

def objective_and_constraints(params):
    """
    params: [x1, y1, ..., x12, y12, S, phi]
    Returns (objective, constraints)
    """
    n_vars = 2 * N
    xs = params[0:n_vars:2]
    ys = params[1:n_vars:2]
    S = params[n_vars]
    phi = params[n_vars + 1]
    
    # Objective: S
    obj = S
    
    # Constraints:
    # 1. Non-overlap: distance >= 2
    cons = []
    for i in range(N):
        for j in range(i + 1, N):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist_sq = dx*dx + dy*dy
            # dist >= 2 => dist_sq >= 4
            cons.append(dist_sq - 4.0)
            
    # 2. Inside outer hexagon
    # Outer hexagon centered at (0,0), side S, angle phi
    for i in range(N):
        if not point_in_hexagon((xs[i], ys[i]), (0.0, 0.0), S, phi):
            # We need a continuous constraint for the optimizer.
            # Use the minimum slack over the 6 sides.
            # Slack = r - dot(p, n)
            # We want min_slack >= 0.
            
            dx = xs[i]
            dy = ys[i]
            cos_a = np.cos(-phi)
            sin_a = np.sin(-phi)
            rx = dx * cos_a - dy * sin_a
            ry = dx * sin_a + dy * cos_a
            
            r = S * np.sqrt(3) / 2
            
            normals = np.array([
                [1, 0],
                [0.5, np.sqrt(3)/2],
                [-0.5, np.sqrt(3)/2],
                [-1, 0],
                [-0.5, -np.sqrt(3)/2],
                [0.5, -np.sqrt(3)/2]
            ])
            
            dots = normals.dot([rx, ry])
            slacks = r - dots
            min_slack = np.min(slacks)
            cons.append(min_slack)
            
    return obj, cons

def construct():
    # Initial guess: 3x4 grid
    # Centers for 3x4 hex lattice
    # Row 0: y=0, x=0, 2, 4, 6
    # Row 1: y=sqrt(3), x=1, 3, 5, 7
    # Row 2: y=2*sqrt(3), x=0, 2, 4, 6
    
    sqrt3 = np.sqrt(3)
    init_xs = []
    init_ys = []
    
    for k in range(3):
        y = k * sqrt3
        if k % 2 == 0:
            xs = [0, 2, 4, 6]
        else:
            xs = [1, 3, 5, 7]
        for x in xs:
            init_xs.append(x)
            init_ys.append(y)
            
    # Center the grid
    mean_x = np.mean(init_xs)
    mean_y = np.mean(init_ys)
    init_xs = [x - mean_x for x in init_xs]
    init_ys = [y - mean_y for y in init_ys]
    
    # Estimate S
    # Max radius of points
    max_r = np.max(np.sqrt(np.array(init_xs)**2 + np.array(init_ys)**2))
    # Hexagon inradius r = S*sqrt(3)/2
    # We need r >= max_r * some factor
    # For a point on the axis, S >= max_r / (sqrt(3)/2) ? No.
    # If point is at (R, 0), it needs to be inside.
    # Max x in hex is S (vertex). But sides are closer.
    # Let's just guess S = 4.0
    S_init = 4.0
    phi_init = 0.0
    
    x0 = init_xs + init_ys + [S_init, phi_init]
    
    # Bounds for S
    # S must be positive
    bounds = []
    for _ in range(N):
        bounds.append((-10, 10)) # x
        bounds.append((-10, 10)) # y
    bounds.append((3.0, 10.0)) # S
    bounds.append((-np.pi, np.pi)) # phi
    
    # Run optimization
    # Use SLSQP
    res = None
    try:
        res = minimize(
            lambda p: objective_and_constraints(p)[0],
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=[
                {'type': 'ineq', 'fun': lambda p: objective_and_constraints(p)[1]}
            ],
            options={'maxiter': 500, 'ftol': 1e-9}
        )
    except Exception as e:
        print(f"Optimization error: {e}")
        # Fallback to initial guess if it's valid
        pass
    
    if res is None or not res.success:
        # Try a different initial guess or method
        # Or just return a known good solution
        # Let's try Trust-Constr
        try:
            res = minimize(
                lambda p: objective_and_constraints(p)[0],
                x0,
                method='trust-constr',
                bounds=bounds,
                constraints=[
                    {'type': 'ineq', 'fun': lambda p: objective_and_constraints(p)[1]}
                ],
                options={'maxiter': 500}
            )
        except Exception as e:
            print(f"Optimization error 2: {e}")
            
    if res is not None and res.success:
        best_params = res.x
        best_S = best_params[2*N]
    else:
        # Fallback: Use the initial guess if it's valid, otherwise a large S
        # Check if initial guess is valid
        obj, cons = objective_and_constraints(x0)
        if np.all(np.array(cons) >= -1e-6):
            best_S = x0[2*N]
        else:
            best_S = 5.0 # Safe fallback

    # Construct the final result
    # We need to return:
    # inner: list of [x, y, angle_degrees]
    # center: (x, y) of outer hex
    # side: outer side length
    # angle: outer rotation
    
    # The optimizer found positions relative to (0,0) and outer hex at (0,0)
    # We can just return these.
    
    inner = []
    for i in range(N):
        x = best_params[2*i]
        y = best_params[2*i + 1]
        angle_deg = 0.0 # Inner hexagons are aligned
        inner.append([x, y, angle_deg])
        
    center = (0.0, 0.0)
    side = best_S
    angle = best_params[2*N + 1] # Outer hex rotation
    
    return inner, center, side, angle

# EVOLVE-BLOCK-END
