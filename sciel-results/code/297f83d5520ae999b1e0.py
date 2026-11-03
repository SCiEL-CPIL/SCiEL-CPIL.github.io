import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def hexagon_corners(cx, cy, side, angle_deg):
    """Get the 6 corner points of a regular hexagon."""
    a = np.radians(angle_deg)
    corners = []
    for i in range(6):
        theta = a + i * np.pi / 3.0
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        corners.append((x, y))
    return corners

def all_inner_corners(positions):
    """Get all 6*N corner points of inner hexagons."""
    corners = []
    for x, y, ang in positions:
        cs = hexagon_corners(x, y, 1.0, ang)
        corners.extend(cs)
    return np.array(corners)

def smooth_max(values, beta=50.0):
    """Soft-max via log-sum-exp."""
    values = np.array(values, dtype=float)
    if values.size == 0:
        return 0.0
    vmax = np.max(values)
    if vmax < -100:
        return 0.0
    shifted = values - vmax
    result = vmax + np.log(np.sum(np.exp(beta * shifted)) / len(values)) / beta
    return result

def objective(params, beta=50.0):
    """
    params: [x1,y1,a1, ..., xN,yN,aN, cx, cy, side, outer_angle]
    Returns smooth surrogate for the minimum outer side length.
    """
    n = N
    inner_params = params[:3*n]
    cx = params[3*n]
    cy = params[3*n+1]
    side = params[3*n+2]
    outer_angle = params[3*n+3]
    
    if side < 0.1:
        return 1e6
    
    positions = []
    for i in range(n):
        x = inner_params[3*i]
        y = inner_params[3*i+1]
        ang = inner_params[3*i+2]
        positions.append((x, y, ang))
    
    corners = all_inner_corners(positions)
    
    a_rad = np.radians(outer_angle)
    apothem_factor = np.sqrt(3) / 2.0
    
    required_sides = []
    for px, py in corners:
        dx = px - cx
        dy = py - cy
        for i in range(6):
            theta = a_rad + i * np.pi / 3.0 + np.pi / 2
            nx = np.cos(theta)
            ny = np.sin(theta)
            dist = dx * nx + dy * ny
            if dist > 0:
                required_sides.append(dist / apothem_factor)
    
    if not required_sides:
        return side
    
    required_sides = np.array(required_sides)
    return smooth_max(required_sides, beta)

def construct():
    t0 = time.time()
    DEADLINE = 140.0
    
    sqrt3 = np.sqrt(3)
    
    def axial_to_cartesian(q, r):
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        return (x, y)
    
    axial_coords = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, -1), (2, -2), (-2, 1), (-2, 2)
    ]
    
    positions = []
    for q, r in axial_coords:
        x, y = axial_to_cartesian(q, r)
        positions.append([x, y, 0.0])
    
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx0 = (max(xs) + min(xs)) / 2.0
    cy0 = (max(ys) + min(ys)) / 2.0
    for p in positions:
        p[0] -= cx0
        p[1] -= cy0
    
    params0 = []
    for p in positions:
        params0.extend([p[0], p[1], p[2]])
    params0.extend([0.0, 0.0, 4.0, 0.0])
    
    bounds = []
    for i in range(N):
        bounds.append((-20, 20))
        bounds.append((-20, 20))
        bounds.append((-180, 180))
    bounds.append((-20, 20))
    bounds.append((-20, 20))
    bounds.append((1.0, 20))
    bounds.append((-180, 180))
    
    best_params = np.array(params0)
    best_obj = objective(best_params)
    
    rng = np.random.default_rng(42)
    n_restarts = 8
    
    for restart in range(n_restarts):
        if time.time() - t0 > DEADLINE - 30:
            break
        
        if restart == 0:
            x0 = np.array(params0)
        else:
            x0 = np.array(params0)
            for i in range(3 * N):
                if i % 3 == 2:
                    x0[i] += rng.uniform(-30, 30)
                else:
                    x0[i] += rng.uniform(-2, 2)
            x0[3*N] += rng.uniform(-1, 1)
            x0[3*N+1] += rng.uniform(-1, 1)
            x0[3*N+3] += rng.uniform(-30, 30)
        
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if result.fun < best_obj:
            best_obj = result.fun
            best_params = result.x
    
    if time.time() - t0 < DEADLINE - 10:
        result = minimize(
            objective,
            best_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        if result.fun < best_obj:
            best_obj = result.fun
            best_params = result.x
    
    inner = []
    for i in range(N):
        inner.append([best_params[3*i], best_params[3*i+1], best_params[3*i+2]])
    
    cx = best_params[3*N]
    cy = best_params[3*N+1]
    outer_angle = best_params[3*N+3]
    
    corners = all_inner_corners(inner)
    a_rad = np.radians(outer_angle)
    apothem_factor = np.sqrt(3) / 2.0
    max_side = 0.0
    for px, py in corners:
        dx = px - cx
        dy = py - cy
        for i in range(6):
            theta = a_rad + i * np.pi / 3.0 + np.pi / 2
            nx = np.cos(theta)
            ny = np.sin(theta)
            dist = dx * nx + dy * ny
            if dist > 0:
                s = dist / apothem_factor
                if s > max_side:
                    max_side = s
    
    final_side = max(max_side, 1.0) * 1.0001
    
    return inner, (cx, cy), final_side, outer_angle
# EVOLVE-BLOCK-END
