import math
import numpy as np
import time
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 20.0

def construct():
    # Initial guess: Triangular lattice points
    # Axial coordinates
    axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    
    pts = np.array([[math.sqrt(3) * (q + r / 2.0), 1.5 * r] for q, r in axial])
    
    # Center the points
    centroid = np.mean(pts, axis=0)
    pts = pts - centroid

    # Objective function: Given positions and container rotation, return S
    def calc_S(pos, theta):
        # pos is (N, 2)
        # theta is container rotation
        rel_pts = pos  # Already centered? We assume pos is relative to centroid
        # To be safe, center them
        rel_pts = rel_pts - np.mean(rel_pts, axis=0)
        
        normals = []
        for k in range(6):
            ang = theta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        max_S = 0.0
        for nx, ny in normals:
            projections = rel_pts[:, 0] * nx + rel_pts[:, 1] * ny
            abs_proj = np.max(np.abs(projections))
            required_S = (abs_proj + APOTHEM) / APOTHEM
            if required_S > max_S:
                max_S = required_S
        return max_S

    # Total objective: min over theta
    def objective(vars):
        # vars: x1..x11, y1..y11, theta
        n = len(vars)
        pos = vars[:2*N].reshape(N, 2)
        theta = vars[2*N]
        
        # Check time
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return 1e6
        
        return calc_S(pos, theta)

    # Constraints: Distance between any pair >= 2.0
    def dist_constraints(vars):
        pos = vars[:2*N].reshape(N, 2)
        cons = []
        for i in range(N):
            for j in range(i+1, N):
                # d_ij >= 2
                # 1 - (d_ij / 2) >= 0 ? No, d_ij - 2 >= 0
                # SLSQP requires g(vars) >= 0
                # g = d_ij - 2
                # We use 'ineq'
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                d = math.sqrt(dx*dx + dy*dy)
                cons.append({'type': 'ineq', 'fun': lambda v, i=i, j=j: math.sqrt((v[2*i]-v[2*j])**2 + (v[2*i+1]-v[2*j+1])**2) - 2.0})
        return cons

    # Initial vars
    x_coords = pts[:, 0]
    y_coords = pts[:, 1]
    init_vars = np.concatenate([x_coords, y_coords, [0.0]])
    
    # Bounds for theta
    # Positions can move, but keep them bounded to prevent drift
    bounds = [(-10, 10)] * (2*N) + [(0, math.pi/3)]
    
    try:
        res = minimize(objective, init_vars, method='SLSQP', 
                       constraints=dist_constraints(init_vars),
                       bounds=bounds, options={'maxiter': 100, 'ftol': 1e-4})
        
        if res.success and res.fun < calc_S(pts, 0.0):
            best_pos = res.x[:2*N].reshape(N, 2)
            best_theta = res.x[2*N]
            best_S = res.fun
        else:
            # Fallback to initial lattice
            best_pos = pts
            best_theta = 0.0
            best_S = calc_S(pts, 0.0)
            
            # Try a few manual rotations for the lattice
            for t in np.linspace(0, math.pi/6, 10):
                s_t = calc_S(pts, t)
                if s_t < best_S:
                    best_S = s_t
                    best_theta = t
                    
    except Exception:
        best_pos = pts
        best_theta = 0.0
        best_S = calc_S(pts, 0.0)

    # Finalize
    # Re-center best_pos
    best_pos = best_pos - np.mean(best_pos, axis=0)
    
    inner = []
    # Inner hexes are aligned with the container?
    # The problem says "each inner hexagon may translate and rotate".
    # If we assume they are aligned with the container, the angle is 0 relative to container.
    # The output angle is the rotation of the INNER hexagons.
    # If we assume optimal packing aligns them, angle = 0.
    # The container rotation is separate.
    angle_deg = 0.0
    
    for x, y in best_pos:
        inner.append([float(x), float(y), angle_deg])
        
    center = [0.0, 0.0] # We centered the points, so center is 0
    side = float(best_S)
    rotation = float(math.degrees(best_theta))
    
    return inner, center, side, rotation
