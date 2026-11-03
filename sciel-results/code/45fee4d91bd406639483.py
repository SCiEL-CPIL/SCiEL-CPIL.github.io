import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0
_SQRT3 = math.sqrt(3)
_INVSQRT3 = 1.0 / _SQRT3
_EPS = 1e-6

def _get_hex_support(directions, angle_deg=0.0):
    """
    Calculate support function of a unit hexagon (angle 0) for given directions.
    directions: array of angles.
    Returns array of max projections.
    """
    # Vertices of unit hexagon with angle 0 (vertex at 0)
    # Angles: 0, 60, 120, 180, 240, 300
    verts = np.array([[math.cos(a), math.sin(a)] for a in np.radians(np.arange(0, 360, 60))])
    # Rotate by angle_deg
    theta = math.radians(angle_deg)
    R = np.array([[math.cos(theta), -math.sin(theta)],
                  [math.sin(theta), math.cos(theta)]])
    verts = verts @ R.T
    
    # directions: (M, 2) unit vectors
    # Projection: verts @ directions.T -> (6, M)
    proj = verts @ directions.T
    return np.max(proj, axis=0)

def construct():
    if time.time() > _DEADLINE - 5.0:
        # Fallback
        return [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)], (0.0, 0.0), 40.0, 0.0

    # Initial guess: Hexagonal lattice
    u = np.array([_SQRT3, 0.0])
    v = np.array([_SQRT3/2, 1.5])
    
    # 1 center, 6 ring 1, 4 ring 2
    centers = []
    centers.append((0, 0))
    ring1 = [(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
    for i, j in ring1:
        centers.append(tuple(i*u + j*v))
    ring2 = [(2,0),(1,1),(0,2),(-1,2),(-2,1),(-2,0),(-1,-1),(0,-2),(1,-2),(2,-1)]
    # Pick 4 from ring 2 that are "closest" to center or symmetric
    # Let's pick (2,0), (0,2), (-2,0), (0,-2) for a diamond? No, let's just take the first 4 in a sorted list by distance
    ring2_sorted = sorted(ring2, key=lambda p: np.dot(i*u+j*v, i*u+j*v) if False else (p[0]**2+p[1]**2))
    # Actually, for 11 hexagons, a common optimal shape is a hexagon of side 2 (7 hexes) + 4 extra.
    # Let's use the 4 from ring 2 that are at distance 2 (all are).
    # To make it compact, let's pick (2,0), (1,1), (-1,2), (-2,1)? No, that's 4.
    # Let's try (2,0), (0,2), (-2,0), (0,-2) -> forms a cross?
    # Let's just use the first 4 from ring2_sorted.
    for i, j in ring2_sorted[:4]:
        centers.append(tuple(i*u + j*v))
        
    centers = np.array(centers)
    
    # Variables: Cx, Cy, S, x1..x11, y1..y11
    # Total 3 + 22 = 25 variables
    x0 = np.concatenate([
        [0.0, 0.0, 3.0], # Cx, Cy, S
        centers.flatten()
    ])
    
    # Precompute outer normals
    # Outer hexagon angle 0 means vertices at 0, 60, ...
    # Normals are at 30, 90, ...
    outer_normal_angles = np.radians(np.arange(30, 390, 60))
    outer_normals = np.column_stack([np.cos(outer_normal_angles), np.sin(outer_normal_angles)])
    
    # Precompute inner support function for these 6 directions
    # Inner hexagons are fixed at angle 0
    inner_support = _get_hex_support(outer_normals, angle_deg=0.0)
    
    def objective(params):
        if time.time() > _DEADLINE - 2.0:
            return 1e10
        Cx, Cy, S = params[0], params[1], params[2]
        if S < 1.0: return 1e10
        centers = params[3:].reshape(N, 2)
        
        # Check non-overlap
        # Min distance between centers of two unit hexagons (angle 0) is sqrt(3)
        # d_ij >= sqrt(3) + eps
        for i in range(N):
            for j in range(i+1, N):
                d2 = np.sum((centers[i] - centers[j])**2)
                if d2 < (_SQRT3 + _EPS)**2:
                    return 1e10 + (d2 - (_SQRT3 + _EPS)**2) # Penalty
        
        # Check containment
        # n_k . (c_i - C) + h_inner(k) <= S * sqrt(3)/2
        C = np.array([Cx, Cy])
        for i in range(N):
            rel = centers[i] - C
            # proj = rel @ outer_normals.T -> (6,)
            proj = rel @ outer_normals.T
            # Constraint: proj[k] + inner_support[k] <= S * _SQRT3 / 2
            violations = proj + inner_support - S * _SQRT3 / 2
            if np.any(violations > _EPS):
                return 1e10 + np.sum(violations[violations > _EPS])**2
        
        return S

    def constraints(params):
        if time.time() > _DEADLINE - 2.0:
            return [{} for _ in range(100)] # Dummy
        Cx, Cy, S = params[0], params[1], params[2]
        centers = params[3:].reshape(N, 2)
        cons = []
        
        # Non-overlap constraints
        for i in range(N):
            for j in range(i+1, N):
                d2 = np.sum((centers[i] - centers[j])**2)
                cons.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: np.sum((x[3+2*i:3+2*i+2] - x[3+2*j:3+2*j+2])**2) - (_SQRT3 + _EPS)**2
                })
                
        # Containment constraints
        C = np.array([Cx, Cy])
        for i in range(N):
            for k in range(6):
                n = outer_normals[k]
                h = inner_support[k]
                cons.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, n=n, h=h: x[2] * _SQRT3 / 2 - (n[0]*(x[3+2*i] - x[0]) + n[1]*(x[3+2*i+1] - x[1]) + h)
                })
        return cons

    # Optimize
    try:
        res = minimize(objective, x0, method='SLSQP', options={'maxiter': 500, 'ftol': 1e-12})
        if res.success or (res.fun < 1e9):
            best_S = res.fun
            best_C = (res.x[0], res.x[1])
            best_centers = res.x[3:].reshape(N, 2)
            best_angles = [0.0] * N
            # Return format: inner (x, y, angle), center, side, angle
            inner = [[best_centers[i][0], best_centers[i][1], best_angles[i]] for i in range(N)]
            return inner, best_C, best_S, 0.0
    except Exception as e:
        pass

    # Fallback to simple grid if optimization fails
    return [[4.0*k, 0.0, 0.0] for k in range(-(N//2), N - N//2)], (0.0, 0.0), 40.0, 0.0
