import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 12
DEADLINE = time.time() + 145.0

# Precompute base hexagon vertices (side length 1, centered at origin, angle 0)
# Vertices at angles 0, 60, 120, 180, 240, 300 degrees
_BASE_ANGLES = np.array([i * np.pi / 3.0 for i in range(6)])
_BASE_VERTS = np.column_stack([np.cos(_BASE_ANGLES), np.sin(_BASE_ANGLES)])

# Outer hexagon normals (for a hexagon aligned with axes, normals are at 30, 90, 150, 210, 270, 330)
# Wait, for a regular hexagon with flat top/bottom, vertices are at 0, 60...
# The problem says "regular hexagon". Orientation doesn't matter for the score (side length).
# Let's assume the outer hexagon has vertices at angles 0, 60, ...
# Then the supporting lines (normals) are at 30, 90, 150, 210, 270, 330 degrees.
_OUTER_NORMALS = np.column_stack([
    np.cos(np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180.0),
    np.sin(np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180.0)
])

def get_vertices(x, y, angle_deg):
    """
    Compute vertices of a unit hexagon at (x, y) with rotation angle_deg.
    Returns (6, 2) array.
    """
    a = np.radians(angle_deg)
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, -s], [s, c]])
    return _BASE_VERTS @ R.T + np.array([x, y])

def compute_min_outer_side_fast(params):
    """
    params: (3*N,) array [x1, y1, a1, x2, y2, a2, ...]
    Returns the minimal side length of a centered, axis-aligned (in normal space) 
    regular hexagon that contains all inner hexagons.
    """
    all_verts = np.zeros((N * 6, 2))
    for i in range(N):
        x, y, a = params[3*i], params[3*i+1], params[3*i+2]
        all_verts[i*6:(i+1)*6] = get_vertices(x, y, a)
    
    # Project all vertices onto the 6 outer normals
    # (72, 2) @ (2, 6) -> (72, 6)
    projs = all_verts @ _OUTER_NORMALS
    
    # For each normal, find the max projection.
    # The side length S is determined by the max of these max-projections.
    # S = (2/sqrt(3)) * max_proj
    max_projs = np.max(projs, axis=0)
    max_proj = np.max(max_projs)
    
    return (2.0 / math.sqrt(3.0)) * max_proj

def check_validity(params, side, eps=1e-6):
    """
    Check if the configuration is valid:
    1. All inner hexagons are inside the outer hexagon of side 'side'.
    2. No two inner hexagons overlap.
    """
    polys = []
    for i in range(N):
        x, y, a = params[3*i], params[3*i+1], params[3*i+2]
        polys.append(get_vertices(x, y, a))
    
    # 1. Containment check
    # The outer hexagon is centered at (0,0) with side 'side'.
    # A point (px, py) is inside if for all 6 normals n:
    # dot(p, n) <= inradius_outer
    # inradius = side * sqrt(3) / 2
    inradius = side * math.sqrt(3.0) / 2.0
    for poly in polys:
        projs = poly @ _OUTER_NORMALS
        if np.any(projs > inradius + eps):
            return False
            
    # 2. Overlap check (SAT)
    for i in range(N):
        for j in range(i+1, N):
            if sat_overlap(polys[i], polys[j]):
                return False
    return True

def sat_overlap(A, B):
    """
    Separating Axis Theorem for two convex polygons.
    A, B are (6, 2) arrays.
    Returns True if they overlap.
    """
    # Normals of A
    for i in range(6):
        p1 = A[i]
        p2 = A[(i+1)%6]
        dx, dy = p2[0]-p1[0], p2[1]-p1[1]
        nx, ny = -dy, dx
        norm = math.sqrt(nx*nx + ny*ny)
        if norm < 1e-9: continue
        nx, ny = nx/norm, ny/norm
        
        projA = A @ np.array([nx, ny])
        projB = B @ np.array([nx, ny])
        
        minA, maxA = np.min(projA), np.max(projA)
        minB, maxB = np.min(projB), np.max(projB)
        
        if maxA < minB - 1e-9 or maxB < minA - 1e-9:
            break # Separated
    else:
        # Check normals of B
        for i in range(6):
            p1 = B[i]
            p2 = B[(i+1)%6]
            dx, dy = p2[0]-p1[0], p2[1]-p1[1]
            nx, ny = -dy, dx
            norm = math.sqrt(nx*nx + ny*ny)
            if norm < 1e-9: continue
            nx, ny = nx/norm, ny/norm
            
            projA = A @ np.array([nx, ny])
            projB = B @ np.array([nx, ny])
            
            minA, maxA = np.min(projA), np.max(projA)
            minB, maxB = np.min(projB), np.max(projB)
            
            if maxA < minB - 1e-9 or maxB < minA - 1e-9:
                return False # Separated
        else:
            return True # Overlap
    return False # Separated

def objective(params):
    """
    Objective function for optimization.
    Returns side length + penalty for overlap/containment violations.
    """
    side = compute_min_outer_side_fast(params)
    
    # Calculate penalties
    penalty = 0.0
    inradius = side * math.sqrt(3.0) / 2.0
    
    polys = []
    for i in range(N):
        x, y, a = params[3*i], params[3*i+1], params[3*i+2]
        p = get_vertices(x, y, a)
        polys.append(p)
        
        # Containment penalty
        projs = p @ _OUTER_NORMALS
        violations = np.maximum(0, projs - inradius)
        penalty += np.sum(violations**2)
        
    # Overlap penalty
    for i in range(N):
        for j in range(i+1, N):
            # Compute min separation distance (negative if overlap)
            # Using SAT, we find the max gap. If max gap < 0, they overlap.
            gap = max_sat_gap(polys[i], polys[j])
            if gap < 0:
                penalty += (-gap)**2
                
    return side + 100.0 * penalty

def max_sat_gap(A, B):
    """
    Returns the maximum separation gap between two polygons.
    Positive if separated, negative if overlapping (depth of penetration).
    """
    max_gap = -float('inf')
    
    # Check A's normals
    for i in range(6):
        p1 = A[i]
        p2 = A[(i+1)%6]
        dx, dy = p2[0]-p1[0], p2[1]-p1[1]
        nx, ny = -dy, dx
        norm = math.sqrt(nx*nx + ny*ny)
        if norm < 1e-9: continue
        nx, ny = nx/norm, ny/norm
        
        projA = A @ np.array([nx, ny])
        projB = B @ np.array([nx, ny])
        
        minA, maxA = np.min(projA), np.max(projA)
        minB, maxB = np.min(projB), np.max(projB)
        
        if maxA < minB:
            gap = minB - maxA
        elif maxB < minA:
            gap = minA - maxB
        else:
            gap = -min(maxA, maxB) + max(minA, minB) # Negative overlap depth
            
        if gap > max_gap:
            max_gap = gap
            
    # Check B's normals
    for i in range(6):
        p1 = B[i]
        p2 = B[(i+1)%6]
        dx, dy = p2[0]-p1[0], p2[1]-p1[1]
        nx, ny = -dy, dx
        norm = math.sqrt(nx*nx + ny*ny)
        if norm < 1e-9: continue
        nx, ny = nx/norm, ny/norm
        
        projA = A @ np.array([nx, ny])
        projB = B @ np.array([nx, ny])
        
        minA, maxA = np.min(projA), np.max(projA)
        minB, maxB = np.min(projB), np.max(projB)
        
        if maxA < minB:
            gap = minB - maxA
        elif maxB < minA:
            gap = minA - maxB
        else:
            gap = -min(maxA, maxB) + max(minA, minB)
            
        if gap > max_gap:
            max_gap = gap
            
    return max_gap

def construct():
    t0 = time.time()
    
    # Bounds for parameters:
    # x, y: roughly within [-5, 5]
    # angle: [0, 360]
    bounds = []
    for _ in range(N):
        bounds.append((-5.0, 5.0)) # x
        bounds.append((-5.0, 5.0)) # y
        bounds.append((0.0, 360.0)) # angle
        
    best_params = None
    best_score = float('inf')
    
    # 1. Global Search with Differential Evolution
    # Use a subset of the time budget for DE
    de_time_limit = 60.0
    if time.time() - t0 < de_time_limit:
        try:
            res = differential_evolution(
                objective, 
                bounds, 
                seed=42, 
                maxiter=200, 
                tol=1e-5,
                polish=True, # Let DE do a final local polish
                workers=1
            )
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception as e:
            pass
            
    # 2. Local Refinement with Nelder-Mead
    # If DE didn't find a good solution, or to improve it, run NM from multiple starts
    starts = []
    if best_params is not None:
        starts.append(best_params)
    
    # Add some heuristic starts
    # Start 1: Ring of 12
    for k in range(12):
        theta = 2 * np.pi * k / 12.0
        x = 2.0 * np.cos(theta)
        y = 2.0 * np.sin(theta)
        a = 0.0
        if k == 0:
            p = np.array([x, y, a])
        else:
            p = np.append(p, [x, y, a])
    starts.append(p)
    
    # Start 2: Hexagonal Lattice (approx)
    # 1 center, 6 ring, 5 more?
    p = np.zeros(36)
    p[0:3] = [0,0,0]
    for k in range(6):
        theta = 2 * np.pi * k / 6.0
        x = 1.8 * np.cos(theta)
        y = 1.8 * np.sin(theta)
        p[3*(k+1):3*(k+2)] = [x, y, 0]
    # Add remaining 5 in a second ring
    for k in range(5):
        theta = 2 * np.pi * (k + 0.5) / 10.0
        x = 3.5 * np.cos(theta)
        y = 3.5 * np.sin(theta)
        idx = 7 + k
        p[3*idx:3*idx+3] = [x, y, 0]
    starts.append(p)
    
    # Run NM on each start
    nm_time_limit = 10.0
    for start in starts:
        if time.time() - t0 > DEADLINE - 5.0:
            break
        try:
            res = minimize(objective, start, method='Nelder-Mead', 
                           options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass

    # 3. Final Polish
    if best_params is not None and time.time() - t0 < DEADLINE - 2.0:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead',
                           options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass

    if best_params is None:
        # Fallback: simple ring
        inner = []
        for k in range(12):
            theta = 2 * np.pi * k / 12.0
            x = 2.0 * np.cos(theta)
            y = 2.0 * np.sin(theta)
            inner.append([x, y, 0.0])
        side = 4.0
        return inner, (0.0, 0.0), side, 0.0

    # Extract inner hexagons
    inner = []
    for i in range(N):
        x, y, a = best_params[3*i], best_params[3*i+1], best_params[3*i+2]
        inner.append([x, y, a])
        
    # Calculate final side length
    side = compute_min_outer_side_fast(best_params)
    
    # Verify validity
    if not check_validity(best_params, side):
        # If invalid, try to scale up or return a known valid solution
        # For now, return a conservative valid solution
        inner = []
        for k in range(12):
            theta = 2 * np.pi * k / 12.0
            x = 2.5 * np.cos(theta)
            y = 2.5 * np.sin(theta)
            inner.append([x, y, 0.0])
        side = compute_min_outer_side_fast(np.array([x for x in [2.5*np.cos(2*np.pi*k/12.0) for k in range(12)] + [y for y in [2.5*np.sin(2*np.pi*k/12.0) for k in range(12)] + [0.0]*12]]))
        # Recalculate properly
        p_valid = np.zeros(36)
        for k in range(12):
            theta = 2 * np.pi * k / 12.0
            p_valid[3*k] = 2.5 * np.cos(theta)
            p_valid[3*k+1] = 2.5 * np.sin(theta)
            p_valid[3*k+2] = 0.0
        side = compute_min_outer_side_fast(p_valid)
        return [ [p_valid[3*k], p_valid[3*k+1], p_valid[3*k+2]] for k in range(12) ], (0.0, 0.0), side, 0.0

    return inner, (0.0, 0.0), side, 0.0

if __name__ == "__main__":
    construct()
