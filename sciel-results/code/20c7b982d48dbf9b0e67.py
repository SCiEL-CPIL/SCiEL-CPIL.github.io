import numpy as np
from scipy.optimize import minimize
import time

N = 12

def hex_vertices(cx, cy, side, angle_deg=0.0):
    """Return vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    a = np.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + np.pi / 3 * i
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        verts.append([x, y])
    return np.array(verts)

def polygons_overlap(v1, v2, eps=1e-9):
    """Check if two convex polygons overlap using SAT (Separating Axis Theorem)."""
    edges = []
    for v in (v1, v2):
        for i in range(len(v)):
            j = (i + 1) % len(v)
            edge = v[j] - v[i]
            # Normal to edge
            n = np.array([-edge[1], edge[0]])
            if np.linalg.norm(n) > eps:
                n = n / np.linalg.norm(n)
            edges.append(n)
    
    for n in edges:
        min1, max1 = np.min(v1 @ n), np.max(v1 @ n)
        min2, max2 = np.min(v2 @ n), np.max(v2 @ n)
        if max1 < min2 - eps or max2 < min1 - eps:
            return False
    return True

def compute_min_outer_side(inner_positions, inner_angles, outer_center, outer_angle, outer_side):
    """Check if all inner hexagons fit inside the outer hexagon. Return violation (0 if valid)."""
    outer_verts = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    # For each inner hexagon, check all vertices are inside outer
    # Also check no two inner hexagons overlap
    inner_verts_list = []
    for i in range(len(inner_positions)):
        v = hex_vertices(inner_positions[i][0], inner_positions[i][1], 1.0, inner_angles[i])
        inner_verts_list.append(v)
        # Check if all vertices of inner hex are inside outer hex
        for pt in v:
            if not point_in_convex_polygon(pt, outer_verts):
                return 1.0  # violation
    # Check pairwise overlaps
    for i in range(len(inner_verts_list)):
        for j in range(i+1, len(inner_verts_list)):
            if polygons_overlap(inner_verts_list[i], inner_verts_list[j]):
                return 1.0
    return 0.0

def point_in_convex_polygon(pt, poly):
    """Check if point is inside convex polygon using cross product method."""
    n = len(poly)
    sign = 0
    for i in range(n):
        j = (i + 1) % n
        edge = poly[j] - poly[i]
        vec = pt - poly[i]
        cross = edge[0] * vec[1] - edge[1] * vec[0]
        if abs(cross) < 1e-12:
            continue
        if cross > 0:
            if sign < 0:
                return False
            sign = 1
        else:
            if sign > 0:
                return False
            sign = -1
    return True

def objective(params):
    """Objective: minimize outer side length with penalty for violations."""
    # params: [x1,y1,a1, ..., x12,y12,a12, cx,cy, cangle, side]
    n_inner = N
    inner_positions = np.array(params[:3*n_inner]).reshape(n_inner, 2)
    inner_angles = params[3*n_inner:6*n_inner]
    cx, cy = params[6*n_inner:6*n_inner+2]
    cangle = params[6*n_inner+2]
    side = params[6*n_inner+3]
    
    if side < 0:
        return 1e6
    
    violation = compute_min_outer_side(inner_positions, inner_angles, (cx, cy), cangle, side)
    if violation > 0:
        return side + 100.0 * violation
    return side

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Initial arrangement: hexagonal lattice pattern
    # 12 hexagons: try a 3x4 or similar grid, or symmetric arrangement
    # Let's use a pattern: 1 in center, 6 around, 5 more
    # Better: use a triangular lattice
    
    # Symmetric initial guess: place hexagons in a hexagonal pattern
    # For 12 hexagons, a good pattern is two rings or a 4x3 arrangement
    
    # Try: 3 rows of 4, offset like hex packing
    positions = []
    angles = np.zeros(N)
    
    # Hexagonal close-packed arrangement
    # Row 0: 4 hexagons
    # Row 1: 4 hexagons (offset)
    # Row 2: 4 hexagons
    for row in range(3):
        for col in range(4):
            x = col * 1.8 + (0.9 if row % 2 == 1 else 0.0)
            y = row * 1.732
            positions.append([x, y])
    
    positions = np.array(positions)
    # Center the arrangement
    positions -= positions.mean(axis=0)
    
    # Initial outer hexagon
    cx, cy = 0.0, 0.0
    cangle = 0.0
    # Estimate side length
    max_dist = np.max(np.sqrt(positions[:,0]**2 + positions[:,1]**2)) + 1.0
    side = max_dist + 1.0
    
    # Initial params
    params0 = np.concatenate([positions.flatten(), angles, [cx, cy, cangle, side]])
    
    best_params = params0.copy()
    best_score = objective(params0)
    
    # Try multiple random restarts and optimizations
    np.random.seed(42)
    n_restarts = 8
    n_steps = 0
    
    for restart in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        if restart == 0:
            params = params0.copy()
        else:
            # Perturb initial positions
            perturbed_positions = positions + np.random.normal(0, 0.5, size=positions.shape)
            perturbed_angles = np.random.uniform(-30, 30, size=N)
            perturbed_side = side * np.random.uniform(0.8, 1.2)
            params = np.concatenate([perturbed_positions.flatten(), perturbed_angles, [cx, cy, cangle, perturbed_side]])
        
        # Use Nelder-Mead for optimization
        try:
            result = minimize(
                objective, params, method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            n_steps += 1
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
        
        if time.time() > deadline - 10.0:
            break
    
    # Extract best result
    n_inner = N
    inner_positions = best_params[:3*n_inner].reshape(n_inner, 2)
    inner_angles = best_params[3*n_inner:6*n_inner]
    cx, cy = best_params[6*n_inner:6*n_inner+2]
    cangle = best_params[6*n_inner+2]
    side = best_params[6*n_inner+3]
    
    # Ensure side is positive
    side = max(side, 3.0)
    
    # Build inner list
    inner = []
    for i in range(n_inner):
        inner.append([float(inner_positions[i][0]), float(inner_positions[i][1]), float(inner_angles[i])])
    
    return inner, (float(cx), float(cy)), float(side), float(cangle)

