"""Optimized hexagon packing: minimize outer hexagon side length for N unit hexagons."""
import numpy as np
import time
import math
from scipy.optimize import minimize, differential_evolution

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def hexagon_vertices(cx, cy, s, angle_deg):
    """Return 6 vertices of a regular hexagon with side s, center (cx,cy), rotation angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.pi/3 * np.arange(6)
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon as list of (x, y) tuples."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts

def polygons_overlap(v1, v2, tol=1e-9):
    """Check if two convex polygons overlap using SAT."""
    axes = []
    for v in (v1, v2):
        n = len(v)
        for i in range(n):
            edge = v[(i+1) % n] - v[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                axes.append(normal / norm)
    
    for axis in axes:
        proj1 = np.dot(v1, axis)
        proj2 = np.dot(v2, axis)
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2 - tol or max2 < min1 - tol:
            return False, axis
    return True, None

def compute_outer_side_length(inner_hexagons, outer_center, outer_angle):
    """Compute minimum outer hexagon side length that contains all inner hexagons."""
    ox, oy = outer_center
    oa = np.radians(outer_angle)
    
    normals = []
    for k in range(6):
        theta = oa + np.radians(30 + 60 * k)
        normals.append(np.array([np.cos(theta), np.sin(theta)]))
    
    min_S = 0.0
    for cx, cy, ang in inner_hexagons:
        verts = hexagon_vertices(cx, cy, 1.0, ang)
        for v in verts:
            for n in normals:
                d = np.dot(v - np.array([ox, oy]), n)
                if d > 0:
                    S_needed = 2.0 * d / SQRT3
                    if S_needed > min_S:
                        min_S = S_needed
    return min_S

def check_validity(inner_hexagons, outer_center, outer_side, outer_angle):
    """Check if the configuration is valid: no overlaps, all inside outer hexagon."""
    ox, oy = outer_center
    
    # Check no overlaps between inner hexagons
    for i in range(len(inner_hexagons)):
        for j in range(i+1, len(inner_hexagons)):
            cx1, cy1, a1 = inner_hexagons[i]
            cx2, cy2, a2 = inner_hexagons[j]
            v1 = hexagon_vertices(cx1, cy1, 1.0, a1)
            v2 = hexagon_vertices(cx2, cy2, 1.0, a2)
            overlap, _ = polygons_overlap(v1, v2, tol=1e-8)
            if overlap:
                return False
    
    # Check all inner hexagons are inside outer hexagon
    outer_verts = hexagon_vertices(ox, oy, outer_side, outer_angle)
    
    for cx, cy, ang in inner_hexagons:
        verts = hexagon_vertices(cx, cy, 1.0, ang)
        for v in verts:
            px, py = v
            n = len(outer_verts)
            inside = True
            for i in range(n):
                v1 = outer_verts[i]
                v2 = outer_verts[(i+1) % n]
                edge = v2 - v1
                normal = np.array([-edge[1], edge[0]])
                d1 = np.dot(normal, np.array([px, py]) - v1)
                d2 = np.dot(normal, outer_verts[(i+2) % n] - v1)
                if d1 * d2 < 0:
                    inside = False
                    break
            if not inside:
                return False
    
    return True

def exact_objective(params):
    """Exact objective: outer side length + hard overlap penalty."""
    n = N
    inner = []
    for i in range(n):
        cx = params[3*i]
        cy = params[3*i+1]
        ang = np.degrees(params[3*i+2]) % 360
        inner.append((cx, cy, ang))
    ox = params[3*n]
    oy = params[3*n+1]
    outer_angle = np.degrees(params[3*n+2]) % 360
    
    S = compute_outer_side_length(inner, (ox, oy), outer_angle)
    
    penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            cx1, cy1, a1 = inner[i]
            cx2, cy2, a2 = inner[j]
            v1 = hexagon_vertices(cx1, cy1, 1.0, a1)
            v2 = hexagon_vertices(cx2, cy2, 1.0, a2)
            overlap, _ = polygons_overlap(v1, v2, tol=1e-8)
            if overlap:
                penalty += 10.0
    
    return S + penalty

def construct():
    """Main entry point: find optimal packing of N unit hexagons in minimum outer hexagon."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Hexagonal lattice positions for N=11 using axial coordinates
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (1, 1), (-2, 1), (-1, -1)
    ]
    
    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]
    
    # Convert axial to Cartesian with tight packing
    epsilon = 1e-6
    scale = 1.0 + epsilon
    
    centers = []
    for q, r in axial_coords:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        centers.append((x, y))
    
    cx_avg = sum(c[0] for c in centers) / len(centers)
    cy_avg = sum(c[1] for c in centers) / len(centers)
    
    best_result = None
    best_score = float('inf')
    
    def try_result(params):
        nonlocal best_result, best_score
        inner = []
        for i in range(N):
            cx = params[3*i]
            cy = params[3*i+1]
            ang = np.degrees(params[3*i+2]) % 360
            inner.append([cx, cy, ang])
        ox = params[3*N]
        oy = params[3*N+1]
        outer_angle = np.degrees(params[3*N+2]) % 360
        S = compute_outer_side_length(inner, (ox, oy), outer_angle)
        if check_validity(inner, (ox, oy), S, outer_angle):
            if S < best_score:
                best_score = S
                best_result = (inner, (ox, oy), S, outer_angle)
        return S
    
    # Phase 1: Differential Evolution global search
    np.random.seed(42)
    bounds = [(-4, 4)] * N + [(-4, 4)] * N + [(-math.pi, math.pi)] * N + [(-1, 1), (-1, 1), (-math.pi, math.pi)]
    # Reorder: for each hex: x, y, angle; then outer x, y, angle
    bounds = []
    for i in range(N):
        bounds.append((-4, 4))  # x
        bounds.append((-4, 4))  # y
        bounds.append((-math.pi, math.pi))  # angle
    bounds.append((-1, 1))  # outer x
    bounds.append((-1, 1))  # outer y
    bounds.append((-math.pi, math.pi))  # outer angle
    
    remaining = deadline - time.time()
    if remaining > 40:
        try:
            de_result = differential_evolution(
                exact_objective, bounds,
                seed=42, maxiter=100, popsize=15,
                tol=1e-8, mutation=(0.3, 1.0), recombination=0.8,
                polish=True, init='latinhypercube',
                updating='immediate'
            )
            if de_result.fun < best_score + 5:
                try_result(de_result.x)
        except Exception:
            pass
    
    # Phase 2: L-BFGS-B local refinement from multiple starts
    initial_conditions = []
    
    # IC1: Standard hexagonal lattice
    ic1 = np.zeros(3 * N + 3)
    for i, (px, py) in enumerate(centers):
        ic1[3*i] = px
        ic1[3*i+1] = py
        ic1[3*i+2] = 0.0
    ic1[3*N] = cx_avg
    ic1[3*N+1] = cy_avg
    ic1[3*N+2] = 0.0
    initial_conditions.append(ic1)
    
    # IC2: Perturbed lattice
    np.random.seed(42)
    ic2 = ic1 + np.random.randn(3*N+3) * 0.15
    ic2[3*N] = cx_avg
    ic2[3*N+1] = cy_avg
    initial_conditions.append(ic2)
    
    # IC3: Rotated inner hexagons
    ic3 = ic1.copy()
    for i in range(N):
        ic3[3*i+2] = np.radians(30)
    initial_conditions.append(ic3)
    
    # IC4: Different perturbation
    np.random.seed(123)
    ic4 = ic1 + np.random.randn(3*N+3) * 0.2
    ic4[3*N] = cx_avg
    ic4[3*N+1] = cy_avg
    initial_conditions.append(ic4)
    
    # IC5: Various outer angles
    for angle in [15, 30, 45, 60, 75, 90]:
        ic = ic1.copy()
        ic[3*N+2] = np.radians(angle)
        initial_conditions.append(ic)
    
    # If DE found something good, add it as a start
    if best_result is not None:
        ic_de = np.zeros(3*N+3)
        for i, (cx, cy, ang) in enumerate(best_result[0]):
            ic_de[3*i] = cx
            ic_de[3*i+1] = cy
            ic_de[3*i+2] = np.radians(ang)
        ic_de[3*N] = best_result[1][0]
        ic_de[3*N+1] = best_result[1][1]
        ic_de[3*N+2] = np.radians(best_result[3])
        initial_conditions.append(ic_de)
    
    for idx, ic in enumerate(initial_conditions):
        if time.time() > deadline - 20:
            break
        
        try:
            res = minimize(exact_objective, ic, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            try_result(res.x)
        except Exception:
            pass
    
    # Fallback
    if best_result is None:
        inner = [[px, py, 0.0] for px, py in centers]
        S = compute_outer_side_length(inner, (cx_avg, cy_avg), 0.0) + 0.1
        best_result = (inner, (cx_avg, cy_avg), S, 0.0)
        best_score = S
    
    # Phase 3: Nelder-Mead refinement on best result
    if time.time() < deadline - 10:
        try:
            params_best = np.zeros(3*N+3)
            for i, (cx, cy, ang) in enumerate(best_result[0]):
                params_best[3*i] = cx
                params_best[3*i+1] = cy
                params_best[3*i+2] = np.radians(ang)
            params_best[3*N] = best_result[1][0]
            params_best[3*N+1] = best_result[1][1]
            params_best[3*N+2] = np.radians(best_result[3])
            
            res = minimize(exact_objective, params_best, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10})
            try_result(res.x)
        except Exception:
            pass
    
    # Phase 4: Second Nelder-Mead with perturbation
    if time.time() < deadline - 5:
        try:
            params_best = np.zeros(3*N+3)
            for i, (cx, cy, ang) in enumerate(best_result[0]):
                params_best[3*i] = cx
                params_best[3*i+1] = cy
                params_best[3*i+2] = np.radians(ang)
            params_best[3*N] = best_result[1][0]
            params_best[3*N+1] = best_result[1][1]
            params_best[3*N+2] = np.radians(best_result[3])
            
            np.random.seed(777)
            params_best += np.random.randn(3*N+3) * 0.05
            
            res = minimize(exact_objective, params_best, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10})
            try_result(res.x)
        except Exception:
            pass
    
    # Final time check
    if time.time() > deadline:
        return best_result
    
    return best_result

# EVOLVE-BLOCK-END
