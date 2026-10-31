"""Optimized hexagon packing: minimize outer hexagon side length for N unit hexagons."""
import numpy as np
import time
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START

def hexagon_vertices(cx, cy, s, angle_deg):
    """Return 6 vertices of a regular hexagon with side s, center (cx,cy), rotation angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.pi/3 * np.arange(6)
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

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
                    S_needed = 2.0 * d / np.sqrt(3.0)
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

def objective(params, deadline):
    """Objective function: outer side length + penalty for overlaps."""
    if time.time() > deadline:
        return float('inf')
    
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
            d = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
            if d < 2.5:
                v1 = hexagon_vertices(cx1, cy1, 1.0, a1)
                v2 = hexagon_vertices(cx2, cy2, 1.0, a2)
                overlap, _ = polygons_overlap(v1, v2, tol=1e-8)
                if overlap:
                    penalty += (2.0 - d) ** 2 * 100
    
    return S + penalty

def construct():
    """Main entry point: find optimal packing of N unit hexagons in minimum outer hexagon."""
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit
    
    # Hexagonal lattice positions for N=11
    # Center + ring of 6 + 4 from ring 2
    hex_positions = [(0.0, 0.0)]
    for k in range(6):
        angle = np.radians(60 * k)
        x = np.sqrt(3) * np.cos(angle)
        y = np.sqrt(3) * np.sin(angle)
        hex_positions.append((x, y))
    for k in range(6):
        angle = np.radians(60 * k)
        x = 2 * np.sqrt(3) * np.cos(angle)
        y = 2 * np.sqrt(3) * np.sin(angle)
        hex_positions.append((x, y))
        if len(hex_positions) >= N:
            break
    
    hex_positions = hex_positions[:N]
    
    best_result = None
    best_score = float('inf')
    
    # Try multiple initial conditions
    initial_conditions = []
    
    # Condition 1: Hexagonal lattice
    ic1 = np.zeros(3 * N + 3)
    for i, (px, py) in enumerate(hex_positions):
        ic1[3*i] = px
        ic1[3*i+1] = py
        ic1[3*i+2] = 0.0
    ic1[3*N] = 0.0
    ic1[3*N+1] = 0.0
    ic1[3*N+2] = 0.0
    initial_conditions.append(ic1)
    
    # Condition 2: Perturbed hexagonal lattice
    np.random.seed(42)
    ic2 = ic1 + np.random.randn(3*N+3) * 0.1
    initial_conditions.append(ic2)
    
    # Condition 3: Different outer angle
    ic3 = ic1.copy()
    ic3[3*N+2] = np.radians(30)
    initial_conditions.append(ic3)
    
    # Condition 4: Shifted center
    ic4 = ic1.copy()
    ic4[3*N] = 0.5
    ic4[3*N+1] = 0.5
    initial_conditions.append(ic4)
    
    # Condition 5: Tighter hexagonal lattice (scale 0.99)
    ic5 = np.zeros(3 * N + 3)
    tight_scale = 0.99
    tight_positions = [(0.0, 0.0)]
    for k in range(6):
        angle = np.radians(60 * k)
        x = np.sqrt(3) * tight_scale * np.cos(angle)
        y = np.sqrt(3) * tight_scale * np.sin(angle)
        tight_positions.append((x, y))
    for k in range(6):
        angle = np.radians(60 * k)
        x = 2 * np.sqrt(3) * tight_scale * np.cos(angle)
        y = 2 * np.sqrt(3) * tight_scale * np.sin(angle)
        tight_positions.append((x, y))
        if len(tight_positions) >= N:
            break
    tight_positions = tight_positions[:N]
    for i, (px, py) in enumerate(tight_positions):
        ic5[3*i] = px
        ic5[3*i+1] = py
        ic5[3*i+2] = 0.0
    ic5[3*N] = 0.0
    ic5[3*N+1] = 0.0
    ic5[3*N+2] = 0.0
    initial_conditions.append(ic5)
    
    # Condition 6: Rotated inner hexagons
    ic6 = ic1.copy()
    for i in range(N):
        ic6[3*i+2] = np.radians(30)
    initial_conditions.append(ic6)
    
    # Condition 7: Slightly perturbed with different seed
    np.random.seed(123)
    ic7 = ic1 + np.random.randn(3*N+3) * 0.05
    initial_conditions.append(ic7)
    
    for ic in initial_conditions:
        if time.time() > deadline - 25:
            break
        
        try:
            res = minimize(objective, ic, args=(deadline,), method='L-BFGS-B', 
                          options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if res.success or res.fun < best_score:
                params = res.x
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
        except Exception:
            pass
    
    # Fallback
    if best_result is None:
        inner = [[px, py, 0.0] for px, py in hex_positions]
        S = compute_outer_side_length(inner, (0.0, 0.0), 0.0) + 0.1
        best_result = (inner, (0.0, 0.0), S, 0.0)
    
    # Refine outer hexagon parameters using Nelder-Mead
    if time.time() < deadline - 20:
        try:
            inner_fixed = best_result[0]
            
            def outer_obj(params_outer):
                ox, oy, oa = params_outer
                S = compute_outer_side_length(inner_fixed, (ox, oy), oa)
                return S
            
            outer_ic = np.array([best_result[1][0], best_result[1][1], np.radians(best_result[3])])
            res_outer = minimize(outer_obj, outer_ic, method='Nelder-Mead',
                               options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12})
            
            if res_outer.fun < best_score:
                ox, oy, oa = res_outer.x
                outer_angle = np.degrees(oa) % 360
                S = res_outer.fun
                if check_validity(inner_fixed, (ox, oy), S, outer_angle):
                    best_result = (inner_fixed, (ox, oy), S, outer_angle)
                    best_score = S
        except Exception:
            pass
    
    # Final refinement of all parameters
    if time.time() < deadline - 15:
        try:
            params_best = np.zeros(3*N+3)
            for i, (cx, cy, ang) in enumerate(best_result[0]):
                params_best[3*i] = cx
                params_best[3*i+1] = cy
                params_best[3*i+2] = np.radians(ang)
            params_best[3*N] = best_result[1][0]
            params_best[3*N+1] = best_result[1][1]
            params_best[3*N+2] = np.radians(best_result[3])
            
            res = minimize(objective, params_best, args=(deadline,), method='L-BFGS-B',
                          options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-12})
            
            if res.fun < best_score:
                params = res.x
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
                    best_result = (inner, (ox, oy), S, outer_angle)
        except Exception:
            pass
    
    return best_result

# EVOLVE-BLOCK-END
