"""Optimized hexagon packing: minimize outer hexagon side length for N unit hexagons."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon as (x, y) tuples."""
    angles = angle_rad + np.pi / 3.0 * np.arange(6)
    return [(cx + side * math.cos(a), cy + side * math.sin(a)) for a in angles]

def compute_required_side(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_deg)
    outer_center: (x, y)
    outer_angle_deg: rotation of the outer hexagon
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_projection = 0.0
    
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj
    
    return 2.0 * max_projection / SQRT3

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=50.0):
    """
    Smooth approximation using log-sum-exp for the max projection.
    This provides better gradient information for optimization.
    """
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    projections = []
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                projections.append(proj)
    
    if not projections:
        return 0.0
    
    max_proj = max(projections)
    if max_proj < -1000:
        return 0.0
    
    scaled = [beta * (p - max_proj) for p in projections]
    # Clip to avoid overflow
    scaled = [max(-500, min(500, s)) for s in scaled]
    sum_exp = sum(math.exp(s) for s in scaled)
    smooth_max = max_proj + math.log(sum_exp) / beta
    
    return 2.0 * smooth_max / SQRT3

def sat_overlap_check(v1, v2, eps=1e-9):
    """
    Check if two convex polygons (defined by their vertices) overlap using SAT.
    v1, v2: lists of (x, y) tuples.
    Returns True if they overlap (including touching).
    """
    axes = []
    
    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    for i in range(len(v2)):
        p1 = v2[i]
        p2 = v2[(i + 1) % len(v2)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    for axis in axes:
        min1 = min(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        max1 = max(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        min2 = min(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        max2 = max(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    
    return True

def check_all_disjoint(inner, eps=1e-6):
    """Check if all inner hexagons are pairwise disjoint using SAT."""
    n = len(inner)
    for i in range(n):
        vi = get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2]))
        for j in range(i + 1, n):
            vj = get_hex_vertices(inner[j][0], inner[j][1], 1.0, math.radians(inner[j][2]))
            if sat_overlap_check(vi, vj, eps):
                return False
    return True

def check_inside(inner, outer_center, outer_side, outer_angle_deg, eps=1e-6):
    """Check if all inner hexagons are inside the outer hexagon."""
    ox, oy = outer_center
    outer_verts = get_hex_vertices(ox, oy, outer_side, math.radians(outer_angle_deg))
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for (vx, vy) in verts:
            inside = True
            for i in range(6):
                p1 = outer_verts[i]
                p2 = outer_verts[(i + 1) % 6]
                ex = p2[0] - p1[0]
                ey = p2[1] - p1[1]
                px = vx - p1[0]
                py = vy - p1[1]
                cross = ex * py - ey * px
                if cross < -eps:
                    inside = False
                    break
            if not inside:
                return False
    return True

def check_validity(inner, outer_center, outer_side, outer_angle_deg):
    """Check if the configuration is valid: no overlaps, all inside outer hexagon."""
    if not check_all_disjoint(inner):
        return False
    if not check_inside(inner, outer_center, outer_side, outer_angle_deg):
        return False
    return True

def build_initial_conditions():
    """Generate diverse initial conditions for optimization."""
    ics = []
    
    # Arrangement 1: Standard hexagonal lattice (axial coords)
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, -1), (1, 1), (-2, 1), (-1, -1)
    ]
    
    def axial_to_cartesian(q, r, scale=1.0):
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        return (x, y)
    
    # IC1: Standard lattice, scale 1.0
    ic = np.zeros(3 * N + 3)
    for i, (q, r) in enumerate(axial_coords[:N]):
        x, y = axial_to_cartesian(q, r, 1.0)
        ic[3*i] = x
        ic[3*i+1] = y
        ic[3*i+2] = 0.0
    ic[3*N] = 0.0
    ic[3*N+1] = 0.0
    ic[3*N+2] = 0.0
    ics.append(ic)
    
    # IC2: Slightly tighter lattice (scale 0.99)
    ic = np.zeros(3 * N + 3)
    for i, (q, r) in enumerate(axial_coords[:N]):
        x, y = axial_to_cartesian(q, r, 0.99)
        ic[3*i] = x
        ic[3*i+1] = y
        ic[3*i+2] = 0.0
    ic[3*N] = 0.0
    ic[3*N+1] = 0.0
    ic[3*N+2] = 0.0
    ics.append(ic)
    
    # IC3: Rotated outer hexagon (15 deg)
    ic = np.zeros(3 * N + 3)
    for i, (q, r) in enumerate(axial_coords[:N]):
        x, y = axial_to_cartesian(q, r, 1.0)
        ic[3*i] = x
        ic[3*i+1] = y
        ic[3*i+2] = 0.0
    ic[3*N] = 0.0
    ic[3*N+1] = 0.0
    ic[3*N+2] = math.radians(15)
    ics.append(ic)
    
    # IC4: Perturbed lattice
    np.random.seed(42)
    ic = np.zeros(3 * N + 3)
    for i, (q, r) in enumerate(axial_coords[:N]):
        x, y = axial_to_cartesian(q, r, 1.0)
        ic[3*i] = x + np.random.randn() * 0.02
        ic[3*i+1] = y + np.random.randn() * 0.02
        ic[3*i+2] = np.random.randn() * 0.01
    ic[3*N] = 0.0
    ic[3*N+1] = 0.0
    ic[3*N+2] = 0.0
    ics.append(ic)
    
    # IC5: Different cluster - center + ring + outer partial
    positions = [(0.0, 0.0)]
    for k in range(6):
        angle = math.radians(60 * k)
        x = SQRT3 * math.cos(angle)
        y = SQRT3 * math.sin(angle)
        positions.append((x, y))
    for k in range(4):
        angle = math.radians(30 + 60 * k)
        x = 2 * SQRT3 * math.cos(angle)
        y = 2 * SQRT3 * math.sin(angle)
        positions.append((x, y))
    
    ic = np.zeros(3 * N + 3)
    for i, (x, y) in enumerate(positions[:N]):
        ic[3*i] = x
        ic[3*i+1] = y
        ic[3*i+2] = 0.0
    ic[3*N] = 0.0
    ic[3*N+1] = 0.0
    ic[3*N+2] = 0.0
    ics.append(ic)
    
    # IC6: Another variant with slight rotation of inner hexagons
    ic = np.zeros(3 * N + 3)
    for i, (q, r) in enumerate(axial_coords[:N]):
        x, y = axial_to_cartesian(q, r, 1.0)
        ic[3*i] = x
        ic[3*i+1] = y
        ic[3*i+2] = math.radians(30)
    ic[3*N] = 0.0
    ic[3*N+1] = 0.0
    ic[3*N+2] = math.radians(30)
    ics.append(ic)
    
    return ics

def objective_smooth(params, beta=50.0):
    """Objective function using smooth max for better gradients."""
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
    
    S = compute_required_side_smooth(inner, (ox, oy), outer_angle, beta)
    
    # Overlap penalty using center distance as a quick check
    penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            cx1, cy1, a1 = inner[i]
            cx2, cy2, a2 = inner[j]
            d = math.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
            if d < 2.0:
                v1 = get_hex_vertices(cx1, cy1, 1.0, math.radians(a1))
                v2 = get_hex_vertices(cx2, cy2, 1.0, math.radians(a2))
                overlap, _ = sat_overlap_check(v1, v2, 1e-8)
                if overlap:
                    penalty += (2.0 - d) ** 2 * 50
    
    return S + penalty

def objective(params):
    """Objective function: outer side length + penalty for overlaps."""
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
    
    S = compute_required_side(inner, (ox, oy), outer_angle)
    
    penalty = 0.0
    for i in range(n):
        for j in range(i+1, n):
            cx1, cy1, a1 = inner[i]
            cx2, cy2, a2 = inner[j]
            d = math.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
            if d < 2.0:
                v1 = get_hex_vertices(cx1, cy1, 1.0, math.radians(a1))
                v2 = get_hex_vertices(cx2, cy2, 1.0, math.radians(a2))
                overlap, _ = sat_overlap_check(v1, v2, 1e-8)
                if overlap:
                    penalty += (2.0 - d) ** 2 * 50
    
    return S + penalty

def extract_result(params):
    """Extract inner hexagons, outer center, side length, and angle from params."""
    inner = []
    for i in range(N):
        cx = params[3*i]
        cy = params[3*i+1]
        ang = np.degrees(params[3*i+2]) % 360
        inner.append([cx, cy, ang])
    ox = params[3*N]
    oy = params[3*N+1]
    outer_angle = np.degrees(params[3*N+2]) % 360
    S = compute_required_side(inner, (ox, oy), outer_angle)
    return inner, (ox, oy), S, outer_angle

def construct():
    """Main entry point: find optimal packing of N unit hexagons in minimum outer hexagon."""
    start_time = time.time()
    deadline = start_time + 140.0
    
    initial_conditions = build_initial_conditions()
    
    best_result = None
    best_score = float('inf')
    
    # Phase 1: Optimize each initial condition with smooth objective
    for idx, ic in enumerate(initial_conditions):
        if time.time() > deadline - 30:
            break
        
        try:
            # Use smooth objective for better gradients
            res = minimize(objective_smooth, ic, method='L-BFGS-B', 
                          options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
            
            params = res.x
            inner, oc, S, oa = extract_result(params)
            
            if check_validity(inner, oc, S, oa):
                if S < best_score:
                    best_score = S
                    best_result = (inner, oc, S, oa)
        except Exception:
            pass
    
    # Phase 2: Refine best result with exact objective
    if best_result is not None and time.time() < deadline - 20:
        try:
            params_best = np.zeros(3*N+3)
            for i, (cx, cy, ang) in enumerate(best_result[0]):
                params_best[3*i] = cx
                params_best[3*i+1] = cy
                params_best[3*i+2] = math.radians(ang)
            params_best[3*N] = best_result[1][0]
            params_best[3*N+1] = best_result[1][1]
            params_best[3*N+2] = math.radians(best_result[3])
            
            # Refine with exact objective
            res = minimize(objective, params_best, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
            
            params = res.x
            inner, oc, S, oa = extract_result(params)
            
            if check_validity(inner, oc, S, oa):
                if S < best_score:
                    best_score = S
                    best_result = (inner, oc, S, oa)
        except Exception:
            pass
    
    # Phase 3: Try additional refinements with different beta values
    if best_result is not None and time.time() < deadline - 10:
        for beta in [20.0, 100.0]:
            if time.time() > deadline - 5:
                break
            try:
                params_best = np.zeros(3*N+3)
                for i, (cx, cy, ang) in enumerate(best_result[0]):
                    params_best[3*i] = cx
                    params_best[3*i+1] = cy
                    params_best[3*i+2] = math.radians(ang)
                params_best[3*N] = best_result[1][0]
                params_best[3*N+1] = best_result[1][1]
                params_best[3*N+2] = math.radians(best_result[3])
                
                # Small perturbation to escape local minima
                params_perturbed = params_best + np.random.randn(3*N+3) * 0.005
                res = minimize(objective_smooth, params_perturbed, method='L-BFGS-B',
                              options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 30})
                
                params = res.x
                inner, oc, S, oa = extract_result(params)
                
                if check_validity(inner, oc, S, oa):
                    if S < best_score:
                        best_score = S
                        best_result = (inner, oc, S, oa)
            except Exception:
                pass
    
    # Fallback
    if best_result is None:
        axial_coords = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, -1), (1, 1), (-2, 1), (-1, -1)
        ]
        inner = []
        for q, r in axial_coords[:N]:
            x = 1.5 * q
            y = SQRT3 * (r + q / 2.0)
            inner.append([x, y, 0.0])
        S = compute_required_side(inner, (0.0, 0.0), 0.0) + 0.1
        best_result = (inner, (0.0, 0.0), S, 0.0)
    
    return best_result

# EVOLVE-BLOCK-END
