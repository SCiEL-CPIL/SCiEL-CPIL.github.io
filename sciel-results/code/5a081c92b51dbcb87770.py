"""
Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Combines multi-topology lattice packing with SLSQP optimization and smooth objectives.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = time.time() + 140.0  # safety margin before 150s kill

# --- Geometry helpers ---

def hex_vertices(cx, cy, s, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy), side s, rotation angle_deg."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
    return verts

def point_in_convex_polygon(px, py, poly):
    """True if (px,py) is inside or on boundary of convex polygon (CCW or CW)."""
    n = len(poly)
    sign = 0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if abs(cross) < 1e-9:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True

def polygons_overlap_sat(A, B):
    """SAT: True if two convex polygons intersect (including touching)."""
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            dx, dy = x2 - x1, y2 - y1
            nx, ny = -dy, dx
            if nx == 0 and ny == 0:
                continue
            minA = min(nx * px + ny * py for px, py in A)
            maxA = max(nx * px + ny * py for px, py in A)
            minB = min(nx * px + ny * py for px, py in B)
            maxB = max(nx * px + ny * py for px, py in B)
            if maxA < minB - 1e-9 or maxB < minA - 1e-9:
                return False
    return True

def check_validity(inner, outer_center, outer_side, outer_angle, eps=1e-7):
    """Check all inner hexagons are inside outer and pairwise non-overlapping."""
    outer = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    polys = []
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        polys.append(p)
    for p in polys:
        for (px, py) in p:
            if not point_in_convex_polygon(px, py, outer):
                return False
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            if polygons_overlap_sat(polys[i], polys[j]):
                return False
    return True

def compute_min_outer_side(inner, outer_center, outer_angle):
    """Minimum outer side length such that all unit hexagons fit inside."""
    cx, cy = outer_center
    a = math.radians(outer_angle)
    normals = []
    for i in range(6):
        theta = a + (i + 0.5) * math.pi / 3.0
        normals.append((math.cos(theta), math.sin(theta)))
    max_proj = 0.0
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        for (px, py) in p:
            dx, dy = px - cx, py - cy
            for (nx, ny) in normals:
                proj = dx * nx + dy * ny
                if proj > max_proj:
                    max_proj = proj
    return (2.0 / math.sqrt(3.0)) * max_proj

# --- Topology builders ---

def build_ring12(scale, inner_rot, outer_rot):
    inner = []
    for k in range(12):
        theta = math.radians(inner_rot + k * 30.0)
        x = scale * math.cos(theta)
        y = scale * math.sin(theta)
        inner.append((x, y, inner_rot))
    return inner, (0.0, 0.0), outer_rot

def build_hex6_plus6(scale1, scale2, rot1, rot2, outer_rot):
    inner = []
    for k in range(6):
        theta = math.radians(rot1 + k * 60.0)
        x = scale1 * math.cos(theta)
        y = scale1 * math.sin(theta)
        inner.append((x, y, rot1))
    for k in range(6):
        theta = math.radians(rot2 + k * 60.0)
        x = scale2 * math.cos(theta)
        y = scale2 * math.sin(theta)
        inner.append((x, y, rot2))
    return inner, (0.0, 0.0), outer_rot

def build_rect3x4(scale_x, scale_y, rot, outer_rot):
    inner = []
    for i in range(3):
        for j in range(4):
            x = (i - 1) * scale_x
            y = (j - 1.5) * scale_y
            inner.append((x, y, rot))
    return inner, (0.0, 0.0), outer_rot

# --- Smooth objective helpers ---

def smooth_max(values, k=50.0):
    """Log-sum-exp based smooth max."""
    v = np.array(values, dtype=float)
    v_max = np.max(v)
    if v_max < -1e10:
        return v_max
    shifted = v - v_max
    result = v_max + math.log(np.sum(np.exp(k * shifted)) / k) / k
    return result

def sat_gap(A, B):
    """Compute maximum SAT gap between two polygons. Positive = separated, negative = overlap."""
    max_gap = -float('inf')
    for poly in (A, B):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            dx, dy = x2 - x1, y2 - y1
            nx, ny = -dy, dx
            norm = math.sqrt(nx*nx + ny*ny)
            if norm < 1e-9:
                continue
            nx, ny = nx/norm, ny/norm
            projA = np.array([nx*px + ny*py for px, py in A])
            projB = np.array([nx*px + ny*py for px, py in B])
            minA, maxA = np.min(projA), np.max(projA)
            minB, maxB = np.min(projB), np.max(projB)
            if maxA < minB:
                gap = minB - maxA
            elif maxB < minA:
                gap = minA - maxB
            else:
                gap = -(min(maxA, maxB) - max(minA, minB))
            if gap > max_gap:
                max_gap = gap
    return max_gap

def smooth_objective(params, topology, outer_rot, k=30.0):
    """Smooth objective function for SLSQP optimization."""
    if topology == 'ring12':
        scale, inner_rot = params
        if scale < 0.5 or scale > 3.0:
            return 1000.0
        inner, center, orot = build_ring12(scale, inner_rot, outer_rot)
    elif topology == 'hex6+6':
        scale1, scale2, rot1, rot2 = params
        if scale1 < 0.5 or scale2 < 0.5 or scale2 < scale1 + 0.3:
            return 1000.0
        inner, center, orot = build_hex6_plus6(scale1, scale2, rot1, rot2, outer_rot)
    elif topology == 'rect3x4':
        sx, sy, rot = params
        if sx < 1.5 or sy < 1.5:
            return 1000.0
        inner, center, orot = build_rect3x4(sx, sy, rot, outer_rot)
    else:
        return 1000.0
    
    # Compute smooth outer side
    cx, cy = center
    a = math.radians(orot)
    normals = []
    for i in range(6):
        theta = a + (i + 0.5) * math.pi / 3.0
        normals.append((math.cos(theta), math.sin(theta)))
    
    all_projs = []
    for (x, y, ang) in inner:
        p = hex_vertices(x, y, 1.0, ang)
        for (px, py) in p:
            dx, dy = px - cx, py - cy
            for (nx, ny) in normals:
                all_projs.append(dx * nx + dy * ny)
    
    side_smooth = (2.0 / math.sqrt(3.0)) * smooth_max(all_projs, k)
    
    # Smooth penalty for overlaps using log-sum-exp on negative gaps
    polys = [hex_vertices(x, y, 1.0, ang) for (x, y, ang) in inner]
    overlap_penalty = 0.0
    for i in range(len(polys)):
        for j in range(i + 1, len(polys)):
            gap = sat_gap(polys[i], polys[j])
            if gap < -1e-6:
                overlap_penalty += (-gap) ** 2
    
    return side_smooth + 15.0 * overlap_penalty

def objective(params, topology, outer_rot):
    """Objective function for optimization with stronger penalty for invalid configs."""
    if topology == 'ring12':
        scale, inner_rot = params
        if scale < 0.5 or scale > 3.0:
            return 1000.0
        inner, center, orot = build_ring12(scale, inner_rot, outer_rot)
    elif topology == 'hex6+6':
        scale1, scale2, rot1, rot2 = params
        if scale1 < 0.5 or scale2 < 0.5 or scale2 < scale1 + 0.3:
            return 1000.0
        inner, center, orot = build_hex6_plus6(scale1, scale2, rot1, rot2, outer_rot)
    elif topology == 'rect3x4':
        sx, sy, rot = params
        if sx < 1.5 or sy < 1.5:
            return 1000.0
        inner, center, orot = build_rect3x4(sx, sy, rot, outer_rot)
    else:
        return 1000.0
    
    side = compute_min_outer_side(inner, center, orot)
    if not check_validity(inner, center, side, orot):
        # Stronger penalty: compute max violation
        outer = hex_vertices(center[0], center[1], side, orot)
        polys = [hex_vertices(x, y, 1.0, ang) for (x, y, ang) in inner]
        max_violation = 0.0
        for p in polys:
            for (px, py) in p:
                for i in range(6):
                    x1, y1 = outer[i]
                    x2, y2 = outer[(i+1)%6]
                    cross = (x2-x1)*(py-y1) - (y2-y1)*(px-x1)
                    if cross < -1e-7:
                        max_violation = max(max_violation, -cross)
        return side + 10.0 * max_violation + 5.0
    return side

def optimize_topology(topology, outer_rot, time_budget):
    """Optimize a single topology using grid search + SLSQP + Nelder-Mead + polish."""
    start_time = time.time()
    
    if topology == 'ring12':
        x0 = [2.0, 0.0]
        bounds = [(0.8, 2.5), (0, 360)]
    elif topology == 'hex6+6':
        x0 = [1.5, 3.0, 0.0, 30.0]
        bounds = [(0.8, 2.5), (1.5, 4.0), (0, 360), (0, 360)]
    elif topology == 'rect3x4':
        x0 = [2.0, 2.0, 0.0]
        bounds = [(1.5, 3.0), (1.5, 3.0), (0, 360)]
    else:
        return None, None, None
    
    best_x = x0
    best_f = objective(x0, topology, outer_rot)
    
    # Coarse grid search
    n_grid = 7
    if topology == 'ring12':
        grid1 = np.linspace(0.8, 2.5, n_grid)
        grid2 = np.linspace(0, 360, n_grid)
        for g1 in grid1:
            for g2 in grid2:
                f = objective([g1, g2], topology, outer_rot)
                if f < best_f:
                    best_f = f
                    best_x = [g1, g2]
                if time.time() - start_time > time_budget:
                    return best_x, best_f, topology
    elif topology == 'hex6+6':
        grid1 = np.linspace(0.8, 2.5, n_grid)
        grid2 = np.linspace(1.5, 4.0, n_grid)
        grid3 = np.linspace(0, 360, n_grid)
        grid4 = np.linspace(0, 360, n_grid)
        for g1 in grid1:
            for g2 in grid2:
                if g2 < g1 + 0.3:
                    continue
                for g3 in grid3:
                    for g4 in grid4:
                        f = objective([g1, g2, g3, g4], topology, outer_rot)
                        if f < best_f:
                            best_f = f
                            best_x = [g1, g2, g3, g4]
                        if time.time() - start_time > time_budget:
                            return best_x, best_f, topology
    elif topology == 'rect3x4':
        grid1 = np.linspace(1.5, 3.0, n_grid)
        grid2 = np.linspace(1.5, 3.0, n_grid)
        grid3 = np.linspace(0, 360, n_grid)
        for g1 in grid1:
            for g2 in grid2:
                for g3 in grid3:
                    f = objective([g1, g2, g3], topology, outer_rot)
                    if f < best_f:
                        best_f = f
                        best_x = [g1, g2, g3]
                    if time.time() - start_time > time_budget:
                        return best_x, best_f, topology
    
    # SLSQP optimization with smooth objective
    if time.time() - start_time < time_budget * 0.5:
        try:
            result = minimize(smooth_objective, best_x, args=(topology, outer_rot, 50.0),
                             method='SLSQP', bounds=bounds,
                             options={'maxiter': 300, 'ftol': 1e-8})
            if result.fun < best_f:
                best_x = result.x
                best_f = objective(result.x, topology, outer_rot)
        except Exception:
            pass
    
    # Nelder-Mead refinement
    if time.time() - start_time < time_budget * 0.7:
        try:
            result = minimize(objective, best_x, args=(topology, outer_rot),
                             method='Nelder-Mead', bounds=bounds,
                             options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
            if result.fun < best_f:
                best_x = result.x
                best_f = result.fun
        except Exception:
            pass
    
    # High-precision polish pass
    if time.time() - start_time < time_budget * 0.9:
        try:
            # Use multiple starting points near best_x for polish
            for perturb in [0.01, -0.01, 0.05, -0.05]:
                if time.time() - start_time > time_budget * 0.9:
                    break
                x_perturbed = [xi + perturb * (0.5 if i % 2 == 0 else -0.5) for i, xi in enumerate(best_x)]
                # Clip to bounds
                x_perturbed = [min(max(x_perturbed[i], bounds[i][0]), bounds[i][1]) for i in range(len(x_perturbed))]
                try:
                    result = minimize(objective, x_perturbed, args=(topology, outer_rot),
                                     method='Nelder-Mead', bounds=bounds,
                                     options={'maxiter': 100, 'xatol': 1e-7, 'fatol': 1e-9})
                    if result.fun < best_f:
                        best_x = result.x
                        best_f = result.fun
                except Exception:
                    pass
        except Exception:
            pass
    
    return best_x, best_f, topology

def optimize():
    """Try all topologies and outer rotations."""
    best_side = 40.0
    best_params = None
    best_topology = None
    best_outer_rot = 0.0
    best_valid = False
    
    topologies = ['ring12', 'hex6+6', 'rect3x4']
    outer_rots = [0.0, 30.0, 60.0]
    
    # Adaptive time allocation: start with more time, reduce if finding good solutions
    time_per = 40.0
    
    for orot in outer_rots:
        for topo in topologies:
            if time.time() > DEADLINE:
                break
            # Allocate remaining time proportionally
            remaining = DEADLINE - time.time()
            if remaining < 5.0:
                break
            params, f, topo_used = optimize_topology(topo, orot, min(time_per, remaining * 0.4))
            if params is not None and f < best_side:
                if topo_used == 'ring12':
                    inner, center, orot_used = build_ring12(params[0], params[1], orot)
                elif topo_used == 'hex6+6':
                    inner, center, orot_used = build_hex6_plus6(params[0], params[1], params[2], params[3], orot)
                else:
                    inner, center, orot_used = build_rect3x4(params[0], params[1], params[2], orot)
                
                side = compute_min_outer_side(inner, center, orot_used)
                if check_validity(inner, center, side, orot_used):
                    best_side = side
                    best_params = params
                    best_topology = topo_used
                    best_outer_rot = orot
                    best_valid = True
                    # Reduce time per topology if we found a good solution
                    if best_side < 10.0:
                        time_per = 25.0
                    elif best_side < 12.0:
                        time_per = 30.0
    
    return best_side, best_params, best_topology, best_outer_rot, best_valid

def construct():
    t0 = time.time()
    try:
        side, params, topology, outer_rot, valid = optimize()
        if valid and params is not None:
            if topology == 'ring12':
                inner, center, orot = build_ring12(params[0], params[1], outer_rot)
            elif topology == 'hex6+6':
                inner, center, orot = build_hex6_plus6(params[0], params[1], params[2], params[3], outer_rot)
            else:
                inner, center, orot = build_rect3x4(params[0], params[1], params[2], outer_rot)
            return inner, center, side, orot
    except Exception:
        pass

    # Fallback: linear arrangement
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0
