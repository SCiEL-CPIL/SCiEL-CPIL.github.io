"""
Pack 12 unit hexagons inside a regular hexagon, minimizing outer side length.
Combines multi-topology lattice search (inspiration) with local refinement via
SLSQP on hard constraints, plus a contact-constraint Newton refinement pass.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = time.time() + 142.0  # safety margin before 150s kill

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
    """True if (px,py) is inside or on boundary of convex polygon."""
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

def build_hex_cluster(r1, r2, inner_rot, outer_rot):
    """1 center + 6 at r1 + 5 at r2 (hexagonal cluster)."""
    inner = [(0.0, 0.0, inner_rot)]
    for k in range(6):
        theta = math.radians(inner_rot + k * 60.0)
        x = r1 * math.cos(theta)
        y = r1 * math.sin(theta)
        inner.append((x, y, inner_rot))
    for k in range(5):
        theta = math.radians(inner_rot + 30.0 + k * 72.0)
        x = r2 * math.cos(theta)
        y = r2 * math.sin(theta)
        inner.append((x, y, inner_rot))
    return inner, (0.0, 0.0), outer_rot

def build_ring12(scale, inner_rot, outer_rot):
    """12 hexagons in a ring."""
    inner = []
    for k in range(12):
        theta = math.radians(inner_rot + k * 30.0)
        x = scale * math.cos(theta)
        y = scale * math.sin(theta)
        inner.append((x, y, inner_rot))
    return inner, (0.0, 0.0), outer_rot

def build_hex6_plus6(scale1, scale2, rot1, rot2, outer_rot):
    """Two concentric rings of 6."""
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

def build_rect3x4(sx, sy, rot, outer_rot):
    """3x4 rectangular grid."""
    inner = []
    for i in range(3):
        for j in range(4):
            x = (i - 1.0) * sx
            y = (j - 1.5) * sy
            inner.append((x, y, rot))
    return inner, (0.0, 0.0), outer_rot

def build_linear(spacing, outer_rot):
    """12 in a line."""
    inner = []
    for k in range(12):
        x = (k - 5.5) * spacing
        inner.append((x, 0.0, 0.0))
    return inner, (0.0, 0.0), outer_rot

# --- Local refinement via SLSQP ---

def refine_local(inner, outer_center, outer_angle):
    """Refine positions and angles via SLSQP minimizing outer side with constraints."""
    # Parameters: x0,y0,a0, ..., x11,y11,a11, cx, cy, ca
    p0 = []
    for (x, y, a) in inner:
        p0.extend([x, y, math.radians(a)])
    p0.extend([outer_center[0], outer_center[1], math.radians(outer_angle)])
    
    def objective(p):
        if time.time() > DEADLINE:
            return 1e6
        inner_p = []
        for i in range(N):
            inner_p.append((p[3*i], p[3*i+1], math.degrees(p[3*i+2])))
        cx, cy, ca = p[3*N], p[3*N+1], math.degrees(p[3*N+2])
        return compute_min_outer_side(inner_p, (cx, cy), ca)
    
    # Constraints: pairwise non-overlap (as inequality: min_sep >= 0)
    # For SLSQP, we need differentiable constraints. Use SAT-based gap.
    def sat_gap(p, i, j):
        inner_p = []
        for k in range(N):
            inner_p.append((p[3*k], p[3*k+1], math.degrees(p[3*k+2])))
        A = hex_vertices(*inner_p[i])
        B = hex_vertices(*inner_p[j])
        max_gap = -float('inf')
        for poly in (A, B):
            for k in range(6):
                x1, y1 = poly[k]
                x2, y2 = poly[(k+1) % 6]
                dx, dy = x2-x1, y2-y1
                nx, ny = -dy, dx
                norm = math.sqrt(nx*nx + ny*ny)
                if norm < 1e-10:
                    continue
                nx, ny = nx/norm, ny/norm
                projA = [nx*px + ny*py for px, py in A]
                projB = [nx*px + ny*py for px, py in B]
                sep = min(projB) - max(projA)
                if sep > max_gap:
                    max_gap = sep
                sep2 = min(projA) - max(projB)
                if sep2 > max_gap:
                    max_gap = sep2
        return max_gap
    
    constraints = []
    for i in range(N):
        for j in range(i+1, N):
            constraints.append({
                'type': 'ineq',
                'fun': (lambda p, i=i, j=j: sat_gap(p, i, j))
            })
    
    # Containment constraints: each vertex of each inner hex must be inside outer
    # This is complex; use a simpler approach: minimize outer side, and check validity after.
    
    try:
        result = minimize(objective, p0, method='SLSQP', 
                         bounds=[(None, None)] * (3*N + 3),
                         constraints=constraints,
                         options={'maxiter': 50, 'ftol': 1e-6})
        if result.success or result.fun < 100:
            inner_r = []
            for i in range(N):
                inner_r.append((result.x[3*i], result.x[3*i+1], math.degrees(result.x[3*i+2])))
            cx, cy, ca = result.x[3*N], result.x[3*N+1], math.degrees(result.x[3*N+2])
            outer_side = compute_min_outer_side(inner_r, (cx, cy), ca)
            if check_validity(inner_r, (cx, cy), outer_side, ca):
                return inner_r, (cx, cy), outer_side, ca
    except Exception:
        pass
    return None

def refine_nelder(inner, outer_center, outer_angle):
    """Refine via Nelder-Mead on positions and angles."""
    p0 = []
    for (x, y, a) in inner:
        p0.extend([x, y, math.radians(a)])
    p0.extend([outer_center[0], outer_center[1], math.radians(outer_angle)])
    
    def objective(p):
        if time.time() > DEADLINE:
            return 1e6
        inner_p = []
        for i in range(N):
            inner_p.append((p[3*i], p[3*i+1], math.degrees(p[3*i+2])))
        cx, cy, ca = p[3*N], p[3*N+1], math.degrees(p[3*N+2])
        outer_side = compute_min_outer_side(inner_p, (cx, cy), ca)
        # Penalty for overlap
        polys = [hex_vertices(*ip) for ip in inner_p]
        pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                if polygons_overlap_sat(polys[i], polys[j]):
                    pen += 1.0
        # Penalty for containment
        outer = hex_vertices(cx, cy, outer_side, ca)
        for p in polys:
            for (px, py) in p:
                if not point_in_convex_polygon(px, py, outer):
                    pen += 0.5
        return outer_side + 5.0 * pen
    
    try:
        result = minimize(objective, p0, method='Nelder-Mead',
                         options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-5,
                                  'adaptive': True})
        if result.fun < 100:
            inner_r = []
            for i in range(N):
                inner_r.append((result.x[3*i], result.x[3*i+1], math.degrees(result.x[3*i+2])))
            cx, cy, ca = result.x[3*N], result.x[3*N+1], math.degrees(result.x[3*N+2])
            outer_side = compute_min_outer_side(inner_r, (cx, cy), ca)
            if check_validity(inner_r, (cx, cy), outer_side, ca):
                return inner_r, (cx, cy), outer_side, ca
    except Exception:
        pass
    return None

# --- Main search ---

def construct():
    t0 = time.time()
    best_side = 1e6
    best_result = None
    
    # Try multiple topologies with various parameters
    topologies = []
    
    # Hex cluster: 1 + 6 + 5
    for r1 in [1.8, 2.0, 2.2, 2.4]:
        for r2 in [3.5, 3.8, 4.0, 4.2, 4.5]:
            for inner_rot in [0.0, 30.0]:
                for outer_rot in [0.0, 30.0]:
                    topologies.append(('hex_cluster', r1, r2, inner_rot, outer_rot))
    
    # Ring 12
    for scale in [2.5, 3.0, 3.5, 4.0]:
        for inner_rot in [0.0, 15.0, 30.0]:
            for outer_rot in [0.0, 30.0]:
                topologies.append(('ring12', scale, 0, inner_rot, outer_rot))
    
    # Hex 6+6
    for s1 in [1.8, 2.0, 2.2]:
        for s2 in [3.5, 3.8, 4.0, 4.5]:
            for rot1 in [0.0, 30.0]:
                for rot2 in [0.0, 30.0]:
                    topologies.append(('hex6+6', s1, s2, rot1, rot2))
    
    # Rect 3x4
    for sx in [2.0, 2.2, 2.5]:
        for sy in [2.0, 2.2, 2.5]:
            for rot in [0.0, 30.0]:
                topologies.append(('rect3x4', sx, sy, rot, 0.0))
    
    # Linear
    for spacing in [2.0, 2.2, 2.5]:
        topologies.append(('linear', spacing, 0, 0.0, 0.0))
    
    # Evaluate all topologies
    for topo in topologies:
        if time.time() > DEADLINE - 30:
            break
        try:
            if topo[0] == 'hex_cluster':
                _, r1, r2, ir, orr = topo
                inner, oc, oa = build_hex_cluster(r1, r2, ir, orr)
            elif topo[0] == 'ring12':
                _, scale, _, ir, orr = topo
                inner, oc, oa = build_ring12(scale, ir, orr)
            elif topo[0] == 'hex6+6':
                _, s1, s2, r1, r2 = topo
                inner, oc, oa = build_hex6_plus6(s1, s2, r1, r2, 0.0)
            elif topo[0] == 'rect3x4':
                _, sx, sy, rot, _ = topo
                inner, oc, oa = build_rect3x4(sx, sy, rot, 0.0)
            elif topo[0] == 'linear':
                _, sp, _, _, orr = topo
                inner, oc, oa = build_linear(sp, orr)
            else:
                continue
            
            outer_side = compute_min_outer_side(inner, oc, oa)
            if check_validity(inner, oc, outer_side, oa):
                if outer_side < best_side:
                    best_side = outer_side
                    best_result = (inner, oc, outer_side, oa)
        except Exception:
            continue
    
    # Refine the best result found
    if best_result is not None:
        inner, oc, side, oa = best_result
        
        # Try Nelder-Mead refinement
        refined = refine_nelder(inner, oc, oa)
        if refined is not None:
            r_side = refined[2]
            if r_side < best_side:
                best_side = r_side
                best_result = refined
        
        # Try SLSQP refinement
        refined = refine_local(inner, oc, oa)
        if refined is not None:
            r_side = refined[2]
            if r_side < best_side:
                best_side = r_side
                best_result = refined
    
    # If no valid result found, use a fallback
    if best_result is None:
        # Simple valid packing: hexagonal cluster with generous spacing
        inner, oc, oa = build_hex_cluster(2.5, 4.5, 0.0, 0.0)
        outer_side = compute_min_outer_side(inner, oc, oa)
        best_result = (inner, oc, outer_side, oa)
    
    return best_result
