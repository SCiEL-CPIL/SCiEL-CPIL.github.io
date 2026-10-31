"""Optimized hexagon packing: minimize outer hexagon side length for N unit hexagons.
Strategy: Enumerate connected 11-cell polyhexes via BFS on axial grid,
compute minimal enclosing hexagon analytically for each topology.
"""
import numpy as np
import time
import math
from collections import deque
from itertools import combinations

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon (pointy-top at angle 0)."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts

def axial_to_cartesian(q, r, scale=1.0):
    """Convert axial hex coordinates to Cartesian (pointy-top orientation)."""
    x = scale * SQRT3 * (q + r / 2.0)
    y = scale * 1.5 * r
    return (x, y)

def compute_required_side(centers, outer_center, outer_angle_deg):
    """
    Compute minimum outer side length S to contain all unit hexagons.
    centers: list of (x, y) for inner hexagon centers (all aligned, angle=0)
    outer_center: (x, y)
    outer_angle_deg: rotation of outer hexagon
    """
    if not centers:
        return 0.0
    
    ox, oy = outer_center
    # For a regular hexagon with side S and angle 0 (vertex at 0°),
    # the 6 outward normals are at 30°, 90°, 150°, 210°, 270°, 330°
    # The apothem is S * sqrt(3) / 2
    # A point (px, py) relative to center is inside iff
    #   max_i (dot(point, normal_i)) <= apothem
    # So S >= 2 * max_proj / sqrt(3)
    
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    max_projection = 0.0
    
    for (cx, cy) in centers:
        # The farthest point of a unit hexagon (side 1) from its center
        # in direction normal n is: 1 * max over vertices of dot(vertex, n)
        # For a unit hexagon centered at origin with angle 0:
        # vertices at angles 0, 60, 120, 180, 240, 300 with radius 1
        # Actually for side=1, the circumradius = 1, inradius = sqrt(3)/2
        # The support function h(n) = max over vertices of dot(v, n)
        # For a regular hexagon with circumradius R=1:
        # h(n) = R * max over 6 vertex directions of cos(angle_diff)
        # But we need the max over ALL points in the hexagon, which is the support function.
        # For a regular hexagon, h(n) = R * max_k cos(theta_n - theta_k) where theta_k are vertex angles
        # But actually for a convex polygon, h(n) = max over vertices of dot(v, n)
        
        # For unit hexagon (side 1), circumradius = 1
        # Vertices at angles 0, 60, 120, 180, 240, 300
        # h(n) for direction n at angle phi: max_k cos(phi - k*pi/3)
        # This equals max(cos(phi), cos(phi-60°), cos(phi-120°), ...)
        # = 1 - (distance to nearest multiple of 60° in angular sense)... 
        # Actually h(n) = cos(min_k |phi - k*pi/3|) but capped...
        # Let's just compute it: for a regular hexagon with circumradius R,
        # h(n) = R * max_k cos(phi - k*pi/3)
        # The max of cos over 6 equally spaced angles is cos(0) = 1 when phi aligns with a vertex
        # and cos(30°) = sqrt(3)/2 when phi aligns with an edge normal
        # So h(n) ranges from sqrt(3)/2 to 1
        
        # The farthest point of the inner hexagon from outer center in direction n:
        # = dot(center - outer_center, n_rotated) + h_inner(n_rotated)
        # where n_rotated is the normal direction in the inner hexagon's frame
        
        dx = cx - ox
        dy = cy - oy
        # Rotate to align with outer hexagon's frame
        lx = dx * cos_a + dy * sin_a
        ly = -dx * sin_a + dy * cos_a
        
        for nx, ny in normals:
            proj = lx * nx + ly * ny
            # Add the support function of the unit hexagon in direction (nx, ny)
            # For unit hexagon (side 1, circumradius 1), vertices at 0,60,...,300 degrees
            # h(n) = max over 6 vertex directions of dot(v_k, n)
            # v_k = (cos(k*pi/3), sin(k*pi/3)) for k=0..5
            # h(n) = max_k cos(angle_of_n - k*pi/3)
            angle_n = math.atan2(ny, nx)
            h_val = 0.0
            for k in range(6):
                c = math.cos(angle_n - k * math.pi / 3.0)
                if c > h_val:
                    h_val = c
            # h_val is the support function value (circumradius=1 for side=1 hexagon)
            # Wait: for a regular hexagon with side length s, circumradius = s
            # So for side=1, circumradius=1, and h(n) = max_k cos(angle_n - k*pi/3)
            # But this gives values between cos(30°)=sqrt(3)/2 and 1
            # The actual support function: h(n) = s * max_k cos(angle_n - k*pi/3)
            # For s=1: h(n) = max_k cos(angle_n - k*pi/3)
            
            total_proj = proj + h_val
            if total_proj > max_projection:
                max_projection = total_proj
    
    return 2.0 * max_projection / SQRT3

def compute_required_side_fast(centers, outer_center, outer_angle_deg):
    """
    Faster version: precompute the 6 support values for the unit hexagon.
    For a unit hexagon (side 1), the support function in 6 normal directions
    (aligned with the outer hexagon's normals) can be precomputed.
    """
    if not centers:
        return 0.0
    
    ox, oy = outer_center
    
    # The 6 outward normal directions of the outer hexagon (in its local frame)
    # For angle 0: normals at 30°, 90°, 150°, 210°, 270°, 330°
    # Support function of unit hexagon (side 1, circumradius 1) in these directions:
    # h(30°) = cos(30° - 0°) vs cos(30° - 60°) = cos(-30°) = cos(30°) = sqrt(3)/2
    # Actually let me compute: vertex angles are 0, 60, 120, 180, 240, 300
    # For normal at 30°: max cos(30°-0°), cos(30°-60°), cos(30°-120°), ...
    # = max(cos30, cos(-30), cos(-90), cos(-150), cos(-210), cos(-270))
    # = max(√3/2, √3/2, 0, -√3/2, -√3/2, 0) = √3/2
    # For normal at 90°: max cos(90), cos(30), cos(-30), cos(-90), cos(-150), cos(-210)
    # = max(0, √3/2, √3/2, 0, -√3/2, -√3/2) = √3/2
    # So for all 6 normal directions, h = √3/2 for a hexagon with the same orientation!
    # This makes sense: the inradius of a unit hexagon is √3/2.
    
    # So if inner and outer hexagons have the same orientation:
    # total_proj = dot(center - outer_center, n) + √3/2
    # S = 2 * max_i (dot(c_i - o, n_i) + √3/2) / √3
    
    # But if the outer hexagon is rotated by angle A, the normals rotate too,
    # and the support function of the inner hexagon in the rotated normal directions
    # is still √3/2 (since the inner hexagon is regular and the 6 normals are
    # still at 60° intervals, just rotated).
    
    # Wait, that's only true if the inner hexagon is also rotated by the same angle.
    # If inner hexagons are all at angle 0 and outer is at angle A,
    # then in the outer's frame, the inner hexagon's normals are at -A, -A+60, ...
    # The support function h(n) for the inner hexagon in direction n depends on
    # the angle between n and the inner hexagon's vertex directions.
    
    # Let's just do it properly but efficiently.
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    
    # Precompute normals in outer frame
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_projection = 0.0
    
    for (cx, cy) in centers:
        dx = cx - ox
        dy = cy - oy
        lx = dx * cos_a + dy * sin_a
        ly = -dx * sin_a + dy * cos_a
        
        for nx, ny in normals:
            proj = lx * nx + ly * ny
            # Support function of unit hexagon (angle 0) in direction (nx, ny)
            angle_n = math.atan2(ny, nx)
            # h = max_k cos(angle_n - k*pi/3) for k=0..5
            # This is periodic with period 60°, so:
            # angle_mod = angle_n % (pi/3)
            # h = cos(min(angle_mod, pi/3 - angle_mod))
            angle_mod = angle_n % (math.pi / 3.0)
            h_val = math.cos(min(angle_mod, math.pi / 3.0 - angle_mod))
            
            total_proj = proj + h_val
            if total_proj > max_projection:
                max_projection = total_proj
    
    return 2.0 * max_projection / SQRT3

def compute_min_side_for_topology(centers, max_angles=7):
    """
    For a given set of centers (from a polyhex topology), find the minimal
    outer hexagon side by trying different outer angles and centers.
    """
    best_side = float('inf')
    best_center = (0, 0)
    best_angle = 0.0
    
    # The optimal outer center is the "center" of the configuration
    # Try centroid and a few offsets
    cx_avg = sum(c[0] for c in centers) / len(centers)
    cy_avg = sum(c[1] for c in centers) / len(centers)
    
    # Try multiple outer angles (exploit 60° symmetry, so 0 to 60° suffices)
    for angle_deg in [i * 5.0 for i in range(13)]:  # 0, 5, 10, ..., 60
        for dx in [-0.5, -0.25, 0.0, 0.25, 0.5]:
            for dy in [-0.5, -0.25, 0.0, 0.25, 0.5]:
                oc = (cx_avg + dx, cy_avg + dy)
                s = compute_required_side_fast(centers, oc, angle_deg)
                if s < best_side:
                    best_side = s
                    best_center = oc
                    best_angle = angle_deg
    
    # Refine around best angle
    for angle_deg in [best_angle - 2.5, best_angle - 1.0, best_angle - 0.5,
                      best_angle + 0.5, best_angle + 1.0, best_angle + 2.5]:
        for dx in [-0.3, -0.1, 0.0, 0.1, 0.3]:
            for dy in [-0.3, -0.1, 0.0, 0.1, 0.3]:
                oc = (cx_avg + dx, cy_avg + dy)
                s = compute_required_side_fast(centers, oc, angle_deg % 360)
                if s < best_side:
                    best_side = s
                    best_center = oc
                    best_angle = angle_deg % 360
    
    return best_side, best_center, best_angle

def generate_polyhexes(n_cells, max_radius=4):
    """
    Generate all connected n-cell polyhexes via BFS on axial grid.
    Returns list of sets of (q, r) tuples, canonicalized by rotation/reflection.
    """
    # Axial neighbors for pointy-top hex grid
    neighbors = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    
    def canonicalize(cells):
        """Generate all 12 symmetries (6 rotations * 2 reflections) and pick min."""
        cell_set = set(cells)
        best = None
        
        for rot in range(6):
            for reflect in [False, True]:
                transformed = []
                for (q, r) in cell_set:
                    # Apply reflection if needed
                    if reflect:
                        q, r = -q - r, -r
                    # Apply rotation
                    for _ in range(rot):
                        q, r = -r, -q - r
                    transformed.append((q, r))
                
                # Normalize: translate so min q, then min r is at origin
                min_q = min(c[0] for c in transformed)
                min_r = min(c[1] for c in transformed)
                normalized = tuple(sorted((q - min_q, r - min_r) for q, r in transformed))
                
                if best is None or normalized < best:
                    best = normalized
        
        return best
    
    # BFS: start with single cell, grow by adding adjacent cells
    # Use a set to track seen canonical forms
    seen_canonical = set()
    results = []
    
    # Start with a single cell at origin
    current = [(0, 0)]
    seen_canonical.add(canonicalize(current))
    
    # BFS levels
    queue = deque()
    queue.append(current)
    
    while queue:
        cells = queue.popleft()
        
        if len(cells) == n_cells:
            canon = canonicalize(cells)
            if canon not in [c for c in seen_canonical if len(c) == n_cells]:
                results.append(cells)
            continue
        
        if len(cells) > n_cells:
            continue
        
        # Try adding each neighbor of each cell
        cell_set = set(cells)
        candidates = set()
        for (q, r) in cells:
            for (dq, dr) in neighbors:
                new_cell = (q + dq, r + dr)
                if new_cell not in cell_set:
                    candidates.add(new_cell)
        
        for new_cell in candidates:
            new_cells = cells + [new_cell]
            canon = canonicalize(new_cells)
            if canon not in seen_canonical:
                seen_canonical.add(canon)
                queue.append(new_cells)
    
    return results

def generate_polyhexes_limited(n_cells, max_cells_to_check=5000, deadline=None):
    """
    Generate polyhexes with a limit on how many we check.
    """
    neighbors = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    
    def canonicalize(cells):
        cell_set = set(cells)
        best = None
        for rot in range(6):
            for reflect in [False, True]:
                transformed = []
                for (q, r) in cell_set:
                    if reflect:
                        q, r = -q - r, -r
                    for _ in range(rot):
                        q, r = -r, -q - r
                    transformed.append((q, r))
                min_q = min(c[0] for c in transformed)
                min_r = min(c[1] for c in transformed)
                normalized = tuple(sorted((q - min_q, r - min_r) for q, r in transformed))
                if best is None or normalized < best:
                    best = normalized
        return best
    
    seen_canonical = set()
    results = []
    queue = deque()
    queue.append([(0, 0)])
    seen_canonical.add(canonicalize([(0, 0)]))
    
    while queue and len(results) < max_cells_to_check:
        if deadline and time.time() > deadline:
            break
        
        cells = queue.popleft()
        
        if len(cells) == n_cells:
            results.append(cells)
            continue
        
        if len(cells) >= n_cells:
            continue
        
        cell_set = set(cells)
        candidates = set()
        for (q, r) in cells:
            for (dq, dr) in neighbors:
                new_cell = (q + dq, r + dr)
                if new_cell not in cell_set:
                    candidates.add(new_cell)
        
        for new_cell in candidates:
            new_cells = cells + [new_cell]
            canon = canonicalize(new_cells)
            if canon not in seen_canonical:
                seen_canonical.add(canon)
                queue.append(new_cells)
    
    return results

def sat_overlap_check(v1, v2, eps=1e-9):
    """Check if two convex polygons overlap using SAT."""
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
    """Check if all inner hexagons are pairwise disjoint."""
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
    """Check validity: no overlaps, all inside."""
    if not check_all_disjoint(inner):
        return False
    if not check_inside(inner, outer_center, outer_side, outer_angle_deg):
        return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Step 1: Generate polyhex topologies for N=11
    # For 11 cells, there are 1605 free polyhexes (known result)
    # But we'll generate them and evaluate each
    
    # Use a tighter scale to allow non-lattice-optimal packings
    # Actually, for lattice-locked hexagons, the natural spacing is:
    # Adjacent hexagons (side 1) touching: center distance = sqrt(3) (for edge-to-edge)
    # In axial coords with scale=1: distance between (0,0) and (1,0) is sqrt(3)
    # This is exactly the touching distance for pointy-top hexagons of side 1.
    
    # Let's use scale = 1.0 (touching) and also try slightly larger for safety
    epsilon = 1e-7
    scale = 1.0 + epsilon
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = (0, 0)
    best_outer_angle = 0.0
    
    # Generate polyhexes
    polyhexes = generate_polyhexes_limited(N, max_cells_to_check=3000, deadline=deadline - 30)
    
    # Also include some hand-crafted configurations
    hand_configs = [
        # Compact 3x3-ish cluster
        [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,-1), (1,-1), (2,-1), (-1,0), (-1,1)],
        # Cross shape
        [(0,0), (1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,1), (1,-1), (-1,-1), (2,0), (0,2)],
        # Ring + center
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,-1), (1,1), (-2,1), (-1,-1)],
        # 4-3-4 arrangement
        [(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(0,-1),(1,-1),(2,-1),(3,-1)],
    ]
    
    # Convert hand configs to center lists
    all_configs = []
    for ph in polyhexes:
        centers = [axial_to_cartesian(q, r, scale) for q, r in ph]
        all_configs.append(centers)
    
    for hc in hand_configs:
        if len(hc) == N:
            centers = [axial_to_cartesian(q, r, scale) for q, r in hc]
            all_configs.append(centers)
    
    # Evaluate each configuration
    for centers in all_configs:
        if time.time() > deadline - 20:
            break
        
        side, oc, oa = compute_min_side_for_topology(centers)
        if side < best_side:
            best_side = side
            best_outer_center = oc
            best_outer_angle = oa
            best_inner = [[c[0], c[1], 0.0] for c in centers]
    
    # If no valid config found, use a simple fallback
    if best_inner is None:
        # Simple ring configuration
        axial = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
                 (2,-1), (1,1), (-2,1), (-1,-1)]
        centers = [axial_to_cartesian(q, r, scale) for q, r in axial]
        best_inner = [[c[0], c[1], 0.0] for c in centers]
        cx_avg = sum(c[0] for c in centers) / len(centers)
        cy_avg = sum(c[1] for c in centers) / len(centers)
        best_outer_center = (cx_avg, cy_avg)
        best_outer_angle = 0.0
        best_side = compute_required_side_fast(centers, best_outer_center, best_outer_angle)
    
    # Step 2: Try to improve by optimizing inner positions slightly
    # Use scipy to nudge positions for better packing
    if time.time() < deadline - 15 and best_inner is not None:
        try:
            from scipy.optimize import minimize
            
            def objective_inner(params):
                inner = []
                for i in range(N):
                    x = params[3*i]
                    y = params[3*i+1]
                    ang = params[3*i+2] * 180.0 / math.pi
                    inner.append([x, y, ang])
                
                side = compute_required_side_fast(
                    [(p[0], p[1]) for p in inner],
                    best_outer_center,
                    best_outer_angle
                )
                
                # Penalty for overlaps
                penalty = 0.0
                for i in range(N):
                    vi = get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2]))
                    for j in range(i + 1, N):
                        vj = get_hex_vertices(inner[j][0], inner[j][1], 1.0, math.radians(inner[j][2]))
                        if sat_overlap_check(vi, vj, 1e-6):
                            dx = inner[i][0] - inner[j][0]
                            dy = inner[i][1] - inner[j][1]
                            dist = math.sqrt(dx*dx + dy*dy)
                            penalty += max(0, 2.0 - dist) ** 2 * 50.0
                
                return side + penalty
            
            inner_params = np.zeros(3 * N)
            for i, (x, y, ang) in enumerate(best_inner):
                inner_params[3*i] = x
                inner_params[3*i+1] = y
                inner_params[3*i+2] = math.radians(ang)
            
            res = minimize(objective_inner, inner_params, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-12})
            
            if res.fun < best_side + 0.05:
                inner_new = []
                for i in range(N):
                    x = res.x[3*i]
                    y = res.x[3*i+1]
                    ang = res.x[3*i+2] * 180.0 / math.pi
                    inner_new.append([x, y, ang])
                
                side_new = compute_required_side_fast(
                    [(p[0], p[1]) for p in inner_new],
                    best_outer_center,
                    best_outer_angle
                )
                
                if check_validity(inner_new, best_outer_center, side_new, best_outer_angle):
                    if side_new < best_side:
                        best_inner = inner_new
                        best_side = side_new
        except Exception:
            pass
    
    # Step 3: Final refinement of outer hexagon
    if time.time() < deadline - 5 and best_inner is not None:
        try:
            from scipy.optimize import minimize
            
            def objective_outer(params):
                cx, cy, angle_deg = params
                side = compute_required_side_fast(
                    [(p[0], p[1]) for p in best_inner],
                    (cx, cy),
                    angle_deg
                )
                return side
            
            init = np.array([best_outer_center[0], best_outer_center[1], best_outer_angle])
            res = minimize(objective_outer, init, method='Nelder-Mead',
                          options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10})
            
            if res.fun < best_side:
                ox, oy, oa = res.x
                side_final = compute_required_side_fast(
                    [(p[0], p[1]) for p in best_inner],
                    (ox, oy),
                    oa
                )
                if check_validity(best_inner, (ox, oy), side_final, oa % 360):
                    best_outer_center = (ox, oy)
                    best_outer_angle = oa % 360
                    best_side = side_final
        except Exception:
            pass
    
    # Ensure validity
    if best_inner is None or not check_validity(best_inner, best_outer_center, best_side, best_outer_angle):
        # Fallback: use a known valid configuration
        axial = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
                 (2,-1), (1,1), (-2,1), (-1,-1)]
        scale_fb = 1.0 + 1e-6
        centers = [axial_to_cartesian(q, r, scale_fb) for q, r in axial]
        best_inner = [[c[0], c[1], 0.0] for c in centers]
        cx_avg = sum(c[0] for c in centers) / len(centers)
        cy_avg = sum(c[1] for c in centers) / len(centers)
        best_outer_center = (cx_avg, cy_avg)
        best_outer_angle = 0.0
        best_side = compute_required_side_fast(centers, best_outer_center, best_outer_angle) + 0.1
    
    return (best_inner, best_outer_center, best_side, best_outer_angle)

# EVOLVE-BLOCK-END
