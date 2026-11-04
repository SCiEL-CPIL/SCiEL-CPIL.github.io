"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side."""
import math
import time
import random
import numpy as np

N = 12
TIME_LIMIT = 115.0  # safety margin below 120s

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Vertices of a regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    verts = []
    for k in range(6):
        theta = math.radians(60 * k)
        x = side * math.cos(theta)
        y = side * math.sin(theta)
        verts.append((cx + ca * x - sa * y, cy + sa * x + ca * y))
    return verts

def _project_polygon(verts, axis):
    """Project polygon vertices onto axis, return (min, max)."""
    ax, ay = axis
    vals = [vx * ax + vy * ay for vx, vy in verts]
    return min(vals), max(vals)

def _axes_from_polygon(verts):
    """Get edge normals (perpendicular to edges) for SAT."""
    axes = []
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-12:
            continue
        nx, ny = -dy / length, dx / length
        axes.append((nx, ny))
    return axes

def _separating_axis_check(vertsA, vertsB):
    """Check if two convex polygons overlap using SAT. Return True if they overlap (not disjoint)."""
    axes = _axes_from_polygon(vertsA) + _axes_from_polygon(vertsB)
    for axis in axes:
        minA, maxA = _project_polygon(vertsA, axis)
        minB, maxB = _project_polygon(vertsB, axis)
        if maxA <= minB + 1e-9 or maxB <= minA + 1e-9:
            return False
    return True

def _sdf_distance(vertsA, vertsB):
    """
    Compute approximate minimum distance between two convex hexagons using SDF.
    Returns negative if overlapping (penetration depth), positive if separated.
    Uses edge-to-edge and vertex-to-edge distance checks.
    """
    def point_to_segment_dist(px, py, ax, ay, bx, by):
        dx, dy = bx - ax, by - ay
        len_sq = dx * dx + dy * dy
        if len_sq < 1e-20:
            return math.hypot(px - ax, py - ay)
        t = ((px - ax) * dx + (py - ay) * dy) / len_sq
        t = max(0.0, min(1.0, t))
        proj_x, proj_y = ax + t * dx, ay + t * dy
        return math.hypot(px - proj_x, py - proj_y)
    
    min_dist = float('inf')
    
    # Check all vertex-to-edge distances
    for i in range(6):
        ax, ay = vertsA[i]
        bx, by = vertsA[(i + 1) % 6]
        for j in range(6):
            px, py = vertsB[j]
            d = point_to_segment_dist(px, py, ax, ay, bx, by)
            if d < min_dist:
                min_dist = d
    
    for i in range(6):
        ax, ay = vertsB[i]
        bx, by = vertsB[(i + 1) % 6]
        for j in range(6):
            px, py = vertsA[j]
            d = point_to_segment_dist(px, py, ax, ay, bx, by)
            if d < min_dist:
                min_dist = d
    
    # If they overlap, we need to compute penetration depth
    if _separating_axis_check(vertsA, vertsB):
        # Find maximum overlap on any axis
        max_overlap = 0.0
        axes = _axes_from_polygon(vertsA) + _axes_from_polygon(vertsB)
        for axis in axes:
            minA, maxA = _project_polygon(vertsA, axis)
            minB, maxB = _project_polygon(vertsB, axis)
            overlap = min(maxA, maxB) - max(minA, minB)
            if overlap > max_overlap:
                max_overlap = overlap
        return -max_overlap
    
    return min_dist

def _outer_hex_side(centers_angles, center, angle_deg):
    """Compute minimum side length S of outer hexagon containing all inner hexagons."""
    a = math.radians(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = a + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    
    cx, cy = center
    max_need = 0.0
    for (px, py, pangle) in centers_angles:
        verts = _hex_vertices(px, py, pangle)
        for (vx, vy) in verts:
            for (nx, ny) in normals:
                val = (nx * (vx - cx) + ny * (vy - cy)) / apothem_unit
                if val > max_need:
                    max_need = val
    return max_need

def _is_valid(centers_angles):
    """Check all pairs of hexagons are disjoint."""
    m = len(centers_angles)
    polys = [_hex_vertices(p[0], p[1], p[2]) for p in centers_angles]
    for i in range(m):
        for j in range(i + 1, m):
            if _separating_axis_check(polys[i], polys[j]):
                return False
    return True

def construct():
    t0 = time.time()
    random.seed(42)
    
    def full_score(cfg, center, angle):
        """Return outer side if valid, else None."""
        if not _is_valid(cfg):
            return None
        return _outer_hex_side(cfg, center, angle)
    
    def make_hex_lattice():
        """3x4 hexagonal lattice with offset."""
        pts = []
        row_height = math.sqrt(3) * 1.0
        col_width = 2.0 * 1.0
        for r in range(3):
            for c in range(4):
                x = c * col_width + (col_width / 2 if r % 2 else 0.0)
                y = r * row_height
                pts.append([x - 3.0, y - row_height, 0.0])
        return pts
    
    def make_ring():
        """12 hexagons in a ring, oriented outward."""
        pts = []
        R = 2.5
        for i in range(N):
            theta = 2 * math.pi * i / N
            x = R * math.cos(theta)
            y = R * math.sin(theta)
            angle_deg = math.degrees(theta)
            pts.append([x, y, angle_deg])
        return pts
    
    def smooth_objective(v):
        """
        Smooth objective: outer side length + collision penalty.
        v = [x1,y1,a1, ..., x12,y12,a12, cx,cy, outer_angle]
        Uses direct max for outer side and SAT-based collision penalty.
        """
        cfg = []
        for i in range(N):
            cfg.append([v[3*i], v[3*i+1], v[3*i+2]])
        center = (v[3*N], v[3*N+1])
        outer_angle = v[3*N+2]
        
        # Compute all vertices for all hexagons
        all_verts = []
        for p in cfg:
            verts = _hex_vertices(p[0], p[1], p[2])
            all_verts.append(verts)
        
        # Outer hexagon normals
        a = math.radians(outer_angle)
        apothem_unit = math.sqrt(3) / 2.0
        normals = []
        for k in range(6):
            na = a + math.radians(90 + 60 * k)
            normals.append((math.cos(na), math.sin(na)))
        
        cx, cy = center
        
        # Compute max projection directly (exact)
        max_val = 0.0
        for verts in all_verts:
            for (vx, vy) in verts:
                for (nx, ny) in normals:
                    val = (nx * (vx - cx) + ny * (vy - cy)) / apothem_unit
                    if val > max_val:
                        max_val = val
        
        # Collision penalty: for each pair, use SDF distance for smooth gradient
        collision_penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dist = _sdf_distance(all_verts[i], all_verts[j])
                if dist < 0:  # Overlapping
                    collision_penalty += dist * dist
                else:
                    # Small repulsive force for near-collisions
                    if dist < 0.5:
                        collision_penalty += 0.01 * (0.5 - dist) ** 2
        
        return max_val + 10.0 * collision_penalty
    
    def nelder_mead_polish(cfg_init, center_init, angle_init, time_budget):
        """Nelder-Mead simplex optimization for polishing."""
        def flatten(cfg, center, angle):
            v = []
            for p in cfg:
                v.extend([p[0], p[1], p[2]])
            v.extend([center[0], center[1], angle])
            return v
        
        v0 = flatten(cfg_init, center_init, angle_init)
        n_params = len(v0)
        
        # Initialize simplex with smaller perturbations for better convergence
        simplex = [v0[:]]
        f_vals = [smooth_objective(v0)]
        
        for i in range(n_params):
            v_pert = v0[:]
            step = 0.05 if i % 3 != 2 else 0.02  # smaller steps for better polish
            v_pert[i] += step
            simplex.append(v_pert)
            f_vals.append(smooth_objective(v_pert))
        
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink
        
        best_idx = min(range(n_params + 1), key=lambda i: f_vals[i])
        best_f = f_vals[best_idx]
        best_v = simplex[best_idx][:]
        
        max_iters = 3000
        iter_count = 0
        no_improve_count = 0
        while iter_count < max_iters and time.time() - t0 < time_budget:
            # Sort simplex by objective value
            order = sorted(range(n_params + 1), key=lambda i: f_vals[i])
            simplex = [simplex[i] for i in order]
            f_vals = [f_vals[i] for i in order]
            
            # Check convergence
            if f_vals[-1] - f_vals[0] < 1e-8:
                break
            
            # Centroid of all points except worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            worst_v = simplex[-1]
            v_reflect = centroid + alpha * (centroid - np.array(worst_v))
            f_reflect = smooth_objective(v_reflect)
            
            improved = False
            if f_reflect < f_vals[0]:
                # Expansion
                v_expand = centroid + gamma * (np.array(v_reflect) - centroid)
                f_expand = smooth_objective(v_expand)
                if f_expand < f_reflect:
                    simplex[-1] = v_expand
                    f_vals[-1] = f_expand
                else:
                    simplex[-1] = v_reflect
                    f_vals[-1] = f_reflect
                improved = True
            elif f_reflect < f_vals[-2]:
                simplex[-1] = v_reflect
                f_vals[-1] = f_reflect
                improved = True
            else:
                # Contraction
                v_contract = centroid + rho * (np.array(worst_v) - centroid)
                f_contract = smooth_objective(v_contract)
                if f_contract < f_vals[-1]:
                    simplex[-1] = v_contract
                    f_vals[-1] = f_contract
                    improved = True
                else:
                    # Shrink
                    for i in range(1, n_params + 1):
                        simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n_params)]
                        f_vals[i] = smooth_objective(simplex[i])
            
            # Update best
            for i in range(n_params + 1):
                if f_vals[i] < best_f:
                    best_f = f_vals[i]
                    best_v = simplex[i][:]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            
            if no_improve_count > 200:
                break
            
            iter_count += 1
        
        return best_v, best_f
    
    # Try multiple initial configurations with diverse outer hexagon params
    best_cfg = None
    best_score = float('inf')
    best_center = [0.0, 0.0]
    best_angle = 0.0
    
    def make_grid_initial():
        """Grid-like arrangement with some rotation variety."""
        pts = []
        row_height = math.sqrt(3) * 1.0
        col_width = 2.0 * 1.0
        for r in range(3):
            for c in range(4):
                x = c * col_width + (col_width / 2 if r % 2 else 0.0)
                y = r * row_height
                ang = random.uniform(-10, 10)
                pts.append([x - 3.0, y - row_height, ang])
        return pts
    
    inits = [make_hex_lattice(), make_ring(), make_grid_initial()]
    outer_angles = [0.0, 15.0, 30.0, -15.0]
    for init in inits:
        for oa in outer_angles:
            s = full_score(init, (0.0, 0.0), oa)
            if s is not None and s < best_score:
                best_score = s
                best_cfg = [row[:] for row in init]
                best_center = [0.0, 0.0]
                best_angle = oa
    
    if best_cfg is None:
        # Fallback: spread out widely
        cfg = []
        for i in range(N):
            cfg.append([random.uniform(-3, 3), random.uniform(-3, 3), 0.0])
        best_cfg = cfg
        best_score = 50.0
        best_center = [0.0, 0.0]
        best_angle = 0.0
    
    cfg = [row[:] for row in best_cfg]
    center = best_center[:]
    angle = best_angle
    cur_score = best_score
    
    T = 0.8
    T_end = 0.003
    alpha = 0.9997
    
    # Reserve time for Nelder-Mead polish
    nm_time_budget = 30.0  # seconds reserved for polishing
    sa_time_limit = TIME_LIMIT - nm_time_budget
    
    while time.time() - t0 < sa_time_limit:
        r = random.random()
        
        if r < 0.45:
            # Move one hexagon
            idx = random.randrange(N)
            old_x, old_y = cfg[idx][0], cfg[idx][1]
            step = 0.25 * (1.0 + 0.5 * T)
            cfg[idx][0] += random.gauss(0, step)
            cfg[idx][1] += random.gauss(0, step)
        elif r < 0.65:
            # Rotate one hexagon
            idx = random.randrange(N)
            old_ang = cfg[idx][2]
            cfg[idx][2] += random.gauss(0, 5.0 + 3.0 * T)
        elif r < 0.75:
            # Move center
            old_cx, old_cy = center[0], center[1]
            step = 0.15 * (1.0 + 0.5 * T)
            center[0] += random.gauss(0, step)
            center[1] += random.gauss(0, step)
        elif r < 0.85:
            # Rotate outer
            old_angle = angle
            angle += random.gauss(0, 3.0 + 3.0 * T)
        elif r < 0.92:
            # Perturb two hexagons
            idx1, idx2 = random.sample(range(N), 2)
            old_x1, old_y1 = cfg[idx1][0], cfg[idx1][1]
            old_x2, old_y2 = cfg[idx2][0], cfg[idx2][1]
            step = 0.2 * (1.0 + 0.5 * T)
            cfg[idx1][0] += random.gauss(0, step)
            cfg[idx1][1] += random.gauss(0, step)
            cfg[idx2][0] += random.gauss(0, step)
            cfg[idx2][1] += random.gauss(0, step)
        else:
            # Perturb three hexagons
            idx1, idx2, idx3 = random.sample(range(N), 3)
            old = [(cfg[i][0], cfg[i][1], cfg[i][2]) for i in (idx1, idx2, idx3)]
            step = 0.15 * (1.0 + 0.5 * T)
            for idx in (idx1, idx2, idx3):
                cfg[idx][0] += random.gauss(0, step)
                cfg[idx][1] += random.gauss(0, step)
        
        new_score = full_score(cfg, tuple(center), angle)
        
        if new_score is not None:
            if new_score < cur_score or (T > T_end and random.random() < math.exp(-(new_score - cur_score) / T)):
                cur_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_cfg = [row[:] for row in cfg]
                    best_center = center[:]
                    best_angle = angle
            else:
                # Revert
                if r < 0.45:
                    cfg[idx][0], cfg[idx][1] = old_x, old_y
                elif r < 0.65:
                    cfg[idx][2] = old_ang
                elif r < 0.75:
                    center[0], center[1] = old_cx, old_cy
                elif r < 0.85:
                    angle = old_angle
                elif r < 0.92:
                    cfg[idx1][0], cfg[idx1][1] = old_x1, old_y1
                    cfg[idx2][0], cfg[idx2][1] = old_x2, old_y2
                else:
                    for idx, (ox, oy, oa) in zip((idx1, idx2, idx3), old):
                        cfg[idx][0], cfg[idx][1], cfg[idx][2] = ox, oy, oa
        else:
            # Invalid: revert
            if r < 0.45:
                cfg[idx][0], cfg[idx][1] = old_x, old_y
            elif r < 0.65:
                cfg[idx][2] = old_ang
            elif r < 0.75:
                center[0], center[1] = old_cx, old_cy
            elif r < 0.85:
                angle = old_angle
            elif r < 0.92:
                cfg[idx1][0], cfg[idx1][1] = old_x1, old_y1
                cfg[idx2][0], cfg[idx2][1] = old_x2, old_y2
            else:
                for idx, (ox, oy, oa) in zip((idx1, idx2, idx3), old):
                    cfg[idx][0], cfg[idx][1], cfg[idx][2] = ox, oy, oa
        
        T = max(T_end, T * alpha)
    
    # Nelder-Mead polish on the best configuration found
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 5.0:  # Only polish if we have enough time
        try:
            nm_deadline = TIME_LIMIT - 3.0
            polished_v, polished_f = nelder_mead_polish(
                best_cfg, best_center, best_angle, nm_deadline
            )
            # Unflatten the polished result
            cfg_polished = []
            for i in range(N):
                cfg_polished.append([polished_v[3*i], polished_v[3*i+1], polished_v[3*i+2]])
            center_polished = [polished_v[3*N], polished_v[3*N+1]]
            angle_polished = polished_v[3*N+2]
            
            # Verify the polished result
            polished_score = full_score(cfg_polished, tuple(center_polished), angle_polished)
            if polished_score is not None and polished_score < best_score:
                best_score = polished_score
                best_cfg = cfg_polished
                best_center = center_polished
                best_angle = angle_polished
        except Exception:
            pass  # Keep the SA result if polish fails
    
    # Final high-precision coordinate descent polish
    if time.time() - t0 < TIME_LIMIT - 5.0:
        try:
            cd_deadline = TIME_LIMIT - 2.0
            for iteration in range(100):
                if time.time() - t0 > cd_deadline:
                    break
                improved = False
                # Try small perturbations in each dimension
                for i in range(N):
                    for dim in range(3):
                        old_val = best_cfg[i][dim]
                        for delta in [-0.01, 0.01, -0.005, 0.005]:
                            best_cfg[i][dim] = old_val + delta
                            test_score = full_score(best_cfg, tuple(best_center), best_angle)
                            if test_score is not None and test_score < best_score:
                                best_score = test_score
                                improved = True
                                break
                            best_cfg[i][dim] = old_val
                # Also try center and angle adjustments
                for dim in range(2):
                    old_val = best_center[dim]
                    for delta in [-0.01, 0.01]:
                        best_center[dim] = old_val + delta
                        test_score = full_score(best_cfg, tuple(best_center), best_angle)
                        if test_score is not None and test_score < best_score:
                            best_score = test_score
                            improved = True
                            break
                        best_center[dim] = old_val
                old_angle = best_angle
                for delta in [-0.5, 0.5]:
                    best_angle = old_angle + delta
                    test_score = full_score(best_cfg, tuple(best_center), best_angle)
                    if test_score is not None and test_score < best_score:
                        best_score = test_score
                        improved = True
                        break
                    best_angle = old_angle
                if not improved:
                    break
        except Exception:
            pass
    
    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_cfg]
    return inner, (float(best_center[0]), float(best_center[1])), float(best_score), float(best_angle)

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
