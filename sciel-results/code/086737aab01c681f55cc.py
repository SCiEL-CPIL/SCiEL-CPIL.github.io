"""Improved hexagon packing using minimum center distance 1.75 for same-orientation hexagons."""
import numpy as np
import math
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s limit
    
    sqrt3 = math.sqrt(3)
    MIN_DIST_SAME = 1.75  # Proven safe minimum for same-orientation unit hexagons
    
    def hexagon_vertices(cx, cy, side, angle_deg):
        """Return 6 vertices of regular hexagon with given center, side, rotation."""
        angle_rad = math.radians(angle_deg)
        verts = []
        for k in range(6):
            theta = angle_rad + k * math.pi / 3
            verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
        return verts
    
    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        """Check if point (px,py) is inside hexagon centered at (cx,cy) with side, angle."""
        dx = px - cx
        dy = py - cy
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)
        rx = dx * cos_a - dy * sin_a
        ry = dx * sin_a + dy * cos_a
        
        apothem = side * sqrt3 / 2
        for k in range(6):
            theta = math.radians(30 + k * 60)
            nx = math.cos(theta)
            ny = math.sin(theta)
            if rx * nx + ry * ny > apothem + 1e-9:
                return False
        return True
    
    def hexagons_disjoint(i1, i2, threshold=1e-6):
        """Check if two unit hexagons (center, angle) are disjoint (no overlap)."""
        x1, y1, a1 = i1
        x2, y2, a2 = i2
        
        dist = math.sqrt((x1-x2)**2 + (y1-y2)**2)
        # Quick reject: if centers are far enough, definitely disjoint
        if dist > 2.0 + threshold:
            return True
        # Quick accept for same orientation: need dist >= MIN_DIST_SAME
        if abs(a1 - a2) < 1e-6:
            return dist >= MIN_DIST_SAME - threshold
        
        # Check if any vertex of hex1 is inside hex2 or vice versa
        v1 = hexagon_vertices(x1, y1, 1.0, a1)
        v2 = hexagon_vertices(x2, y2, 1.0, a2)
        
        for vx, vy in v1:
            if point_in_hexagon(vx, vy, x2, y2, 1.0, a2):
                return False
        for vx, vy in v2:
            if point_in_hexagon(vx, vy, x1, y1, 1.0, a1):
                return False
        
        # Check edge intersections
        for i in range(6):
            ax, ay = v1[i]
            bx, by = v1[(i+1) % 6]
            for j in range(6):
                cx, cy = v2[j]
                dx, dy = v2[(j+1) % 6]
                if segments_intersect(ax, ay, bx, by, cx, cy, dx, dy):
                    return False
        return True
    
    def segments_intersect(p1, p2, p3, p4):
        """Check if segment p1-p2 intersects segment p3-p4."""
        def cross(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        
        a, b, c, d = (p1, p2, p3, p4)
        d1 = cross(c, d, a)
        d2 = cross(c, d, b)
        d3 = cross(a, b, c)
        d4 = cross(a, b, d)
        
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False
    
    def all_disjoint(inner_list, threshold=1e-6):
        """Check all pairs of inner hexagons are disjoint."""
        n = len(inner_list)
        for i in range(n):
            for j in range(i+1, n):
                if not hexagons_disjoint(inner_list[i], inner_list[j], threshold):
                    return False
        return True
    
    def min_outer_side(inner_list, outer_center, outer_angle):
        """Binary search for minimum outer side that contains all inner hexagons."""
        cx, cy = outer_center
        lo = 1.0
        hi = 20.0
        
        def fits(side):
            for ix, iy, iang in inner_list:
                verts = hexagon_vertices(ix, iy, 1.0, iang)
                for vx, vy in verts:
                    if not point_in_hexagon(vx, vy, cx, cy, side, outer_angle):
                        return False
            return True
        
        if not fits(hi):
            return None
        if fits(lo):
            return lo
        
        for _ in range(35):
            mid = (lo + hi) / 2
            if fits(mid):
                hi = mid
            else:
                lo = mid
        return hi + 1e-10
    
    def evaluate_packing(inner_list, outer_center, outer_angle):
        """Evaluate a packing configuration. Returns (side, valid) or (None, False)."""
        if not all_disjoint(inner_list):
            return None, False
        side = min_outer_side(inner_list, outer_center, outer_angle)
        if side is None:
            return None, False
        return side, True
    
    # --- Strategy 1: Compact hexagonal cluster with optimized spacing ---
    def generate_compact_cluster(n_hex):
        """Generate a compact cluster of n_hexagons using hexagonal lattice."""
        # Use axial coordinates with proper spacing for same-orientation hexagons
        # For same-orientation unit hexagons, minimum center distance is ~1.75
        # Hex lattice spacing: horizontal = 2, vertical offset = sqrt(3)
        # But we can use tighter packing with MIN_DIST_SAME
        
        positions = [(0, 0)]
        # Add neighbors in rings
        ring = 1
        while len(positions) < n_hex:
            # Generate ring positions
            q, r = ring, 0
            for side_idx in range(6):
                for step in range(ring):
                    positions.append((q, r))
                    dq = [1, 0, -1, -1, 0, 1][side_idx]
                    dr = [0, -1, -1, 0, 1, 1][side_idx]
                    q += dq
                    r += dr
            ring += 1
        
        return positions[:n_hex]
    
    def axial_to_cartesian(q, r, scale=1.0):
        """Convert axial coordinates to Cartesian with given scale factor."""
        # Standard hex lattice: x = 2*q + r, y = sqrt(3)*r
        # Scale by factor to adjust spacing
        x = scale * (2.0 * q + r)
        y = scale * sqrt3 * r
        return x, y
    
    # Generate initial cluster
    axial_positions = generate_compact_cluster(N)
    
    best_side = float('inf')
    best_result = None
    
    # Try different scale factors and outer angles
    for scale in [1.0, 0.9, 0.85, 0.8, 0.75]:
        if time.time() > deadline - 20:
            break
        
        for outer_angle in [0.0, 30.0, 15.0, 45.0]:
            if time.time() > deadline - 15:
                break
            
            inner = []
            for q, r in axial_positions:
                x, y = axial_to_cartesian(q, r, scale)
                inner.append([x, y, 0.0])
            
            # Center the cluster
            cx_mean = sum(p[0] for p in inner) / len(inner)
            cy_mean = sum(p[1] for p in inner) / len(inner)
            centered_inner = [[p[0] - cx_mean, p[1] - cy_mean, p[2]] for p in inner]
            
            side, valid = evaluate_packing(centered_inner, (0.0, 0.0), outer_angle)
            if valid and side < best_side:
                best_side = side
                best_result = (centered_inner, (0.0, 0.0), side, outer_angle)
    
    # --- Strategy 2: Force-directed optimization with correct minimum distance ---
    def force_directed_optimize(initial_positions, max_iters=1000):
        """Optimize positions using force-directed method with MIN_DIST_SAME."""
        n = len(initial_positions)
        pos = np.array([[p[0], p[1]] for p in initial_positions], dtype=float)
        angles = [p[2] for p in initial_positions]
        
        lr = 0.05
        decay = 0.998
        
        for it in range(max_iters):
            if time.time() > deadline - 25:
                break
            forces = np.zeros_like(pos)
            
            # Repulsion between inner hexagons using MIN_DIST_SAME
            for i in range(n):
                for j in range(i+1, n):
                    diff = pos[i] - pos[j]
                    dist = np.linalg.norm(diff)
                    if dist < MIN_DIST_SAME - 0.3:
                        if dist < 1e-6:
                            diff = np.random.randn(2) * 0.01
                            dist = np.linalg.norm(diff)
                        force = (MIN_DIST_SAME - dist) * 1.5
                        forces[i] += force * diff / dist
                        forces[j] -= force * diff / dist
            
            # Attraction to center (to keep cluster compact)
            for i in range(n):
                to_center = -pos[i]
                dist_to_center = np.linalg.norm(to_center)
                if dist_to_center > 1e-6:
                    forces[i] += 0.05 * to_center / dist_to_center
            
            # Update positions
            pos += lr * forces
            lr *= decay
            
            if np.max(np.abs(forces)) < 1e-4:
                break
        
        inner = []
        for i in range(n):
            inner.append([pos[i][0], pos[i][1], angles[i]])
        
        # Center the cluster
        cx_mean = np.mean([p[0] for p in inner])
        cy_mean = np.mean([p[1] for p in inner])
        centered_inner = [[p[0] - cx_mean, p[1] - cy_mean, p[2]] for p in inner]
        
        return centered_inner
    
    if time.time() < deadline - 40:
        # Start from best found or default lattice
        if best_result:
            base_inner = [list(p) for p in best_result[0]]
        else:
            base_inner = []
            for q, r in axial_positions:
                x, y = axial_to_cartesian(q, r, 1.0)
                base_inner.append([x, y, 0.0])
            cx_mean = sum(p[0] for p in base_inner) / len(base_inner)
            cy_mean = sum(p[1] for p in base_inner) / len(base_inner)
            base_inner = [[p[0] - cx_mean, p[1] - cy_mean, p[2]] for p in base_inner]
        
        optimized = force_directed_optimize(base_inner, max_iters=800)
        for outer_angle in [0.0, 30.0]:
            side, valid = evaluate_packing(optimized, (0.0, 0.0), outer_angle)
            if valid and side < best_side:
                best_side = side
                best_result = (optimized, (0.0, 0.0), side, outer_angle)
    
    # --- Strategy 3: Local search with random perturbations ---
    if time.time() < deadline - 30:
        if best_result:
            base_inner = [list(p) for p in best_result[0]]
            base_angle = best_result[3]
        else:
            base_inner = []
            for q, r in axial_positions:
                x, y = axial_to_cartesian(q, r, 1.0)
                base_inner.append([x, y, 0.0])
            cx_mean = sum(p[0] for p in base_inner) / len(base_inner)
            cy_mean = sum(p[1] for p in base_inner) / len(base_inner)
            base_inner = [[p[0] - cx_mean, p[1] - cy_mean, p[2]] for p in base_inner]
            base_angle = 0.0
        
        rng = np.random.default_rng(42)
        for trial in range(300):
            if time.time() > deadline - 15:
                break
            perturbed = []
            for p in base_inner:
                dx = rng.normal(0, 0.08)
                dy = rng.normal(0, 0.08)
                perturbed.append([p[0] + dx, p[1] + dy, p[2]])
            
            cx_mean = sum(p[0] for p in perturbed) / len(perturbed)
            cy_mean = sum(p[1] for p in perturbed) / len(perturbed)
            centered = [[p[0] - cx_mean, p[1] - cy_mean, p[2]] for p in perturbed]
            
            side, valid = evaluate_packing(centered, (0.0, 0.0), base_angle)
            if valid and side < best_side:
                best_side = side
                best_result = (centered, (0.0, 0.0), side, base_angle)
    
    # Fallback
    if best_result is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_result
# EVOLVE-BLOCK-END
