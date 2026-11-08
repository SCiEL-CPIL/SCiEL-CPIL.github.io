"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 11
DEADLINE = 140.0  # safety margin before 150s kill

def _hexagon_vertices(center, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    angle = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    verts = []
    for i in range(6):
        theta = angle + np.pi * i / 3.0
        x = center[0] + side * np.cos(theta)
        y = center[1] + side * np.sin(theta)
        verts.append((x, y))
    return np.array(verts)

def _point_in_hexagon(px, py, center, side, angle_deg):
    """Check if point (px,py) is inside the hexagon."""
    # Transform point to hexagon's local frame
    angle = np.radians(angle_deg)
    dx = px - center[0]
    dy = py - center[1]
    # Rotate back
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    lx = dx * cos_a + dy * sin_a
    ly = -dx * sin_a + dy * cos_a
    # Hexagon with circumradius = side, vertices at angles 0, 60, ...
    # In local frame, hexagon is |x| <= side, |y| <= side*sqrt(3)/2 for the 'pointy' orientation
    # Actually for vertices at 0,60,...: the hexagon has width 2*side (x) and height sqrt(3)*side (y)
    # Point is inside if: |lx| <= side and |ly| <= sqrt(3)/2 * side and |lx|/side + |ly|/(sqrt(3)/2*side) <= 1? No.
    # Let's use the standard: for a regular hexagon with circumradius R, inradius r = R*sqrt(3)/2
    # The hexagon (flat-top or pointy) can be defined by 6 half-planes.
    # For vertices at angles 0, 60, 120, 180, 240, 300 (pointy-top in y), the constraints are:
    # For each edge, the point must be on the interior side.
    # Simpler: use the fact that a point is in a regular hexagon with circumradius R centered at origin,
    # with a vertex on the x-axis, iff max(|lx|, |ly|*2/sqrt(3), |lx - ly/sqrt(3)|, ...) hmm.
    # Let me just check against all 6 edges via cross product.
    verts = []
    for i in range(6):
        theta = angle + np.pi * i / 3.0
        vx = side * np.cos(theta)
        vy = side * np.sin(theta)
        verts.append((vx, vy))
    verts = np.array(verts)
    # Check if (lx, ly) is inside polygon defined by verts (winding)
    inside = True
    for i in range(6):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % 6]
        # Cross product of edge vector and vector to point
        ex = x2 - x1
        ey = y2 - y1
        px_ = lx - x1
        py_ = ly - y1
        cross = ex * py_ - ey * px_
        if cross < -1e-9:  # outside
            inside = False
            break
    return inside

def _poly_overlap(v1, v2):
    """Check if two convex polygons (arrays of 6 vertices) overlap.
    Returns True if they overlap (including touching)."""
    # Separating axis theorem
    def get_axes(verts):
        axes = []
        n = len(verts)
        for i in range(n):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % n]
            ex = x2 - x1
            ey = y2 - y1
            # Normal
            nx, ny = -ey, ex
            norm = np.sqrt(nx**2 + ny**2)
            if norm > 1e-12:
                axes.append((nx / norm, ny / norm))
        return axes
    
    axes = get_axes(v1) + get_axes(v2)
    for ax, ay in axes:
        # Project both polygons
        proj1 = np.array([vx * ax + vy * ay for vx, vy in v1])
        proj2 = np.array([vx * ax + vy * ay for vx, vy in v2])
        max1, min1 = proj1.max(), proj1.min()
        max2, min2 = proj2.max(), proj2.min()
        if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
            return False  # Separated
    return True  # Overlap

def _min_clearance_inner_outer(inner_verts, outer_center, outer_side, outer_angle):
    """Return minimum distance from inner polygon to outer hexagon boundary (positive if inside)."""
    # For each vertex of inner, check if inside outer. If any outside, return negative.
    # For clearance, compute min distance from each inner vertex to outer boundary.
    min_dist = float('inf')
    outer_verts = _hexagon_vertices(outer_center, outer_side, outer_angle)
    for vx, vy in inner_verts:
        # Check if inside
        if not _point_in_hexagon(vx, vy, outer_center, outer_side, outer_angle):
            # Compute distance to boundary
            # Find closest point on outer polygon
            min_d = float('inf')
            for i in range(6):
                x1, y1 = outer_verts[i]
                x2, y2 = outer_verts[(i + 1) % 6]
                # Project point onto line segment
                dx = x2 - x1
                dy = y2 - y1
                len2 = dx*dx + dy*dy
                if len2 < 1e-12:
                    continue
                t = ((vx - x1)*dx + (vy - y1)*dy) / len2
                t = max(0, min(1, t))
                cx = x1 + t * dx
                cy = y1 + t * dy
                d = np.sqrt((vx - cx)**2 + (vy - cy)**2)
                min_d = min(min_d, d)
            return -min_d  # Negative means outside
        else:
            # Inside, compute distance to boundary
            min_d = float('inf')
            for i in range(6):
                x1, y1 = outer_verts[i]
                x2, y2 = outer_verts[(i + 1) % 6]
                dx = x2 - x1
                dy = y2 - y1
                len2 = dx*dx + dy*dy
                if len2 < 1e-12:
                    continue
                t = ((vx - x1)*dx + (vy - y1)*dy) / len2
                t = max(0, min(1, t))
                cx = x1 + t * dx
                cy = y1 + t * dy
                d = np.sqrt((vx - cx)**2 + (vy - cy)**2)
                min_d = min(min_d, d)
            min_dist = min(min_dist, min_d)
    return min_dist

def construct():
    start_time = time.time()
    
    best_score = float('inf')
    best_result = None
    
    # Initial guess: place hexagons in a hexagonal pattern
    # For N=11, a hexagonal arrangement with 1 center + 6 + 4? Or 3-4-3-1?
    # Let's try a few initial configurations
    
    def make_initial_config(seed_mode):
        """Generate initial positions."""
        rng = np.random.RandomState(seed_mode)
        if seed_mode == 0:
            # Hexagonal packing: center + ring of 6 + partial ring
            centers = [(0.0, 0.0)]
            # First ring: 6 hexagons at distance 2*sqrt(3)/2 = sqrt(3) ~ 1.732
            r1 = np.sqrt(3)
            for i in range(6):
                angle = np.pi / 3 * i
                centers.append((r1 * np.cos(angle), r1 * np.sin(angle)))
            # Second ring partial: 4 more
            r2 = 2 * np.sqrt(3)
            for i in range(4):
                angle = np.pi / 3 * i + np.pi / 6
                centers.append((r2 * np.cos(angle), r2 * np.sin(angle)))
            return centers[:N]
        elif seed_mode == 1:
            # Linear arrangement
            centers = [(2 * np.sqrt(3) * k, 0.0) for k in range(N)]
            return centers
        elif seed_mode == 2:
            # Grid-like
            centers = []
            for i in range(4):
                for j in range(3):
                    if len(centers) < N:
                        x = 2 * np.sqrt(3) * i + (np.sqrt(3) / 2 if j % 2 else 0)
                        y = 1.5 * j
                        centers.append((x, y))
            return centers[:N]
        else:
            # Random
            centers = [(rng.uniform(-5, 5), rng.uniform(-5, 5)) for _ in range(N)]
            return centers
    
    def evaluate(centers, angles, outer_center, outer_side, outer_angle):
        """Compute the score (outer side length) and validity."""
        # Check all inner hexagons are inside outer
        for i in range(N):
            cx, cy = centers[i]
            inner_verts = _hexagon_vertices((cx, cy), 1.0, angles[i])
            clearance = _min_clearance_inner_outer(inner_verts, outer_center, outer_side, outer_angle)
            if clearance < -1e-6:
                return float('inf'), False
        # Check no overlap between inner hexagons
        for i in range(N):
            for j in range(i + 1, N):
                v1 = _hexagon_vertices(centers[i], 1.0, angles[i])
                v2 = _hexagon_vertices(centers[j], 1.0, angles[j])
                if _poly_overlap(v1, v2):
                    return float('inf'), False
        return outer_side, True
    
    def optimize(centers0, angles0, outer_center0, outer_side0, outer_angle0):
        """Gradient descent optimization."""
        centers = np.array(centers0, dtype=float)
        angles = np.array(angles0, dtype=float)
        outer_center = np.array(outer_center0, dtype=float)
        outer_side = float(outer_side0)
        outer_angle = float(outer_angle0)
        
        best_local = float('inf')
        best_local_state = (centers.copy(), angles.copy(), outer_center.copy(), outer_side, outer_angle)
        
        lr = 0.1
        for step in range(2000):
            if time.time() > DEADLINE:
                break
            
            # Compute gradient numerically
            eps = 1e-4
            grad_cx = np.zeros(N)
            grad_cy = np.zeros(N)
            grad_a = np.zeros(N)
            grad_ox = 0.0
            grad_oy = 0.0
            grad_s = 0.0
            
            # Objective: maximize minimum clearance (i.e., minimize -min_clearance)
            # Plus penalty for overlap
            # We'll use a simple penalty method
            
            def objective(c, a, oc, s, oa):
                min_clear = float('inf')
                overlap_pen = 0.0
                for i in range(N):
                    cx, cy = c[i]
                    inner_verts = _hexagon_vertices((cx, cy), 1.0, a[i])
                    clearance = _min_clearance_inner_outer(inner_verts, oc, s, oa)
                    min_clear = min(min_clear, clearance)
                    for j in range(i + 1, N):
                        v2 = _hexagon_vertices((c[j][0], c[j][1]), 1.0, a[j])
                        if _poly_overlap(inner_verts, v2):
                            overlap_pen += 1.0
                return -min_clear + 100.0 * overlap_pen
            
            base_obj = objective(centers, angles, outer_center, outer_side, outer_angle)
            
            # Numerical gradients (too expensive, use finite diff on a subset)
            # For simplicity, use a stochastic approach: try random perturbations
            
            # Actually, let's just do a simple local search with random perturbations
            improved = False
            for _ in range(50):
                if time.time() > DEADLINE:
                    break
                new_centers = centers + rng_local.normal(0, lr, (N, 2))
                new_angles = angles + rng_local.normal(0, lr * 10, N)
                new_oc = outer_center + rng_local.normal(0, lr, 2)
                new_s = max(2.0, outer_side + rng_local.normal(0, lr))
                new_oa = outer_angle + rng_local.normal(0, lr * 10)
                
                score, valid = evaluate(new_centers, new_angles, new_oc, new_s, new_oa)
                if valid and score < best_local:
                    best_local = score
                    best_local_state = (new_centers.copy(), new_angles.copy(), new_oc.copy(), new_s, new_oa)
                    improved = True
                    centers, angles, outer_center, outer_side, outer_angle = new_centers, new_angles, new_oc, new_s, new_oa
                    lr *= 0.95
                elif valid:
                    lr *= 0.98
            
            if not improved:
                lr *= 0.9
                if lr < 1e-5:
                    break
        
        return best_local, best_local_state
    
    rng_global = np.random.RandomState(42)
    rng_local = np.random.RandomState(123)
    
    for seed_mode in range(5):
        if time.time() > DEADLINE:
            break
        
        centers0 = make_initial_config(seed_mode)
        angles0 = [0.0] * N
        outer_center0 = (0.0, 0.0)
        outer_side0 = 10.0
        outer_angle0 = 0.0
        
        score, state = optimize(centers0, angles0, outer_center0, outer_side0, outer_angle0)
        if score < best_score:
            best_score = score
            best_result = state
    
    if best_result is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    centers, angles, outer_center, outer_side, outer_angle = best_result
    inner = [[float(centers[i][0]), float(centers[i][1]), float(angles[i])] for i in range(N)]
    return inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), float(outer_angle)
