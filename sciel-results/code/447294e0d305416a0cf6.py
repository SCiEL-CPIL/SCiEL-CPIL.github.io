import numpy as np
from scipy.optimize import minimize
import time
import math

N = 11
SQRT3 = math.sqrt(3)
DEADLINE = 140.0  # Safety margin before 150s kill

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts

def hexagons_overlap(v1, v2, eps=1e-6):
    """
    Check if two hexagons (defined by their 6 vertices) overlap using SAT.
    v1, v2: lists of (x, y) tuples.
    Returns True if they overlap.
    """
    # To speed up and ensure robustness, we can check the 6 normals of v1 and 6 of v2.
    # For regular hexagons, the normals are 30, 90, 150, 210, 270, 330 degrees relative to the hexagon's orientation.
    # However, since rotations vary, we compute normals from edges.
    
    axes = []
    
    # Get axes from edges of v1
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i + 1) % 6]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    # Get axes from edges of v2
    for i in range(6):
        p1 = v2[i]
        p2 = v2[(i + 1) % 6]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))

    for axis in axes:
        min1 = float('inf')
        max1 = float('-inf')
        min2 = float('inf')
        max2 = float('-inf')
        
        for p in v1:
            proj = p[0] * axis[0] + p[1] * axis[1]
            if proj < min1: min1 = proj
            if proj > max1: max1 = proj
            
        for p in v2:
            proj = p[0] * axis[0] + p[1] * axis[1]
            if proj < min2: min2 = proj
            if proj > max2: max2 = proj
            
        # Check for gap. If there is a gap on any axis, they don't overlap.
        if max1 < min2 - eps or max2 < min1 - eps:
            return False
    return True

def is_layout_valid(inner, eps=1e-5):
    """Check if all inner hexagons are disjoint, allowing for a small epsilon gap."""
    verts_list = []
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        verts_list.append(verts)
    
    n = len(verts_list)
    for i in range(n):
        for j in range(i + 1, n):
            if hexagons_overlap(verts_list[i], verts_list[j], eps):
                return False
    return True

def compute_required_side(inner, outer_center, outer_angle):
    """Compute the minimum outer side length to contain all inner hexagons."""
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    
    # Normals for a hexagon with angle 0
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_projection = 0.0
    
    # Rotate outer frame to match outer_angle
    cos_a = math.cos(math.radians(outer_angle))
    sin_a = math.sin(math.radians(outer_angle))
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            # Translate to outer center
            dx = v[0] - ox
            dy = v[1] - oy
            # Rotate to align with outer hexagon axes
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj
    
    # S * sqrt(3) / 2 >= max_projection
    # S >= 2 * max_projection / sqrt(3)
    return 2.0 * max_projection / SQRT3

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    # 1. Define 11 hexagons on a hexagonal lattice
    # Axial coordinates (q, r)
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-1, 2), (-2, 1)
    ]
    
    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]
    
    # 2. Convert axial to Cartesian
    # Use a scale factor > 1 to ensure initial non-overlap with a safe margin.
    # Standard lattice spacing for side 1 is sqrt(3) for y and 1.5 for x.
    # Scaling by 1.01 adds a small gap.
    scale = 1.01
    
    centers = []
    for q, r in axial_coords:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        centers.append((x, y))
        
    initial_inner = [[x, y, 0.0] for x, y in centers]
    
    # Verify validity. If invalid (should not happen with scale 1.01), fall back to a safe linear arrangement.
    if not is_layout_valid(initial_inner):
        initial_inner = [[i * 3.0, 0.0, 0.0] for i in range(N)]
        centers = [(i * 3.0, 0.0) for i in range(N)]

    best_side = float('inf')
    best_inner = [list(x) for x in initial_inner]
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0

    # 3. Optimize Outer Hexagon
    cx_avg = sum(c[0] for c in centers) / len(centers)
    cy_avg = sum(c[1] for c in centers) / len(centers)
    
    # Try a range of angles and optimize center for each
    for outer_angle_deg in [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]:
        if time.time() > deadline - 30:
            break
            
        # Optimize center for this angle
        def obj_center(params):
            return compute_required_side(best_inner, (params[0], params[1]), outer_angle_deg)
            
        try:
            res = minimize(
                obj_center,
                [cx_avg, cy_avg],
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if res.fun < best_side:
                best_side = res.fun
                best_outer_center = (res.x[0], res.x[1])
                best_outer_angle = outer_angle_deg
        except Exception:
            pass

    # 4. Local Search with Strict Validity Check
    # We will perform a simple local search. At each step, we try to move one hexagon slightly.
    # If the move results in an overlap, it is rejected. If it reduces the outer side length, it is accepted.
    # This is slower than gradient-based methods but much more robust for hard constraints.
    
    if time.time() < deadline - 10:
        current_inner = [list(x) for x in best_inner]
        current_cx, current_cy = best_outer_center
        current_angle = best_outer_angle
        current_side = best_side
        
        step_size = 0.1
        iterations = 0
        max_iterations = 5000
        
        while time.time() < deadline - 5 and iterations < max_iterations:
            iterations += 1
            improved = False
            
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                    
                # Try moving hexagon i in random directions
                for _ in range(5):
                    dx = (np.random.rand() - 0.5) * step_size
                    dy = (np.random.rand() - 0.5) * step_size
                    da = (np.random.rand() - 0.5) * 0.1 # Small rotation in radians, convert to deg later if needed, but inner uses deg
                    
                    new_inner = [list(x) for x in current_inner]
                    new_inner[i][0] += dx
                    new_inner[i][1] += dy
                    new_inner[i][2] += da * 180.0 / math.pi
                    
                    if is_layout_valid(new_inner):
                        # Compute new side length
                        # Optimize center quickly for this configuration? 
                        # For speed, we can just re-center on the centroid or use the current center.
                        # Re-centering on centroid is a good heuristic.
                        cx_new = sum(h[0] for h in new_inner) / N
                        cy_new = sum(h[1] for h in new_inner) / N
                        
                        side_new = compute_required_side(new_inner, (cx_new, cy_new), current_angle)
                        
                        if side_new < current_side - 1e-7:
                            current_inner = new_inner
                            current_cx, current_cy = cx_new, cy_new
                            current_side = side_new
                            best_inner = [list(x) for x in current_inner]
                            best_outer_center = (current_cx, current_cy)
                            best_side = current_side
                            improved = True
                            break # Restart loop for this hexagon if improved
                            
            if not improved and iterations % 100 == 0:
                step_size *= 0.95 # Decrease step size if no improvement

    # Final validation
    if not is_layout_valid(best_inner):
        # Fallback to initial lattice
        best_inner = [[centers[i][0], centers[i][1], 0.0] for i in range(N)]
        best_side = compute_required_side(best_inner, (0.0, 0.0), 0.0)
        best_outer_center = (0.0, 0.0)
        best_outer_angle = 0.0

    final_side = best_side + 1e-9
    return best_inner, best_outer_center, final_side, best_outer_angle
