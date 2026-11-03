import math
import time
import numpy as np

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 3.0

def hex_vertices(cx, cy, angle_deg=0.0):
    a = math.radians(angle_deg)
    pts = []
    for k in range(6):
        theta = a + k * math.pi / 3.0
        pts.append((cx + math.cos(theta), cy + math.sin(theta)))
    return pts

def min_enclosing_hexagon_side_fast(pts, angle_deg=0.0):
    """
    Fast calculation of the minimum side length of a regular hexagon
    with given rotation that encloses all points.
    Returns (S, center_x, center_y).
    """
    a = math.radians(angle_deg)
    normals = []
    for k in range(6):
        # Edge normals are at angles a + pi/6 + k*pi/3
        theta = a + math.pi / 6.0 + k * math.pi / 3.0
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_width = 0.0
    max_width_idx = 0
    
    # Calculate widths in all 6 normal directions
    for k, (nx, ny) in enumerate(normals):
        projections = []
        for x, y in pts:
            proj = x * nx + y * ny
            projections.append(proj)
        w = max(projections) - min(projections)
        if w > max_width:
            max_width = w
            max_width_idx = k
            
    A = max_width / 2.0
    S = 2.0 * A / math.sqrt(3.0)
    
    # To find the center, we can take the average of the "midpoints" of the projections
    # or simply the centroid of the points, which is a good approximation.
    # More accurately, the center c should satisfy c.n_k approx midpoint_k
    # Let's just return the centroid for now, or solve for c if needed.
    # For the score, only S matters. But we need to return a center.
    # The optimal center minimizes the max distance. 
    # A good heuristic center is the centroid.
    cx = np.mean([p[0] for p in pts])
    cy = np.mean([p[1] for p in pts])
    
    return S, cx, cy

def get_optimal_angle_and_side(pts):
    """
    Find the angle that minimizes the enclosing hexagon side length.
    Returns (best_angle, best_S, best_center).
    """
    best_S = float('inf')
    best_angle = 0.0
    best_center = (0.0, 0.0)
    
    # The function S(theta) is periodic with period 60 degrees.
    # It is piecewise linear (max of linear functions).
    # The minimum occurs at a point where two widths are equal, or at a vertex.
    # We can sample finely.
    for angle_deg in np.linspace(0, 60, 121):
        S, cx, cy = min_enclosing_hexagon_side_fast(pts, angle_deg)
        if S < best_S:
            best_S = S
            best_angle = angle_deg
            best_center = (cx, cy)
            
    return best_angle, best_S, best_center

def force_directed_relaxation(initial_positions, max_iter=3000, dt=0.05, repulsion_strength=5.0, attraction_strength=0.1):
    """
    Relax positions using force-directed method.
    Forces:
    1. Repulsion between overlapping hexagons.
    2. Attraction to the centroid to keep the cluster compact.
    """
    n = len(initial_positions)
    pos = np.array(initial_positions, dtype=float)
    
    for it in range(max_iter):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        forces = np.zeros_like(pos)
        centroid = np.mean(pos, axis=0)
        
        # Repulsive forces
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < 1e-10:
                    dx = (np.random.random() - 0.5) * 0.01
                    dy = (np.random.random() - 0.5) * 0.01
                    dist = math.sqrt(dx*dx + dy*dy)
                
                # Approximate minimum distance for non-overlap of two unit hexagons
                # Same orientation: 2 * apothem = sqrt(3) ~ 1.732
                # Different orientations can be closer, but let's use a conservative threshold
                # or a smooth potential.
                # Let's use a potential that is 0 for d > 2.0 and grows for d < 2.0
                # The "hard core" is around 1.732.
                if dist < 2.0:
                    # Strong repulsion if very close
                    force_mag = repulsion_strength * (2.0 - dist) ** 2 / (dist + 1e-10)
                    fx = force_mag * dx / dist
                    fy = force_mag * dy / dist
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        # Attractive force to centroid
        for i in range(n):
            dx = centroid[0] - pos[i, 0]
            dy = centroid[1] - pos[i, 1]
            forces[i, 0] += attraction_strength * dx
            forces[i, 1] += attraction_strength * dy
        
        # Update positions
        pos += dt * forces
        # Decay step size
        dt *= 0.999
        
    return pos

def greedy_densest_placement():
    """
    Initialize positions on a hex grid.
    """
    positions = []
    # 11 hexagons.
    # A cluster of 19 hexagons (radius 2) is too big.
    # Radius 1 (7 hexagons) + 4 more.
    # Let's place them in a hex lattice.
    # Center (0,0)
    # Ring 1: 6 hexagons
    # Ring 2: 12 hexagons. We need 4 from Ring 2.
    
    # Axial coordinates (q, r)
    # (0,0)
    # Ring 1: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2: (2,0), (1,1), (0,2), (-1,2), (-2,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1), (2,-2)?? No, axial is q+r+s=0.
    # Let's just generate a grid and pick the 11 closest to origin.
    
    candidates = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            dist = math.sqrt(x*x + y*y)
            candidates.append((dist, q, r, x, y))
            
    candidates.sort(key=lambda t: t[0])
    
    # Pick top 11
    for i in range(11):
        d, q, r, x, y = candidates[i]
        positions.append((x, y))
        
    return positions

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    best_S = float('inf')
    best_positions = None
    best_angle = 0.0
    best_center = (0.0, 0.0)
    
    # Strategy 1: Hex grid initialization + Relaxation
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            pos1 = greedy_densest_placement()
            pos1_arr = np.array(pos1)
            
            # Relax
            pos1_relaxed = force_directed_relaxation(pos1_arr, max_iter=2000)
            
            # Optimize angle
            angle, S, center = get_optimal_angle_and_side(pos1_relaxed)
            if S < best_S:
                best_S = S
                best_positions = pos1_relaxed
                best_angle = angle
                best_center = center
    except Exception:
        pass
    
    # Strategy 2: Different initialization (e.g. slightly perturbed)
    try:
        if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
            pos2 = greedy_densest_placement()
            # Add noise
            np.random.seed(42)
            pos2_arr = np.array(pos2) + np.random.normal(0, 0.1, size=(11, 2))
            
            pos2_relaxed = force_directed_relaxation(pos2_arr, max_iter=2000)
            
            angle, S, center = get_optimal_angle_and_side(pos2_relaxed)
            if S < best_S:
                best_S = S
                best_positions = pos2_relaxed
                best_angle = angle
                best_center = center
    except Exception:
        pass
        
    # Final optimization: Local search on best
    if best_positions is not None:
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                pos_opt = best_positions.copy()
                step_size = 0.1
                current_S = best_S
                current_angle = best_angle
                current_center = best_center
                
                for it in range(200):
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    
                    improved = False
                    for i in range(11):
                        for dx in [-step_size, 0, step_size]:
                            for dy in [-step_size, 0, step_size]:
                                if dx == 0 and dy == 0:
                                    continue
                                new_pos = pos_opt.copy()
                                new_pos[i, 0] += dx
                                new_pos[i, 1] += dy
                                
                                angle, S, center = get_optimal_angle_and_side(new_pos)
                                if S < current_S - 1e-5:
                                    pos_opt = new_pos
                                    current_S = S
                                    current_angle = angle
                                    current_center = center
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    
                    if not improved:
                        step_size *= 0.8
                        if step_size < 0.01:
                            break
                
                if current_S < best_S:
                    best_S = current_S
                    best_positions = pos_opt
                    best_angle = current_angle
                    best_center = current_center
        except Exception:
            pass
    
    # Fallback
    if best_positions is None or best_S == float('inf'):
        # Fallback to simple grid
        positions = greedy_densest_placement()
        best_positions = np.array(positions)
        best_angle, best_S, best_center = get_optimal_angle_and_side(best_positions)
    
    # Scale down slightly to ensure no overlap?
    # The fast calculation assumes points are centers.
    # The hexagons have side 1.
    # If the enclosing hexagon side is S, and we place centers such that the *points* are enclosed,
    # we need to ensure the *hexagons* are enclosed.
    # The distance from center to vertex of inner hexagon is 1.
    # The distance from center to edge of outer hexagon is A_outer = S * sqrt(3) / 2.
    # We need A_outer >= max distance from outer center to any point on inner hexagons.
    # The fast calc uses the centers of the inner hexagons.
    # So the "width" calculated is the width of the centers.
    # The actual width of the union of hexagons is width_centers + 2 * max_radius_inner.
    # max_radius_inner = 1 (distance to vertex).
    # So we should have added 2 * 1 to the width?
    # Wait.
    # If we have a set of points P (centers), and we want to enclose hexagons of radius R=1.
    # The Minkowski sum of P and a unit hexagon (radius 1) is the set of all points covered.
    # The width of this set in direction n is width(P, n) + width(unit_hex, n).
    # Width of unit hexagon in any direction n is between sqrt(3) (apothem*2) and 2 (vertex*2)?
    # No, width of regular hexagon side 1:
    # In direction normal to edge: 2 * apothem = sqrt(3).
    # In direction to vertex: 2 * radius = 2.
    # So the width of the "thickened" cluster is width_centers + w_hex(n).
    # w_hex(n) depends on the relative angle between the cluster orientation and the hexagon.
    # Since the inner hexagons can rotate, we can align them to minimize the width.
    # If we align inner hexagons such that their flat sides are parallel to the outer hexagon's flat sides,
    # then w_hex(n) = sqrt(3) for the 3 main directions?
    # Actually, if inner hexagons are rotated by 0 degrees, their width in the 6 normal directions of the outer hexagon (which are also at 0, 60, etc relative to outer) is:
    # If outer normal is at 0 deg, inner hex (flat top) has normal at 0 deg. Width is 2*apothem = sqrt(3).
    # So, if we align inner hexagons with outer hexagon, the required apothem A_outer must satisfy:
    # A_outer >= (width_centers(n_k) + sqrt(3)) / 2.
    # So S = 2 * A_outer / sqrt(3) = (width_centers(n_k) + sqrt(3)) / sqrt(3) = width_centers(n_k)/sqrt(3) + 1.
    # My fast calculation returned S based on width_centers only.
    # I need to adjust S.
    # S_actual = S_fast + 1.0 ?
    # Let's verify.
    # S_fast = max_width / sqrt(3).
    # S_actual = (max_width + sqrt(3)) / sqrt(3) = max_width/sqrt(3) + 1 = S_fast + 1.
    # This assumes we can align the inner hexagons optimally.
    # Since we can rotate each inner hexagon, we can always align one pair of parallel sides with the outer hexagon's sides.
    # So yes, add 1.0 to the side length.
    
    S_final = best_S + 1.0
    
    # We need to return the positions.
    # The angles of the inner hexagons should be aligned with the outer hexagon to achieve this minimal size.
    # Outer hexagon angle is best_angle.
    # So inner hexagons should also have angle best_angle.
    
    inner = []
    for x, y in best_positions:
        inner.append([float(x), float(y), float(best_angle)])
    
    center = [float(best_center[0]), float(best_center[1])]
    side = float(S_final)
    rotation = float(best_angle)
    
    return inner, center, side, rotation
