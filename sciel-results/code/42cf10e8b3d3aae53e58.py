import numpy as np
import time
from scipy.optimize import minimize

N = 12
START_TIME = time.time()
TIME_LIMIT = 120.0  # Leave some margin from 150s

def time_remaining():
    return TIME_LIMIT - (time.time() - START_TIME)

def hex_vertices(cx, cy, side, angle_deg):
    """
    Return vertices of a regular hexagon centered at (cx, cy) with side length 'side'
    and rotation angle in degrees.
    Vertices are at angles 0, 60, 120, 180, 240, 300 degrees (offset by angle_deg).
    """
    angles = np.radians(np.arange(6) * 60 + angle_deg)
    return cx + side * np.cos(angles), cy + side * np.sin(angles)

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """
    Check if point (px, py) is inside the hexagon centered at (cx, cy) with side 'side' and rotation 'angle_deg'.
    Uses the property that a regular hexagon is the intersection of 6 half-planes.
    For a hexagon with side s, the apothem (distance from center to side midpoint) is s * sqrt(3) / 2.
    The normals to the sides point outward.
    """
    # Normal vectors to the 6 sides.
    # For a standard hexagon (angle=0), vertices are at 0, 60, ...
    # Side 0 is between vertex 0 and 1. Midpoint angle is 30. Normal is 30.
    # Actually, let's use the distance from center to each side.
    # The distance from center to a side is the apothem a = side * sqrt(3)/2.
    # The normal direction for side i is at angle (i * 60 + 30) + angle_deg?
    # Let's check. Vertex i is at angle i*60 + angle_deg.
    # Side i connects vertex i and i+1.
    # Midpoint angle is (i*60 + (i+1)*60 + angle_deg)/2 = i*60 + 30 + angle_deg.
    # The outward normal is perpendicular to the side.
    # The side direction is tangent. The normal points away from center.
    # So normal angle is i*60 + 30 + angle_deg.
    
    # Wait, standard hexagon flat-top vs pointy-top.
    # If vertices are at 0, 60, ..., it's pointy-top (vertex at 0).
    # Side between 0 and 60 deg. Midpoint at 30 deg.
    # Vector from center to midpoint is at 30 deg. This is the outward normal.
    # So yes, normal angles are 30, 90, 150, 210, 270, 330 + angle_deg.
    
    normal_angles = np.radians(np.arange(6) * 60 + 30 + angle_deg)
    apothem = side * np.sqrt(3) / 2.0
    
    # Vector from hex center to point
    dx = px - cx
    dy = py - cy
    
    # Project point onto each normal
    # dist = dx * cos(n) + dy * sin(n)
    # If dist > apothem, point is outside that side.
    dists = dx * np.cos(normal_angles) + dy * np.sin(normal_angles)
    
    return np.all(dists <= apothem + 1e-9)

def check_collision(inner, outer_center, outer_side, outer_angle):
    """
    Check if any inner hexagon is outside the outer hexagon or collides with another.
    Returns True if valid (no collisions, all inside).
    """
    ox, oy = outer_center
    # Check containment
    for i in range(N):
        cx, cy, ang = inner[i]
        # Check if center is inside outer? No, need to check vertices or use Minkowski sum.
        # Two convex polygons P and Q are disjoint if and only if ...
        # But here we need P (inner) to be inside Q (outer).
        # This is equivalent to: all vertices of P are inside Q? No, that's not sufficient if P is large and Q is small?
        # Actually, if P is convex and Q is convex, P is inside Q iff all vertices of P are inside Q?
        # Yes, because Q is convex. If all vertices are inside, the convex hull (P) is inside.
        # Wait, this is only true if Q is convex. Yes, regular hexagon is convex.
        # So we just need to check if all 6 vertices of inner hexagon are inside outer hexagon.
        
        vx, vy = hex_vertices(cx, cy, 1.0, ang)
        for k in range(6):
            if not point_in_hexagon(vx[k], vy[k], ox, oy, outer_side, outer_angle):
                return False
        
        # Check collision with other inner hexagons
        # Two hexagons are disjoint if distance between centers > sum of inradii? No.
        # Two convex polygons are disjoint if there is a separating axis.
        # Or simpler: check if any vertex of A is inside B or any vertex of B is inside A?
        # No, that's for intersection. We want disjoint.
        # They are disjoint if no vertex of A is inside B AND no vertex of B is inside A?
        # Not quite. They could intersect without vertices inside?
        # For convex polygons, if they intersect, either a vertex is inside or edges cross.
        # But for "disjoint" (no overlap), we can check:
        # 1. Distance between centers > 2 * circumradius? No, that's sufficient but not necessary.
        # 2. Separating Axis Theorem (SAT).
        
        for j in range(i + 1, N):
            cx2, cy2, ang2 = inner[j]
            if not are_hexagons_disjoint(cx, cy, ang, cx2, cy2, ang2):
                return False

    return True

def are_hexagons_disjoint(cx1, cy1, ang1, cx2, cy2, ang2):
    """
    Check if two unit hexagons are disjoint.
    Using SAT or simple distance check?
    Since they are regular, we can check the distance between centers.
    The minimum distance between two non-overlapping regular hexagons of side 1 depends on orientation.
    The "inradius" (apothem) is sqrt(3)/2 ~ 0.866.
    The "circumradius" (center to vertex) is 1.
    If aligned, min distance is 2 * apothem = sqrt(3) ~ 1.732.
    If rotated 30 deg, min distance is 2 * circumradius? No.
    
    Let's use a robust check:
    Check if any vertex of hex1 is inside hex2 OR any vertex of hex2 is inside hex1.
    If neither, they might still touch or intersect edges?
    For convex polygons, if no vertex is inside the other, they are disjoint?
    Yes, for convex polygons, if they intersect, at least one vertex must be inside the other OR edges cross.
    But if edges cross, vertices are not necessarily inside.
    However, for regular hexagons, if they are "close", vertices are usually inside.
    To be safe, let's check edge intersections too? Or just use a slightly larger margin?
    
    Actually, a simpler sufficient condition for disjointness:
    Distance between centers > 2 * circumradius = 2.0. This is sufficient but not necessary.
    We want to pack them tight.
    
    Let's implement a proper SAT check for two convex polygons.
    """
    v1x, v1y = hex_vertices(cx1, cy1, 1.0, ang1)
    v2x, v2y = hex_vertices(cx2, cy2, 1.0, ang2)
    
    # SAT: For each edge normal, project both polygons. If projections don't overlap, they are disjoint.
    # Normals for hex1
    for i in range(6):
        # Edge from i to i+1
        x1, y1 = v1x[i], v1y[i]
        x2, y2 = v1x[(i+1)%6], v1y[(i+1)%6]
        # Normal vector
        dx = x2 - x1
        dy = y2 - y1
        # Normalize
        norm = np.sqrt(dx*dx + dy*dy)
        if norm < 1e-9: continue
        nx = -dy / norm
        ny = dx / norm
        
        # Project hex1
        p1 = v1x * nx + v1y * ny
        min1, max1 = np.min(p1), np.max(p1)
        
        # Project hex2
        p2 = v2x * nx + v2y * ny
        min2, max2 = np.min(p2), np.max(p2)
        
        # Check overlap
        if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
            return True # Disjoint
    
    # Also check normals from hex2?
    # For convex polygons, checking normals from one polygon is sufficient?
    # No, we need to check all potential separating axes, which are normals to edges of both.
    # But if they are disjoint, there exists a separating axis.
    # It is sufficient to check normals of edges of both polygons.
    
    for i in range(6):
        x1, y1 = v2x[i], v2y[i]
        x2, y2 = v2x[(i+1)%6], v2y[(i+1)%6]
        dx = x2 - x1
        dy = y2 - y1
        norm = np.sqrt(dx*dx + dy*dy)
        if norm < 1e-9: continue
        nx = -dy / norm
        ny = dx / norm
        
        p1 = v1x * nx + v1y * ny
        min1, max1 = np.min(p1), np.max(p1)
        
        p2 = v2x * nx + v2y * ny
        min2, max2 = np.min(p2), np.max(p2)
        
        if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
            return True
            
    return False # Not disjoint (overlap or touch)

def objective(params, outer_side, outer_angle, outer_center):
    """
    Objective function for optimization.
    We want to minimize outer_side, but it's a constraint.
    Instead, we can fix outer_side and check feasibility, then binary search on outer_side.
    Or, we can minimize a penalty function.
    
    Let's try: Minimize outer_side.
    Variables: positions and angles of inner hexagons, and outer_side.
    Constraints: All inner hexagons inside outer, no collisions.
    
    Since scipy.optimize.minimize doesn't handle hard constraints well for this,
    we can use a penalty method.
    Penalty = sum of violations.
    
    Violation 1: Inner hexagon outside outer.
    For each inner hex, calculate max distance of its vertices from outer center in the "hex metric"?
    Or simply: for each vertex of inner hex, if it's outside outer hex, add distance to boundary.
    
    Violation 2: Collision between inner hexes.
    For each pair, if they overlap, add overlap depth.
    
    This is complex. Let's try a different approach.
    
    Approach:
    1. Start with a large outer_side.
    2. Optimize positions to minimize some spread or just place them.
    3. Reduce outer_side.
    4. Repeat.
    
    Actually, let's use `scipy.optimize.minimize` with constraints.
    But defining constraints for polygon containment/collision is tricky.
    
    Alternative:
    Use a grid search or random restarts with local optimization.
    
    Let's try a simpler heuristic:
    1. Place hexagons in a honeycomb pattern.
    2. Optimize the scale/position.
    
    Honeycomb packing of hexagons is efficient.
    For N=12, we can arrange them in a hexagonal shape.
    e.g., 1-2-3-2-1 = 9? No.
    1-2-3-4-3-2-1 = 16?
    2-3-3-2 = 10?
    3-3-3-3 = 12? A 4x3 grid?
    
    Let's try to pack 12 hexagons in a hexagonal cluster.
    The smallest hexagonal number >= 12 is 19 (radius 2).
    So we can take 12 from a radius 2 hexagon.
    
    Let's generate a hexagonal grid of side length 2 (radius 2).
    Centers are at (i + j/2, j*sqrt(3)/2) in axial coordinates?
    
    Let's just use a generic local search.
    Start with random positions, then use `minimize` to reduce a penalty function.
    
    Penalty Function:
    P = sum( max(0, dist_outside) ) + sum( max(0, overlap_depth) )
    
    To make it differentiable, we can use smooth approximations.
    But for now, let's try a discrete optimization or simple gradient descent with subgradients.
    
    Given the time limit, a robust random search with local refinement might be better.
    
    Let's try:
    1. Initialize 12 hexagons in a circle or grid.
    2. Use `scipy.optimize.differential_evolution` or `basinhopping`?
       Basinhopping is good for local minima.
    3. Objective: Return outer_side required to contain all hexagons + collision penalty.
    
    How to calculate required outer_side?
    For a given set of inner hexagons, the minimal outer hexagon side length is the maximum over all inner hexagons of the "hexagonal distance" from outer center?
    No, the outer hexagon can also rotate.
    
    This is getting complicated. Let's simplify.
    Fix outer hexagon at origin, side S, angle 0.
    Optimize inner positions to fit inside S.
    If they fit, reduce S.
    
    Let's use a binary search on S.
    For a given S, check if 12 hexagons can fit.
    Use optimization to place them.
    
    Objective for placement (fixed S):
    Minimize sum of squared distances from center? No, just minimize violations.
    Violation = sum of (distance of vertices outside outer hex) + sum of (collision overlaps).
    
    If min violation is ~0, then S is feasible.
    
    Let's implement this.
    """
    # Extract positions and angles
    # params: [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11]
    inner = np.array(params).reshape(N, 3)
    
    penalty = 0.0
    
    # Containment penalty
    ox, oy = outer_center
    for i in range(N):
        cx, cy, ang = inner[i]
        vx, vy = hex_vertices(cx, cy, 1.0, ang)
        for k in range(6):
            # Check if vertex is outside outer hex
            # Calculate distance to boundary?
            # If outside, add the "excess" distance.
            # Approximation: project onto normals.
            normal_angles = np.radians(np.arange(6) * 60 + 30 + outer_angle)
            apothem = outer_side * np.sqrt(3) / 2.0
            dx = vx[k] - ox
            dy = vy[k] - oy
            dists = dx * np.cos(normal_angles) + dy * np.sin(normal_angles)
            excess = np.maximum(0, dists - apothem)
            penalty += np.sum(excess)
            
    # Collision penalty
    for i in range(N):
        for j in range(i+1, N):
            cx1, cy1, ang1 = inner[i]
            cx2, cy2, ang2 = inner[j]
            # Approximate collision by distance between centers
            # If dist < 2 * apothem (for aligned), they might collide.
            # Better: check if any vertex is inside the other?
            # For penalty, let's use a smooth approximation of the minimum distance between polygons.
            # Or just check center distance and add penalty if < 2.0 (circumradius sum).
            # This is a soft constraint.
            dist = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
            # Minimum distance for disjoint hexagons is sqrt(3) ~ 1.732 (if aligned sides).
            # If we use 1.8 as threshold, it's safe but not tight.
            # Let's use 1.732.
            if dist < 1.732:
                penalty += (1.732 - dist) ** 2
                
    return penalty

def solve():
    best_score = 100.0
    best_solution = None
    
    # Binary search on outer_side
    # Lower bound: sqrt(3) (one hexagon)
    # Upper bound: large
    lo = 1.732
    hi = 10.0
    
    # We will iterate a few times to refine
    for _ in range(20):
        mid = (lo + hi) / 2.0
        if time_remaining() < 10:
            break
            
        # Try to fit 12 hexagons in outer_side = mid
        # Use multiple random restarts
        found = False
        best_pen = float('inf')
        
        for restart in range(10):
            if time_remaining() < 5:
                break
                
            # Initial guess: random positions in a circle of radius mid
            r = mid * 0.5
            theta = np.random.uniform(0, 2*np.pi, N)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            angles = np.zeros(N)
            x0 = np.concatenate([x, y, angles])
            
            # Optimize
            try:
                res = minimize(objective, x0, args=(mid, 0.0, (0.0, 0.0)), method='Nelder-Mead',
                               options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-6})
                if res.fun < best_pen:
                    best_pen = res.fun
                    if res.fun < 1e-4: # Consider it feasible
                        found = True
                        best_solution = (res.x, (0.0, 0.0), mid, 0.0)
            except Exception as e:
                pass
                
        if found:
            hi = mid
        else:
            lo = mid
            
    if best_solution:
        return best_solution
    else:
        # Fallback: return a valid but large solution
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

def construct():
    result = solve()
    inner, center, side, angle = result
    # Ensure inner is list of lists
    inner = [list(map(float, row)) for row in inner]
    return inner, center, side, angle

# Quick test to ensure it runs
if __name__ == "__main__":
    inner, center, side, angle = construct()
    print(f"Side: {side}, Center: {center}, Angle: {angle}")
    print(f"Inner: {inner[:2]}...")
