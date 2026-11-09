"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    sqrt3 = math.sqrt(3.0)
    
    def get_hex_vertices(cx, cy, angle_deg, side=1.0):
        """Get vertices of a regular hexagon."""
        vertices = []
        angle_rad = math.radians(angle_deg)
        for i in range(6):
            theta = angle_rad + i * math.pi / 3.0
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return vertices

    def point_in_hex(px, py, cx, cy, angle_deg, side=1.0):
        """Check if a point is inside a regular hexagon."""
        # Transform point to hexagon's local coordinate system
        angle_rad = math.radians(angle_deg)
        # Rotate point by -angle
        dx = px - cx
        dy = py - cy
        x_rot = dx * math.cos(-angle_rad) - dy * math.sin(-angle_rad)
        y_rot = dx * math.sin(-angle_rad) + dy * math.cos(-angle_rad)
        
        # Hexagon with flat top (vertices at 0, 60, ... degrees) 
        # Actually, standard regular hexagon with side 1:
        # Vertices at (1,0), (1/2, sqrt3/2), ...
        # Let's check distance to edges.
        # A regular hexagon with side s is the intersection of 6 half-planes.
        # The distance from center to each edge (apothem) is s * sqrt(3) / 2.
        # The normal to each edge is at angle 30 + i*60.
        
        apothem = side * sqrt3 / 2.0
        
        for i in range(6):
            # Normal direction for edge i
            # Edge i is perpendicular to radius at angle 30 + i*60
            theta = math.radians(30.0 + i * 60.0)
            nx = math.cos(theta)
            ny = math.sin(theta)
            
            # Project rotated point onto this normal
            proj = x_rot * nx + y_rot * ny
            
            # If projection exceeds apothem, point is outside
            if proj > apothem + 1e-9:
                return False
        return True

    def check_overlap(centers, angles, eps=1e-6):
        """Check if any two hexagons overlap. Returns True if they do."""
        n = len(centers)
        for i in range(n):
            for j in range(i + 1, n):
                # Quick check: distance between centers
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                
                # If distance is greater than 2*side (diameter of circumcircle), no overlap
                # Actually, max distance between any two points in two unit hexagons
                # If centers are far enough, we can skip detailed check.
                # Min distance between two non-overlapping unit hexagons can be small.
                # Max distance between any two points in a unit hexagon is 2.
                # So if center distance > 2 + 2 = 4, definitely no overlap.
                # But we need strict non-overlap.
                
                # Detailed check: Check vertices of i against hex j and vice versa
                verts_i = get_hex_vertices(centers[i][0], centers[i][1], angles[i])
                verts_j = get_hex_vertices(centers[j][0], centers[j][1], angles[j])
                
                # Check if any vertex of i is inside j
                for vx, vy in verts_i:
                    if point_in_hex(vx, vy, centers[j][0], centers[j][1], angles[j]):
                        return True
                
                # Check if any vertex of j is inside i
                for vx, vy in verts_j:
                    if point_in_hex(vx, vy, centers[i][0], centers[i][1], angles[i]):
                        return True
                        
                # Check edge intersections? 
                # For convex polygons, if no vertex is inside the other, they don't overlap.
                # This is true unless one is completely inside the other without vertices inside?
                # No, if A is inside B, vertices of A are inside B.
                # If B is inside A, vertices of B are inside A.
                # If they intersect, either a vertex is inside or edges cross.
                # For convex polygons, if no vertex of A is in B and no vertex of B is in A,
                # they are disjoint.
                
        return False

    def required_L(centers, angles, outer_angle=0.0):
        """Calculate minimum side length of outer hexagon to enclose all inner hexagons."""
        # The outer hexagon is centered at (0,0) for simplicity in calculation, 
        # but the problem allows translation of outer hexagon.
        # However, due to symmetry and translation invariance, we can fix outer center at (0,0)
        # and optimize inner positions. Or optimize outer center too?
        # The interface returns outer center.
        # Let's fix outer center at (0,0) to reduce variables, as the inner hexagons 
        # will center themselves around it.
        
        # For a regular hexagon of side L centered at origin, rotated by outer_angle,
        # a point (x,y) is inside if its distance to each of the 6 edges is <= apothem.
        # Apothem = L * sqrt(3) / 2.
        
        # We need to find L such that all vertices of all inner hexagons are inside.
        # For a specific point (x,y), the minimum L required for it to be inside 
        # a hexagon centered at origin with rotation outer_angle is:
        # L >= (2 / sqrt(3)) * max_i (dot product of (x,y) rotated by -outer_angle with edge normals)
        
        outer_angle_rad = math.radians(outer_angle)
        max_proj = 0.0
        
        for i in range(N):
            verts = get_hex_vertices(centers[i][0], centers[i][1], angles[i])
            for vx, vy in verts:
                # Rotate point by -outer_angle
                x_rot = vx * math.cos(-outer_angle_rad) - vy * math.sin(-outer_angle_rad)
                y_rot = vx * math.sin(-outer_angle_rad) + vy * math.cos(-outer_angle_rad)
                
                # Project onto 6 edge normals
                for k in range(6):
                    theta = math.radians(30.0 + k * 60.0)
                    nx = math.cos(theta)
                    ny = math.sin(theta)
                    proj = x_rot * nx + y_rot * ny
                    if proj > max_proj:
                        max_proj = proj
        
        # L * sqrt(3) / 2 >= max_proj  => L >= 2 * max_proj / sqrt(3)
        return 2.0 * max_proj / sqrt3

    def objective_and_constraints(params):
        """
        Objective: Minimize the enclosing side length L.
        Constraints: No overlap.
        """
        centers = []
        angles = []
        for i in range(N):
            cx = params[2*i]
            cy = params[2*i+1]
            ang = params[2*N + i]
            centers.append((cx, cy))
            angles.append(ang)
            
        # Calculate required L
        L = required_L(centers, angles, 0.0)
        
        # Check overlap
        if check_overlap(centers, angles):
            return L + 100.0 # Penalty for overlap
        
        return L

    # Initial bounds
    # A rough estimate for L is around 4-5.
    # Centers should be within [-L, L] roughly.
    bounds = []
    for i in range(N):
        bounds.append((-5.0, 5.0)) # x
        bounds.append((-5.0, 5.0)) # y
    for i in range(N):
        bounds.append((0.0, 180.0)) # angle (symmetry)

    # Use Differential Evolution for global search
    # This is robust but can be slow. We use a small population and max iterations.
    
    best_L = float('inf')
    best_params = None
    
    # Seed initial guess
    # Arrange in a circle or grid
    init_centers = []
    init_angles = []
    for i in range(N):
        angle = 2 * math.pi * i / N
        r = 2.0
        init_centers.append((r * math.cos(angle), r * math.sin(angle)))
        init_angles.append(0.0)
        
    init_params = []
    for i in range(N):
        init_params.append(init_centers[i][0])
        init_params.append(init_centers[i][1])
    for i in range(N):
        init_params.append(init_angles[i])

    # Run DE
    try:
        # maxiter=50, popsize=20 might be too slow? 
        # 36 vars. DE population size ~ 10*36 = 360. 
        # 50 iterations * 360 evals = 18000 evals.
        # Each eval is ~ N^2 * 6 checks = 144 * 6 = 864 point_in_hex calls.
        # 18000 * 864 = 15.5 million calls. Python might be slow.
        # Let's reduce.
        
        # We will use a simpler local search from multiple random starts if DE is too slow.
        # But let's try DE with very small budget first.
        
        res = differential_evolution(
            objective_and_constraints, 
            bounds, 
            maxiter=20, 
            popsize=10, 
            seed=42,
            tol=1e-4,
            init='latinhypercube',
            polish=False
        )
        
        if res.success or res.fun < best_L:
            best_L = res.fun
            best_params = res.x
            if time.time() > deadline:
                # Return best found so far
                centers = []
                angles = []
                for i in range(N):
                    centers.append([best_params[2*i], best_params[2*i+1]])
                    angles.append(best_params[2*N + i])
                inner = []
                for i in range(N):
                    inner.append([centers[i][0], centers[i][1], angles[i]])
                return inner, (0.0, 0.0), best_L, 0.0

    except Exception as e:
        pass

    # Refine with local search (Nelder-Mead)
    if best_params is not None:
        try:
            res2 = minimize(
                objective_and_constraints, 
                best_params, 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            if res2.fun < best_L:
                best_L = res2.fun
                best_params = res2.x
        except Exception:
            pass

    # Construct return value
    centers = []
    angles = []
    for i in range(N):
        centers.append([best_params[2*i], best_params[2*i+1]])
        angles.append(best_params[2*N + i])
    
    inner = []
    for i in range(N):
        inner.append([centers[i][0], centers[i][1], angles[i]])
        
    # Final check and L calculation
    final_L = required_L(centers, angles, 0.0)
    
    # If overlap exists in the final solution, we might need to adjust or report a larger L
    # But the objective penalized overlap, so it should be valid.
    # However, if it's invalid, the score is 0.
    # Let's verify validity.
    if check_overlap(centers, angles):
        # Fallback to a known valid configuration if optimization failed to find valid one
        # Simple grid:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return inner, (0.0, 0.0), final_L, 0.0
# EVOLVE-BLOCK-END
