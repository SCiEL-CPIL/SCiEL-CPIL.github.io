"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import differential_evolution

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_rad):
        """Vertices of a regular hexagon (flat-top when angle=0)."""
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def polygons_disjoint_sat(poly1, poly2, tol=1e-9):
        """Check if two convex polygons are disjoint using SAT."""
        for poly in [poly1, poly2]:
            for i in range(len(poly)):
                v1 = poly[i]
                v2 = poly[(i + 1) % len(poly)]
                edge = v2 - v1
                normal = np.array([-edge[1], edge[0]])
                
                # Project both polygons onto the normal
                proj1 = np.dot(poly1, normal)
                proj2 = np.dot(poly2, normal)
                
                min1, max1 = np.min(proj1), np.max(proj1)
                min2, max2 = np.min(proj2), np.max(proj2)
                
                # Check for gap
                if max1 < min2 - tol or max2 < min1 - tol:
                    return True  # Disjoint (found separating axis)
        return False  # Overlapping

    def hexagons_disjoint(inner_hexes, outer_vertices, tol=1e-9):
        """Check that all inner hexagons are disjoint from each other and inside outer."""
        n = len(inner_hexes)
        
        # Check pairwise disjointness of inner hexagons
        for i in range(n):
            for j in range(i + 1, n):
                if not polygons_disjoint_sat(inner_hexes[i], inner_hexes[j], tol):
                    return False
        
        # Check all inner hexagons are inside outer
        for i in range(n):
            # For each vertex of inner hexagon, check if inside outer
            for v in inner_hexes[i]:
                inside = True
                for k in range(len(outer_vertices)):
                    v1 = outer_vertices[k]
                    v2 = outer_vertices[(k + 1) % len(outer_vertices)]
                    edge = v2 - v1
                    normal = np.array([-edge[1], edge[0]])
                    # Normal should point inward
                    center_dir = np.array([0.0, 0.0]) - v1  # from vertex to center
                    if np.dot(normal, center_dir) < 0:
                        normal = -normal
                    if np.dot(v - v1, normal) < -tol:
                        inside = False
                        break
                if not inside:
                    return False
        return True

    def compute_min_outer_side(inner_positions, inner_angles, outer_center, outer_angle):
        """Compute minimum outer hexagon side that contains all inner hexagons."""
        # For each inner hexagon, find max distance from outer center to any vertex
        # Then compute required outer side length
        
        # Actually, we need to find the smallest outer hexagon (with given center and angle)
        # that contains all inner hexagons
        
        # For a regular hexagon with center c and angle a, the distance from center
        # to each edge is s * cos(0) = s (apothem = s for unit side? No)
        # Apothem of regular hexagon with side s is s * sqrt(3)/2
        
        # Let's compute: for outer hexagon with side S, center (cx,cy), angle theta,
        # the constraint is that all points must satisfy:
        # For each of 6 directions, the projection must be within [-S*sqrt(3)/2, S*sqrt(3)/2]
        
        # Actually, let me think differently. The outer hexagon has 6 edges.
        # For each edge, we need all inner hexagon vertices to be on the correct side.
        
        # Let's compute the required S by checking each of the 6 half-planes.
        
        cx, cy = outer_center
        theta = outer_angle
        
        max_required = 0.0
        for i in range(N):
            vx, vy, va = inner_positions[i], inner_positions[i + N], inner_angles[i]
            # Wait, I need to restructure. Let me use a different parameterization.
            pass
        
        # Let me restructure the objective function properly
        return 100.0  # placeholder

    def objective(params):
        """Objective: minimize outer side length. Returns large value if invalid."""
        if time.time() > deadline:
            return 1e6
        
        # params: [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11, cx, cy, ctheta]
        # 12*3 + 3 = 39 parameters
        
        inner_x = params[0:12]
        inner_y = params[12:24]
        inner_a = params[24:36]
        outer_cx = params[36]
        outer_cy = params[37]
        outer_a = params[38]
        
        # Build inner hexagons
        inner_hexes = []
        for i in range(N):
            verts = hexagon_vertices(inner_x[i], inner_y[i], 1.0, inner_a[i])
            inner_hexes.append(verts)
        
        # Check disjointness of inner hexagons
        for i in range(N):
            for j in range(i + 1, N):
                if not polygons_disjoint_sat(inner_hexes[i], inner_hexes[j], 1e-9):
                    return 1e6
        
        # Compute minimum outer side that contains all inner hexagons
        # For outer hexagon with center (cx,cy), angle theta, side S:
        # The 6 edge normals are at angles theta + k*60 degrees for k=0..5
        # Each edge is at distance S*sqrt(3)/2 from center (apothem)
        
        cx, cy = outer_cx, outer_cy
        theta = outer_a
        
        max_required = 0.0
        for i in range(N):
            for v in inner_hexes[i]:
                # For each of the 6 directions, compute required S
                for k in range(6):
                    edge_angle = theta + np.radians(k * 60)
                    # Normal pointing outward
                    nx = np.cos(edge_angle)
                    ny = np.sin(edge_angle)
                    # Distance from center to this point along normal
                    dist = (v[0] - cx) * nx + (v[1] - cy) * ny
                    # Required apothem is max(dist, 0)
                    # S = apothem / (sqrt(3)/2) = apothem * 2/sqrt(3)
                    if dist > 0:
                        required_S = dist * 2.0 / np.sqrt(3.0)
                        if required_S > max_required:
                            max_required = required_S
        
        return max_required

    # Create a valid initial configuration: hexagons in a line with spacing 2*sqrt(3)
    # Actually, for flat-top hexagons, horizontal spacing for disjoint is 2*sqrt(3)
    # Let's use a grid pattern
    
    # Initial guess: hexagons arranged in two rows
    # Row 1: 6 hexagons at y = 0
    # Row 2: 6 hexagons at y = 3 (sqrt(3) * 2 for flat-top)
    
    init_x = []
    init_y = []
    init_a = []
    for i in range(6):
        init_x.append(i * 2 * np.sqrt(3))
        init_y.append(0.0)
        init_a.append(0.0)
    for i in range(6):
        init_x.append((i + 0.5) * 2 * np.sqrt(3))
        init_y.append(3.0)  # sqrt(3) * 2
        init_a.append(0.0)
    
    init_params = init_x + init_y + init_a + [0.0, 0.0, 0.0]
    
    # Bounds for differential evolution
    bounds = []
    for i in range(N):
        bounds.append((-20, 20))  # x
    for i in range(N):
        bounds.append((-20, 20))  # y
    for i in range(N):
        bounds.append((0, 2 * np.pi))  # angle
    bounds.append((-10, 10))  # outer cx
    bounds.append((-10, 10))  # outer cy
    bounds.append((0, 2 * np.pi))  # outer angle
    
    # Run differential evolution
    best_result = None
    try:
        result = differential_evolution(
            objective, bounds,
            maxiter=50, popsize=20,
            tol=1e-6, mutation=(0.5, 1.0),
            recombination=0.7, seed=42,
            polish=True
        )
        
        if result.success and result.fun < 1e5:
            best_result = result
    except Exception:
        pass
    
    if best_result is None or best_result.fun >= 1e5:
        # Return a valid but not optimal configuration
        # Use the initial configuration with a large outer hexagon
        inner = [[init_x[i], init_y[i], np.degrees(init_a[i])] for i in range(N)]
        return inner, (0.0, 0.0), 30.0, 0.0
    
    # Extract best solution
    params = best_result.x
    inner_x = params[0:12]
    inner_y = params[12:24]
    inner_a = params[24:36]
    outer_cx = params[36]
    outer_cy = params[37]
    outer_a = params[38]
    
    inner = [[inner_x[i], inner_y[i], np.degrees(inner_a[i])] for i in range(N)]
    outer_side = best_result.fun
    
    return inner, (outer_cx, outer_cy), outer_side, np.degrees(outer_a)
# EVOLVE-BLOCK-END
