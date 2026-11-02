"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 11

def hexagon_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
    angle_rad = np.radians(angle_deg)
    angles = angle_rad + np.pi / 3 * np.arange(6)
    return np.column_stack([cx + side * np.cos(angles),
                            cy + side * np.sin(angles)])

def hexagon_support_distance(theta1, theta2):
    """
    Minimum center-to-center distance for two unit regular hexagons
    with rotations theta1, theta2 (in radians) to not overlap.
    This is the support function of the Minkowski sum of two unit hexagons.
    For regular hexagons, the Minkowski sum is also a regular hexagon
    (if rotations are compatible) or a 12-gon.
    
    For two regular hexagons of side 1, the minimum distance for non-overlap
    when they have the same orientation is 2 (the diameter in the flat direction).
    More precisely, it's the maximum over all directions of the sum of support functions.
    
    For regular hexagons, the support function in direction u is:
    h(u) = max_i (v_i · u) where v_i are vertices.
    
    For the Minkowski sum, the support is h1(u) + h2(u).
    The minimum center distance is max_u (h1(u) + h2(u)).
    
    For a unit regular hexagon, the support function depends on orientation.
    Let's compute it numerically for given orientations.
    """
    # Vertices of unit hexagon at origin
    angles1 = theta1 + np.pi / 3 * np.arange(6)
    v1 = np.column_stack([np.cos(angles1), np.sin(angles1)])
    
    angles2 = theta2 + np.pi / 3 * np.arange(6)
    v2 = np.column_stack([np.cos(angles2), np.sin(angles2)])
    
    # Sample directions
    n_dirs = 360
    dirs = np.linspace(0, 2 * np.pi, n_dirs, endpoint=False)
    cos_dirs = np.cos(dirs)
    sin_dirs = np.sin(dirs)
    
    # Support of hex1 in each direction
    h1 = np.max(v1.T @ np.column_stack([cos_dirs, sin_dirs]), axis=0)
    h2 = np.max(v2.T @ np.column_stack([cos_dirs, sin_dirs]), axis=0)
    
    return np.max(h1 + h2)

def check_inner_feasible(positions, angles, outer_center, outer_side, outer_angle):
    """
    Check if all inner hexagons fit inside the outer hexagon without overlapping.
    positions: (n, 2) array of inner hexagon centers
    angles: (n,) array of inner hexagon rotation angles in radians
    outer_center: (2,) array
    outer_side: scalar
    outer_angle: scalar (radians)
    Returns: (feasible, max_violation)
    """
    n = len(positions)
    if n == 0:
        return True, 0.0
    
    # Check containment: each inner hexagon must be inside outer hexagon
    # An inner hexagon with center p, side 1, angle a is inside outer hexagon
    # with center c, side S, angle A if and only if for all directions,
    # the support of inner in that direction + distance from c to p in that direction
    # <= support of outer in that direction.
    
    # More practically: check that each vertex of each inner hexagon is inside the outer hexagon.
    # But that's not sufficient for convex containment... actually for convex polygons,
    # if all vertices of one are inside the other, it's contained.
    
    # Let's use the half-plane representation.
    # Outer hexagon: 6 half-planes.
    # For each inner hexagon, check all 6 vertices against all 6 half-planes.
    
    outer_verts = hexagon_vertices(outer_center[0], outer_center[1], outer_side, np.degrees(outer_angle))
    
    # Get half-plane normals for outer hexagon (inward pointing)
    # Edge i goes from vertex i to vertex i+1
    normals = np.zeros((6, 2))
    for i in range(6):
        v1 = outer_verts[i]
        v2 = outer_verts[(i + 1) % 6]
        edge = v2 - v1
        # Normal pointing inward: rotate edge by -90 degrees (clockwise)
        normal = np.array([edge[1], -edge[0]])
        normal = normal / np.linalg.norm(normal)
        # Check direction: should point toward center
        center_to_mid = outer_center - (v1 + v2) / 2
        if np.dot(normal, center_to_mid) < 0:
            normal = -normal
        normals[i] = normal
    
    max_violation = 0.0
    
    for i in range(n):
        # Check containment
        inner_verts = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, np.degrees(angles[i]))
        for j in range(6):
            for k in range(6):
                # Check if inner_verts[j] is inside half-plane k
                d = normals[k] @ (inner_verts[j] - outer_verts[k])
                if d > max_violation:
                    max_violation = d
        
        # Check overlap with other inner hexagons
        for j in range(i + 1, n):
            dist = np.linalg.norm(positions[i] - positions[j])
            min_dist = hexagon_support_distance(angles[i], angles[j])
            if dist < min_dist:
                violation = min_dist - dist
                if violation > max_violation:
                    max_violation = violation
    
    return max_violation < 1e-8, max_violation

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 20s margin
    
    # Initial configuration: grid of hexagons
    # For N hexagons, use a hexagonal grid arrangement
    # N = 11, so maybe a 3-4-3 or 2-3-3-3 arrangement
    
    # Let's start with a linear arrangement and optimize
    # Better: start with a compact hexagonal cluster
    
    # Hexagonal numbers: 1, 7, 19, 37...
    # For 11, use a central hexagon plus surrounding
    # Let's use a simple grid first
    
    # Place hexagons in a honeycomb pattern
    # Row-based placement
    positions = []
    angles = []
    
    # For N=11, use rows of 2, 3, 3, 2 or similar
    # Let's use a more systematic approach
    # Start with a 3x3 grid offset for hexagonal packing
    
    # Actually, let's just start with a line and let optimization work
    # But that's inefficient. Let's use a better initial guess.
    
    # Honeycomb arrangement for 11 hexagons:
    # Center at (0,0), then 6 around it, then 4 more
    # Or use rows: 3, 4, 3, 1? 
    
    # Let's use a simple approach: place them in a hexagonal grid pattern
    # The spacing between centers in a hexagonal grid of unit hexagons is 2 (for flat-top)
    
    # For flat-top hexagons, horizontal spacing is 3 (side*3 for flat-to-flat? No)
    # For unit hexagon, width (flat to flat) = sqrt(3) * side = sqrt(3)
    # Height (point to point) = 2 * side = 2
    
    # Let's use pointy-top for easier grid
    # Actually, let's just use a reasonable initial placement
    
    # Place in a compact cluster
    # Use the fact that for same orientation, min distance is 2
    # So a hexagonal grid with spacing 2 in the appropriate directions
    
    # Generate positions for a hexagonal grid
    # For N=11, use a 3-4-3 pattern or similar
    
    # Let's use a simpler initial guess: all in a line, then optimize
    # But that's far from optimal.
    
    # Better initial guess: hexagonal cluster
    # Center hexagon at (0,0)
    # 6 hexagons at distance 2 in directions 0, 60, 120, 180, 240, 300
    # Then 4 more at distance 4 in directions 30, 90, 150, 210, 270, 330 (but only 4)
    
    # Actually, let's use a more systematic approach
    # Generate all lattice points within a certain radius
    
    # For unit hexagons with same orientation, the center lattice is a triangular lattice
    # with spacing 2 in three directions 60 degrees apart
    
    # Let's generate points in a triangular lattice
    # Basis vectors: (2, 0) and (1, sqrt(3))
    # Or for pointy-top: (sqrt(3), 1) and (0, 2)
    
    # Let's use flat-top hexagons
    # Horizontal spacing: 3 (for adjacent in same row)
    # Vertical spacing: sqrt(3) * 1.5 = 2.598
    # Offset for odd rows: 1.5
    
    # Actually, for unit flat-top hexagons:
    # Width = 2, Height = sqrt(3)
    # Horizontal spacing between columns: 1.5
    # Vertical spacing between rows: sqrt(3)
    # Odd columns offset by sqrt(3)/2
    
    # Let's just use a simple grid and let optimization fix it
    
    # Initial: place in a 3x4 grid or similar
    # For N=11, use 3 rows: 4, 4, 3 or 3, 4, 4
    
    rows = [4, 4, 3]
    positions = []
    for r, count in enumerate(rows):
        for c in range(count):
            x = c * 3.0  # Horizontal spacing for flat-top
            y = r * 2.598  # Vertical spacing
            if r % 2 == 1:
                x += 1.5  # Offset for odd rows
            positions.append([x, y])
            angles.append(0.0)
    
    positions = np.array(positions)
    angles = np.array(angles)
    
    # Center the configuration
    positions -= np.mean(positions, axis=0)
    
    # Initial outer hexagon: large enough to contain everything
    # Compute bounding box
    min_x = np.min(positions[:, 0]) - 2
    max_x = np.max(positions[:, 0]) + 2
    min_y = np.min(positions[:, 1]) - 2
    max_y = np.max(positions[:, 1]) + 2
    
    width = max_x - min_x
    height = max_y - min_y
    
    # Initial outer side length
    outer_side = max(width, height) * 0.8 + 2
    outer_center = [0.0, 0.0]
    outer_angle = 0.0
    
    # Optimization variables:
    # - positions (2N)
    # - angles (N)
    # - outer_center (2)
    # - outer_side (1)
    # - outer_angle (1)
    # Total: 3N + 4 = 37 for N=11
    
    def objective(x):
        # Unpack
        pos = x[:2*N].reshape(N, 2)
        ang = x[2*N:3*N]
        oc = x[3*N:3*N+2]
        os = x[3*N+2]
        oa = x[3*N+3]
        
        # Ensure outer side is positive
        os = max(os, 1.0)
        
        # Check feasibility
        feasible, max_viol = check_inner_feasible(pos, ang, oc, os, oa)
        
        if feasible:
            return os
        else:
            # Penalize infeasibility
            return os + 10 * max_viol
    
    # Initial guess
    x0 = np.concatenate([positions.flatten(), angles, outer_center, [outer_side], [outer_angle]])
    
    # Bounds
    # Positions: reasonable range
    # Angles: [0, pi/3) due to hexagon symmetry
    # Outer side: [2, 20]
    # Outer center: [-10, 10]
    # Outer angle: [0, pi/3)
    
    bounds = []
    for i in range(N):
        bounds.append((-10, 10))
        bounds.append((-10, 10))
    for i in range(N):
        bounds.append((0, np.pi/3))
    bounds.append((-10, 10))
    bounds.append((-10, 10))
    bounds.append((2, 20))
    bounds.append((0, np.pi/3))
    
    # Try optimization
    best_result = None
    best_score = np.inf
    
    # Multiple restarts
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        if restart == 0:
            x_init = x0
        else:
            # Perturb
            x_init = x0 + np.random.randn(len(x0)) * 0.5
            x_init = np.clip(x_init, [b[0] for b in bounds], [b[1] for b in bounds])
        
        try:
            result = minimize(objective, x_init, method='L-BFGS-B', bounds=bounds,
                             options={'maxiter': 100, 'ftol': 1e-6})
            
            if result.fun < best_score:
                best_score = result.fun
                best_result = result.x
                
        except Exception:
            pass
    
    # Use best result if found
    if best_result is not None:
        pos = best_result[:2*N].reshape(N, 2)
        ang = best_result[2*N:3*N]
        oc = best_result[3*N:3*N+2]
        os = best_result[3*N+2]
        oa = best_result[3*N+3]
        
        # Final feasibility check
        feasible, max_viol = check_inner_feasible(pos, ang, oc, os, oa)
        
        if feasible:
            inner = []
            for i in range(N):
                inner.append([pos[i, 0], pos[i, 1], np.degrees(ang[i])])
            return inner, (oc[0], oc[1]), os, np.degrees(oa)
        else:
            # Fallback to initial
            pass
    
    # Fallback: return initial configuration with a safe outer hexagon
    # Compute a safe outer side length
    max_dist = np.max(np.linalg.norm(positions, axis=1))
    safe_side = max_dist + 2
    
    inner = []
    for i in range(N):
        inner.append([positions[i, 0], positions[i, 1], 0.0])
    
    return inner, (0.0, 0.0), safe_side, 0.0

# EVOLVE-BLOCK-END
