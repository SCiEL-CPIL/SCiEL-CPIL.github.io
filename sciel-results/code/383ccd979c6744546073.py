import numpy as np
import time
import random

N = 11

def _hex_vertices(cx, cy, angle_deg=0.0):
    """Return 6 vertices of a unit hexagon centered at (cx,cy) rotated by angle_deg."""
    angle_rad = np.deg2rad(angle_deg)
    base = np.array([0, 60, 120, 180, 240, 300], dtype=float) * np.pi / 180.0
    angles = base + angle_rad
    return np.column_stack((cx + np.cos(angles), cy + np.sin(angles)))

def _poly_overlap_sat(poly1, poly2):
    """SAT test: return True if two convex polygons overlap (interiors intersect)."""
    # Early exit if bounding boxes don't overlap
    min1 = poly1.min(axis=0)
    max1 = poly1.max(axis=0)
    min2 = poly2.min(axis=0)
    max2 = poly2.max(axis=0)
    if max1[0] < min2[0] or max2[0] < min1[0] or max1[1] < min2[1] or max2[1] < min1[1]:
        return False

    for poly in (poly1, poly2):
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            # Edge normal
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            # Project both polygons onto this normal
            proj1 = poly1 @ normal
            proj2 = poly2 @ normal
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            if max1 < min2 or max2 < min1:
                return False  # Separating axis found
    return True  # No separating axis -> overlap

def _check_nonoverlap(centers, angles_deg, tol=1e-9):
    """Check all pairs for non-overlap. Returns True if valid."""
    polys = [_hex_vertices(centers[i][0], centers[i][1], angles_deg[i]) for i in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            if _poly_overlap_sat(polys[i], polys[j]):
                return False
    return True

def _min_outer_side(centers, angles_deg, outer_angle_deg=0.0):
    """Compute minimum outer hexagon side length to contain all inner hexagons."""
    theta = np.deg2rad(outer_angle_deg)
    # 6 normals of outer hexagon (pointing outward from center)
    # Vertices of outer hexagon are at angles 0, 60, ...
    # Normals are perpendicular to edges.
    # Edge 1 is between angle 0 and 60. Midpoint angle is 30. Normal is 30.
    # So normals are at 30, 90, 150, 210, 270, 330.
    normals = np.array([[np.cos(theta + np.deg2rad(30 + k * 60)),
                         np.sin(theta + np.deg2rad(30 + k * 60))] for k in range(6)])
    
    max_proj = 0.0
    for i in range(N):
        verts = _hex_vertices(centers[i][0], centers[i][1], angles_deg[i])
        projs = verts @ normals.T
        local_max = np.max(projs)
        if local_max > max_proj:
            max_proj = local_max
    
    # For a regular hexagon with side S, the apothem (inradius) = S * sqrt(3)/2
    # max_proj is the max distance from center to any vertex along any normal
    # S = 2 * max_proj / sqrt(3)
    S = 2.0 * max_proj / np.sqrt(3.0)
    return S

def _init_ring_topology(rot_deg=0.0):
    """Center + 6 ring + 4 outer shell. Returns (centers, angles_deg)."""
    A = np.array([np.sqrt(3), 0.0])
    B = np.array([np.sqrt(3)/2, 1.5])
    
    # Ring topology: center, 6 neighbors, 4 from shell 2
    lattice = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
               (2,0), (1,1), (0,2), (-2,0)]
    
    centers = []
    for i, j in lattice:
        c = i * A + j * B
        centers.append(tuple(c))
    
    # Rotate
    rot_rad = np.deg2rad(rot_deg)
    cos_r, sin_r = np.cos(rot_rad), np.sin(rot_rad)
    rotated = [(x * cos_r - y * sin_r, x * sin_r + y * cos_r) for (x, y) in centers]
    
    angles_deg = [0.0] * N
    return rotated, angles_deg

def _calculate_cost(centers, angles_deg, outer_angle_deg, penalty_weight=1000.0):
    """Calculate cost: S + penalty * overlap_depth"""
    S = _min_outer_side(centers, angles_deg, outer_angle_deg)
    
    # Check overlaps and calculate penalty
    polys = [_hex_vertices(centers[i][0], centers[i][1], angles_deg[i]) for i in range(N)]
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = np.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1])
            if d < 2.0:
                if _poly_overlap_sat(polys[i], polys[j]):
                    penalty += max(0, 2.0 - d)
    return S + penalty_weight * penalty

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    # Initial configuration
    centers, angles_deg = _init_ring_topology(0.0)
    outer_angle_deg = 0.0
    
    # Track best valid solution
    best_valid_S = float('inf')
    best_valid_centers = None
    best_valid_angles = None
    best_valid_outer = None
    
    # Check initial
    if _check_nonoverlap(centers, angles_deg):
        S_init = _min_outer_side(centers, angles_deg, outer_angle_deg)
        if S_init < best_valid_S:
            best_valid_S = S_init
            best_valid_centers = [c[:] for c in centers]
            best_valid_angles = angles_deg[:]
            best_valid_outer = outer_angle_deg
    
    # Simulated Annealing
    T = 1.0  # Temperature
    T_min = 0.01
    alpha = 0.999  # Cooling rate
    steps = 0
    max_steps = 500000  # Large number, will stop by time
    
    current_centers = [c[:] for c in centers]
    current_angles = angles_deg[:]
    current_outer = outer_angle_deg
    current_cost = _calculate_cost(current_centers, current_angles, current_outer)
    
    rng = np.random.RandomState(42)
    
    while time.time() < deadline - 5.0:
        # Perturb one hexagon
        idx = rng.randint(N)
        dx = rng.uniform(-0.5, 0.5)
        dy = rng.uniform(-0.5, 0.5)
        da = rng.uniform(-10, 10)  # degrees
        
        old_cx, old_cy = current_centers[idx]
        old_ang = current_angles[idx]
        
        new_cx = old_cx + dx
        new_cy = old_cy + dy
        new_ang = old_ang + da
        
        current_centers[idx] = (new_cx, new_cy)
        current_angles[idx] = new_ang
        
        new_cost = _calculate_cost(current_centers, current_angles, current_outer)
        
        delta_cost = new_cost - current_cost
        if delta_cost < 0 or rng.rand() < np.exp(-delta_cost / T):
            current_cost = new_cost
            # Accept move
            # Check if valid and better
            if _check_nonoverlap(current_centers, current_angles):
                S_valid = _min_outer_side(current_centers, current_angles, current_outer)
                if S_valid < best_valid_S:
                    best_valid_S = S_valid
                    best_valid_centers = [c[:] for c in current_centers]
                    best_valid_angles = current_angles[:]
                    best_valid_outer = current_outer
        else:
            # Reject move, revert
            current_centers[idx] = (old_cx, old_cy)
            current_angles[idx] = old_ang
        
        T *= alpha
        if T < T_min:
            T = T_min
        
        # Occasionally try changing outer angle
        if steps % 100 == 0:
            old_outer = current_outer
            current_outer = old_outer + rng.uniform(-5, 5)
            new_cost_outer = _calculate_cost(current_centers, current_angles, current_outer)
            if new_cost_outer < current_cost:
                current_cost = new_cost_outer
            else:
                current_outer = old_outer
        
        steps += 1
        
        # Periodic check for time
        if time.time() > deadline - 10.0:
            break
    
    # If we didn't find a valid solution (unlikely with ring init), use init
    if best_valid_centers is None:
        best_valid_centers = centers
        best_valid_angles = angles_deg
        best_valid_outer = outer_angle_deg
        best_valid_S = _min_outer_side(best_valid_centers, best_valid_angles, best_valid_outer)

    # Final verification
    inner = [[c[0], c[1], a] for c, a in zip(best_valid_centers, best_valid_angles)]
    
    # Double check validity before returning
    if not _check_nonoverlap(best_valid_centers, best_valid_angles):
        # Fallback to initial ring topology which is known to be valid
        centers, angles_deg = _init_ring_topology(0.0)
        inner = [[c[0], c[1], a] for c, a in zip(centers, angles_deg)]
        best_valid_S = _min_outer_side(centers, angles_deg, 0.0)
        best_valid_outer = 0.0

    return inner, (0.0, 0.0), best_valid_S, best_valid_outer

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
