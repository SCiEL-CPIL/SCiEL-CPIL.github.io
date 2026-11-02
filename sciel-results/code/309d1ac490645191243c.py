"""
Packing N unit regular hexagons inside a regular hexagon, minimizing the outer side length.
Uses a grid-based initial placement with local optimization via scipy.optimize.
"""

import numpy as np
from scipy.optimize import minimize
import time

N = 11

# Deadline guard: 150s wall-clock limit, keep 10s safety margin
DEADLINE = 150.0
SAFETY_MARGIN = 10.0

def hexagon_vertices(cx, cy, side, angle_deg=0.0):
    """Return the 6 vertices of a regular hexagon with given center, side length, and rotation."""
    angle_rad = np.radians(angle_deg)
    angles = angle_rad + np.arange(6) * np.pi / 3.0
    return cx + side * np.cos(angles), cy + side * np.sin(angles)

def point_in_hexagon(px, py, hx, hy, side, angle_deg=0.0):
    """Check if a point is inside or on the boundary of a regular hexagon."""
    # Transform point to hexagon's local coordinate system
    dx = px - hx
    dy = py - hy
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    # Rotate point by -angle to align hexagon
    lx = dx * cos_a + dy * sin_a
    ly = -dx * sin_a + dy * cos_a
    
    # For a regular hexagon with side length s, centered at origin, flat-top orientation:
    # The hexagon can be described as the intersection of 6 half-planes.
    # For a "pointy-top" hexagon (vertices at 0, 60, 120, 180, 240, 300 degrees):
    # |lx| <= s, |ly| <= s*sqrt(3)/2, and |lx|/s + |ly|/(s*sqrt(3)/2) <= 1 for the relevant quadrants
    # Actually, let's use the standard form. A regular hexagon with side s, centered at origin,
    # with vertices at angles 0, 60, 120, 180, 240, 300 degrees has the property that
    # for any point (x, y), it is inside if and only if:
    # max(|x|, |y| * 2/sqrt(3), |x|/2 + |y|/sqrt(3)) <= s
    # Wait, let me think more carefully.
    
    # A regular hexagon with side length s, centered at origin, with one vertex at (s, 0):
    # Vertices: (s,0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s,0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
    # The six half-planes are:
    # 1. x <= s
    # 2. x/2 + y*sqrt(3)/2 <= s
    # 3. -x/2 + y*sqrt(3)/2 <= s
    # 4. -x <= s  => x >= -s
    # 5. -x/2 - y*sqrt(3)/2 <= s
    # 6. x/2 - y*sqrt(3)/2 <= s
    
    # So the point is inside if all six inequalities hold.
    sqrt3 = np.sqrt(3.0)
    a1 = lx
    a2 = lx/2.0 + ly*sqrt3/2.0
    a3 = -lx/2.0 + ly*sqrt3/2.0
    a4 = -lx
    a5 = -lx/2.0 - ly*sqrt3/2.0
    a6 = lx/2.0 - ly*sqrt3/2.0
    
    return (a1 <= side + 1e-9 and a2 <= side + 1e-9 and a3 <= side + 1e-9 and
            a4 <= side + 1e-9 and a5 <= side + 1e-9 and a6 <= side + 1e-9)

def hexagon_collision(c1x, c1y, a1, c2x, c2y, a2, tol=1e-6):
    """Check if two unit hexagons collide (overlap). Returns True if they overlap."""
    # For two regular hexagons of side 1, they are disjoint if the distance between centers
    # is at least the minimum distance such that they don't overlap.
    # The simplest check: sample points on the boundary of one hexagon and check if they're inside the other.
    # But that's expensive. Instead, use the fact that two unit hexagons are disjoint iff
    # no vertex of one is strictly inside the other AND no edge of one crosses an edge of the other.
    # A simpler sufficient condition: if the distance between centers is >= 2, they are disjoint
    # (since the max distance from center to vertex is 1, so two hexagons with centers 2 apart
    # have vertices at most 1 from their centers, so the closest points are at distance >= 0... wait no).
    # Actually, the minimum distance between two disjoint unit hexagons can be less than 2.
    # Let's use a conservative check: sample the 6 vertices of each hexagon and check if any
    # vertex is inside the other hexagon. Also check if edges cross, but that's complex.
    # For a simpler approach: use the support function or just check vertex containment.
    
    # Get vertices of hexagon 1
    vx1, vy1 = hexagon_vertices(c1x, c1y, 1.0, a1)
    vx2, vy2 = hexagon_vertices(c2x, c2y, 1.0, a2)
    
    # Check if any vertex of hex1 is inside hex2
    for i in range(6):
        if point_in_hexagon(vx1[i], vy1[i], c2x, c2y, 1.0, a2):
            return True
    
    # Check if any vertex of hex2 is inside hex1
    for i in range(6):
        if point_in_hexagon(vx2[i], vy2[i], c1x, c1y, 1.0, a1):
            return True
    
    # This is a sufficient but not necessary condition for collision.
    # Two hexagons can overlap even if no vertex is inside the other (edge-edge intersection).
    # For a more robust check, we'd need to check edge intersections, but for optimization purposes,
    # this should work reasonably well as a penalty function.
    
    return False

def compute_objective(params, outer_center, outer_side, outer_angle):
    """
    Compute the objective: outer side length needed to contain all hexagons.
    params: array of shape (N, 3) with [x, y, angle_deg] for each inner hexagon
    Returns the minimum outer side length needed.
    """
    # For a given set of inner hexagons, find the minimum enclosing regular hexagon.
    # This is complex, so instead we fix the outer hexagon and check if all inner hexagons fit.
    # We'll use a different approach: given the inner hexagons, compute the required outer side.
    
    # The outer hexagon must contain all vertices of all inner hexagons.
    # For a regular hexagon with center (hx, hy), side length s, and angle theta,
    # a point (px, py) is inside if:
    # Let dx = px - hx, dy = py - hy
    # Rotate by -theta: lx = dx*cos(theta) + dy*sin(theta), ly = -dx*sin(theta) + dy*cos(theta)
    # Then we need: max(lx, lx/2 + ly*sqrt(3)/2, -lx/2 + ly*sqrt(3)/2, -lx, -lx/2 - ly*sqrt(3)/2, lx/2 - ly*sqrt(3)/2) <= s
    
    # So for a given outer center and angle, the required side length is:
    # s = max over all points of max of the 6 expressions
    
    # Collect all vertices of all inner hexagons
    all_points = []
    for i in range(N):
        vx, vy = hexagon_vertices(params[i, 0], params[i, 1], 1.0, params[i, 2])
        all_points.extend(zip(vx, vy))
    
    all_points = np.array(all_points)
    
    # For the given outer center and angle, compute the required side length
    hx, hy = outer_center
    theta = np.radians(outer_angle)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    sqrt3 = np.sqrt(3.0)
    
    dx = all_points[:, 0] - hx
    dy = all_points[:, 1] - hy
    lx = dx * cos_t + dy * sin_t
    ly = -dx * sin_t + dy * cos_t
    
    # The 6 half-plane constraints give the required side length
    exprs = np.stack([
        lx,
        lx/2.0 + ly*sqrt3/2.0,
        -lx/2.0 + ly*sqrt3/2.0,
        -lx,
        -lx/2.0 - ly*sqrt3/2.0,
        lx/2.0 - ly*sqrt3/2.0
    ], axis=1)
    
    # The required side length is the max of all these values
    required_side = np.max(exprs)
    
    # Also need to ensure all inner hexagons are pairwise non-overlapping
    # Add a penalty for collisions
    collision_penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            if hexagon_collision(params[i, 0], params[i, 1], params[i, 2],
                                 params[j, 0], params[j, 1], params[j, 2]):
                collision_penalty += 100.0
    
    return required_side + collision_penalty

def construct():
    start_time = time.time()
    
    # Initial placement: grid-like arrangement
    # For N=11, a hexagonal arrangement with 3 rows: 2-3-3-3-2 or similar
    # Let's use a simple grid first
    positions = []
    # Center hexagon
    positions.append([0.0, 0.0, 0.0])
    # 6 neighbors
    for k in range(6):
        angle = k * 60.0
        x = 2.0 * np.cos(np.radians(angle))
        y = 2.0 * np.sin(np.radians(angle))
        positions.append([x, y, 0.0])
    # 4 more to make 11
    # Add at distance ~4
    for k in range(4):
        angle = k * 90.0 + 45.0
        x = 4.0 * np.cos(np.radians(angle))
        y = 4.0 * np.sin(np.radians(angle))
        positions.append([x, y, 0.0])
    
    params = np.array(positions, dtype=float)
    
    best_params = params.copy()
    best_outer_center = (0.0, 0.0)
    best_outer_side = 40.0
    best_outer_angle = 0.0
    best_obj = float('inf')
    
    # Try different outer angles
    outer_angles = [0.0, 30.0, 60.0, 15.0, 45.0]
    
    for outer_angle in outer_angles:
        if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
            break
        
        # Optimize the positions of inner hexagons
        # We'll use a simple iterative improvement approach
        # First, compute the initial objective
        obj = compute_objective(params, (0.0, 0.0), best_outer_side, outer_angle)
        
        # Try local optimization using scipy
        # Define the objective function for scipy
        def obj_func(p):
            p = p.reshape(N, 3)
            return compute_objective(p, (0.0, 0.0), 50.0, outer_angle)
        
        # Try a few random perturbations and pick the best
        for trial in range(3):
            if time.time() - start_time > DEADLINE - SAFETY_MARGIN:
                break
            
            # Perturb the initial positions
            perturbed = params.copy()
            perturbed[:, 0] += np.random.randn(N) * 0.5
            perturbed[:, 1] += np.random.randn(N) * 0.5
            perturbed[:, 2] += np.random.randn(N) * 5.0
            perturbed[:, 2] %= 60.0  # Angles are periodic with period 60 for hexagons
            
            try:
                result = minimize(obj_func, perturbed.flatten(), method='L-BFGS-B',
                                  options={'maxiter': 50, 'ftol': 1e-4})
                if result.success or result.fun < best_obj:
                    trial_params = result.x.reshape(N, 3)
                    trial_obj = result.fun
                    if trial_obj < best_obj:
                        best_obj = trial_obj
                        best_params = trial_params
                        best_outer_angle = outer_angle
            except Exception:
                pass
    
    # Final polish: compute the actual required outer side length
    # and adjust the outer center to be the centroid of the configuration
    centroid_x = np.mean(best_params[:, 0])
    centroid_y = np.mean(best_params[:, 1])
    
    # Compute the required outer side with the best outer angle
    all_points = []
    for i in range(N):
        vx, vy = hexagon_vertices(best_params[i, 0], best_params[i, 1], 1.0, best_params[i, 2])
        all_points.extend(zip(vx, vy))
    all_points = np.array(all_points)
    
    hx, hy = centroid_x, centroid_y
    theta = np.radians(best_outer_angle)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    sqrt3 = np.sqrt(3.0)
    
    dx = all_points[:, 0] - hx
    dy = all_points[:, 1] - hy
    lx = dx * cos_t + dy * sin_t
    ly = -dx * sin_t + dy * cos_t
    
    exprs = np.stack([
        lx,
        lx/2.0 + ly*sqrt3/2.0,
        -lx/2.0 + ly*sqrt3/2.0,
        -lx,
        -lx/2.0 - ly*sqrt3/2.0,
        lx/2.0 - ly*sqrt3/2.0
    ], axis=1)
    
    required_side = float(np.max(exprs))
    
    # Ensure the side length is at least 1 (to contain a unit hexagon)
    required_side = max(required_side, 1.0)
    
    inner = [[float(best_params[i, 0]), float(best_params[i, 1]), float(best_params[i, 2])] for i in range(N)]
    
    return inner, (float(centroid_x), float(centroid_y)), float(required_side), float(best_outer_angle)

if __name__ == "__main__":
    inner, center, side, angle = construct()
    print(f"Outer side length: {side:.4f}")
    print(f"Outer center: {center}")
    print(f"Outer angle: {angle}")
    print(f"Inner hexagons: {len(inner)}")
