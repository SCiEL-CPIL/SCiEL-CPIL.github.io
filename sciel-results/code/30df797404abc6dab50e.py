import numpy as np
from scipy.optimize import minimize
import time

N = 11

def hexagon_vertices(cx, cy, side, angle_deg=0.0):
    """Return vertices of a regular hexagon with given center, side length, and rotation."""
    angle_rad = np.deg2rad(angle_deg)
    vertices = []
    for k in range(6):
        theta = angle_rad + k * np.pi / 3.0
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        vertices.append([x, y])
    return np.array(vertices)

def point_in_hexagon(px, py, cx, cy, side, angle_deg=0.0):
    """Check if point (px,py) is inside hexagon."""
    # Transform to hexagon's local frame
    dx = px - cx
    dy = py - cy
    angle_rad = np.deg2rad(angle_deg)
    # Rotate point by -angle
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    # For a unit hexagon with vertex at angle 0, the constraints are:
    # |lx| <= side, |ly| <= side * sqrt(3)/2, |lx|/2 + |ly|/sqrt(3) <= side
    # Actually, for a regular hexagon with circumradius R=side:
    # The hexagon is the intersection of 6 half-planes
    # For vertex at angle 0: vertices at (R,0), (R/2, R*sqrt(3)/2), etc.
    # The half-plane constraints: for each edge, the normal points outward
    # Edge between vertex k and k+1: the outward normal direction
    # For a hexagon with circumradius R, inradius = R*sqrt(3)/2
    # The 6 outward normals are at angles 30°, 90°, 150°, 210°, 270°, 330° (if vertex at 0°)
    # Actually, let me just check: the hexagon with vertices at angles 0,60,120,180,240,300
    # The edges are between consecutive vertices. The outward normal to edge from vertex at angle a to vertex at angle a+60 is at angle a+30.
    # The distance from center to each edge is R*sqrt(3)/2
    # So the constraint is: for each of 6 directions at angles 30+60k, the projection <= R*sqrt(3)/2
    inradius = side * np.sqrt(3) / 2.0
    for k in range(6):
        normal_angle = np.deg2rad(30 + 60 * k)
        nx = np.cos(normal_angle)
        ny = np.sin(normal_angle)
        if lx * nx + ly * ny > inradius:
            return False
    return True

def max_projection_for_hexagon(points, angle_deg=0.0):
    """For a set of points, find the minimum side length of a regular hexagon 
    with given rotation angle that contains all points, centered optimally."""
    # For a regular hexagon with circumradius R and rotation angle, 
    # the constraint is that for each of 6 edge normals, the max projection <= R*sqrt(3)/2
    # The hexagon center can be chosen to minimize R.
    # For a given rotation, the optimal center is the midpoint of the bounding box in each pair of opposite directions.
    # The minimum R is half the maximum of the widths in the 6 normal directions, divided by sqrt(3)/2.
    
    angle_rad = np.deg2rad(angle_deg)
    normals = []
    for k in range(6):
        normal_angle = angle_rad + np.deg2rad(30 + 60 * k)
        normals.append(np.array([np.cos(normal_angle), np.sin(normal_angle)]))
    
    # For each normal direction, compute min and max projection
    # The width in that direction is max_proj - min_proj
    # The hexagon needs inradius >= max(width/2) over all 6 directions
    # inradius = R * sqrt(3)/2, so R = inradius / (sqrt(3)/2)
    
    max_half_width = 0.0
    for n in normals:
        projections = points @ n
        width = np.max(projections) - np.min(projections)
        half_width = width / 2.0
        if half_width > max_half_width:
            max_half_width = half_width
    
    inradius_needed = max_half_width
    R = inradius_needed / (np.sqrt(3) / 2.0)
    
    # Find the center: midpoint of the projections in the tightest direction
    # Actually, we need to find a center such that all projections are within [-R*sqrt(3)/2, R*sqrt(3)/2]
    # The optimal center is the average of the midpoints, but it's a feasibility problem.
    # For a regular hexagon, if we center it at the centroid of the projections' midpoints, it should work.
    # Let's just return R; the center can be computed as the mean of the midpoints.
    
    return R

def compute_cluster_points(positions, cluster_angle_deg):
    """Rotate and translate cluster points to global frame."""
    angle_rad = np.deg2rad(cluster_angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    return positions @ rot.T

def objective(params, base_positions):
    """Objective: minimize the enclosing hexagon side length.
    params = [cluster_angle_deg, container_angle_deg]
    """
    cluster_angle = params[0]
    container_angle = params[1]
    
    # Get cluster points in global frame
    cluster_points = compute_cluster_points(base_positions, cluster_angle)
    
    # Find minimum side length of container hexagon with given rotation
    R = max_projection_for_hexagon(cluster_points, container_angle)
    
    return R

def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120 second deadline
    
    # Define base positions for 11 hexagons in a compact hexagonal arrangement
    # Use a hexagonal lattice: center at (0,0), 6 around it, 4 more
    # For same-orientation unit hexagons, use spacing of 2.0 to ensure no overlap
    # Hexagonal lattice: positions at (i*dx + j*dx/2, j*dy) where dx=2, dy=2*sqrt(3)/2=sqrt(3)
    
    # Actually, let me use a more compact arrangement.
    # Hexagonal numbers: 1, 7, 19... For 11, I'll use a 3-4-3 or 2-3-3-3-2 pattern
    # Let me try: a central hexagon, 6 around it (hexagonal number 7), then 4 more added to make a more compact shape
    
    # Simple approach: use a grid-like hexagonal arrangement
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons  
    # Row 2: 4 hexagons
    # That's 11. Or:
    # Row 0: 2
    # Row 1: 3
    # Row 2: 3
    # Row 3: 3
    # That's 11.
    
    # Let me use a hexagonal lattice with spacing 2.0
    dx = 2.0  # horizontal spacing
    dy = np.sqrt(3)  # vertical spacing for hexagonal lattice
    
    # Position 11 hexagons in a compact cluster
    # Using axial coordinates converted to Cartesian
    # For a hex grid, position (q, r) maps to (q*dx + r*dx/2, r*dy)
    
    # Let me try a 3-4-3 arrangement (rows of 3, 4, 3)
    positions = []
    # Row 0 (top): 3 hexagons
    for q in range(-1, 2):
        positions.append([q * dx, 0.0])
    # Row 1 (middle): 4 hexagons  
    for q in range(-2, 2):
        positions.append([q * dx + dx/2, dy])
    # Row 2 (bottom): 3 hexagons
    for q in range(-1, 2):
        positions.append([q * dx, 2 * dy])
    
    base_positions = np.array(positions, dtype=float)
    
    # Center the cluster
    centroid = np.mean(base_positions, axis=0)
    base_positions = base_positions - centroid
    
    # Optimize cluster rotation and container rotation
    # Try multiple initial angles
    
    best_result = None
    best_R = float('inf')
    
    # Initial guesses: try various angles
    initial_angles = [0, 30, 60, 90, 15, 45, 75]
    
    for init_ca in initial_angles:
        if time.time() > deadline - 30:
            break
        
        x0 = np.array([float(init_ca), float(init_ca)])
        
        try:
            result = minimize(objective, x0, args=(base_positions,), 
                            method='Nelder-Mead',
                            options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8})
            
            if result.fun < best_R:
                best_R = result.fun
                best_result = result.x.copy()
                
        except Exception:
            pass
    
    # Refine with L-BFGS-B if we have a good result
    if best_result is not None:
        if time.time() < deadline - 20:
            try:
                result2 = minimize(objective, best_result, args=(base_positions,),
                                  method='L-BFGS-B',
                                  options={'maxiter': 500, 'ftol': 1e-10})
                if result2.fun < best_R:
                    best_R = result2.fun
                    best_result = result2.x.copy()
            except Exception:
                pass
    
    if best_result is None:
        # Fallback: use initial guess
        best_result = np.array([0.0, 0.0])
        best_R = objective(best_result, base_positions)
    
    cluster_angle = best_result[0]
    container_angle = best_result[1]
    
    # Compute final positions
    final_positions = compute_cluster_points(base_positions, cluster_angle)
    
    # Find the center of the enclosing hexagon
    # The center should be at the midpoint of the projections in the tightest direction
    angle_rad = np.deg2rad(container_angle)
    normals = []
    for k in range(6):
        normal_angle = angle_rad + np.deg2rad(30 + 60 * k)
        normals.append(np.array([np.cos(normal_angle), np.sin(normal_angle)]))
    
    # Find center as average of midpoints
    center = np.zeros(2)
    for n in normals:
        projections = final_positions @ n
        mid = (np.max(projections) + np.min(projections)) / 2.0
        center += mid * n
    center /= len(normals)
    
    # Compute the actual side length
    R = max_projection_for_hexagon(final_positions, container_angle)
    
    # Create inner hexagon list
    inner = []
    for i in range(N):
        x, y = final_positions[i]
        inner.append([float(x), float(y), 0.0])  # All same orientation
    
    return inner, (float(center[0]), float(center[1])), float(R), float(container_angle)
