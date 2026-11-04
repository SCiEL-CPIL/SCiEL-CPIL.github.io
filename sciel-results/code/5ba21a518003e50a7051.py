"""Packing 12 unit hexagons in a regular hexagon using optimization."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12
START_TIME = time.time()
TIME_LIMIT = 130.0  # safety margin before 150s kill

def hexagon_vertices(cx, cy, s, angle_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side s, rotation angle_deg."""
    angle_rad = np.radians(angle_deg)
    # Flat-topped hexagon: vertices at angles 0, 60, 120, 180, 240, 300 degrees
    angles = np.array([0, 60, 120, 180, 240, 300]) + angle_deg
    angles_rad = np.radians(angles)
    return cx + s * np.cos(angles_rad), cy + s * np.sin(angles_rad)

def hexagon_support(cx, cy, s, angle_deg, directions):
    """Return support function values for a hexagon in given directions (unit vectors)."""
    x, y = hexagon_vertices(cx, cy, s, angle_deg)
    verts = np.column_stack([x, y])
    # Support in direction d = max dot product
    supports = np.max(verts @ directions, axis=0)
    return supports

def check_feasible(inner, center, side, angle_deg, tol=1e-9):
    """Check if all inner hexagons fit inside the outer hexagon."""
    cx, cy = center
    # Outer hexagon vertices
    ox, oy = hexagon_vertices(cx, cy, side, angle_deg)
    outer_verts = np.column_stack([ox, oy])
    
    for (ix, iy, iangle) in inner:
        ix_v, iy_v = hexagon_vertices(ix, iy, 1.0, iangle)
        inner_verts = np.column_stack([ix_v, iy_v])
        # Check if all inner vertices are inside outer hexagon
        # Use winding number / point-in-polygon
        for vx, vy in inner_verts:
            # Point in convex polygon check
            inside = True
            for i in range(6):
                x1, y1 = outer_verts[i]
                x2, y2 = outer_verts[(i+1) % 6]
                # Cross product to check same side
                cross = (x2-x1)*(vy-y1) - (y2-y1)*(vx-x1)
                if cross < -tol:
                    inside = False
                    break
            if not inside:
                return False
    return True

def compute_outer_side(inner, center, angle_deg):
    """Compute minimum outer hexagon side length to contain all inner hexagons."""
    cx, cy = center
    # The outer hexagon must contain all vertices of all inner hexagons
    # For a regular hexagon with given center and angle, find minimum side
    # such that all points are inside
    
    # Collect all vertices
    all_verts = []
    for (ix, iy, iangle) in inner:
        vx, vy = hexagon_vertices(ix, iy, 1.0, iangle)
        all_verts.append(np.column_stack([vx, vy]))
    all_verts = np.vstack(all_verts)
    
    # For a regular hexagon with center (cx,cy) and angle, the constraint is:
    # For each of 6 edge normals, the max projection of any point must be <= side * sqrt(3)/2
    # Actually, for a regular hexagon with side s, the apothem (distance from center to edge) = s * sqrt(3)/2
    
    angle_rad = np.radians(angle_deg)
    # Edge normals for flat-topped hexagon: perpendicular to edges
    # Edges are between consecutive vertices, normals point outward
    # For flat-topped: normals at 30, 90, 150, 210, 270, 330 degrees (for angle=0)
    normal_angles = np.array([30, 90, 150, 210, 270, 330]) + angle_deg
    normals = np.column_stack([np.cos(np.radians(normal_angles)), np.sin(np.radians(normal_angles))])
    
    # For each normal direction, compute max projection of all vertices relative to center
    rel = all_verts - np.array([cx, cy])
    projections = rel @ normals  # shape (num_verts, 6)
    max_proj = np.max(projections, axis=0)  # max in each direction
    
    # The apothem must be >= max_proj, and apothem = side * sqrt(3)/2
    # So side >= max_proj / (sqrt(3)/2) = 2 * max_proj / sqrt(3)
    required_side = 2.0 * np.max(max_proj) / np.sqrt(3.0)
    
    return required_side

def objective(params):
    """Objective: minimize outer hexagon side length."""
    # params: x0,y0,ang0, x1,y1,ang1, ..., x11,y11,ang11
    inner = []
    for i in range(N):
        x, y, ang = params[3*i], params[3*i+1], params[3*i+2]
        inner.append([x, y, ang])
    
    # Center at origin, angle 0 for simplicity (we can fix outer to reduce variables)
    center = (0.0, 0.0)
    angle_deg = 0.0
    
    side = compute_outer_side(inner, center, angle_deg)
    
    # Add penalty for overlapping inner hexagons
    # For each pair, check minimum distance between hexagons
    overlap_penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            xi, yi, ai = inner[i]
            xj, yj, aj = inner[j]
            # Approximate: check if centers are too close
            dx = xi - xj
            dy = yi - yj
            dist = np.sqrt(dx*dx + dy*dy)
            # Minimum center distance for non-overlap of two unit hexagons
            # depends on relative orientation; use conservative estimate
            # For same orientation, min distance is 2*apothem = sqrt(3)
            # For different orientations, could be less, but let's use sqrt(3) as lower bound
            min_dist = np.sqrt(3.0)
            if dist < min_dist:
                overlap_penalty += (min_dist - dist) ** 2
    
    return side + 10.0 * overlap_penalty

def construct():
    # Initial hexagonal lattice arrangement
    # Flat-topped hexagons, spacing sqrt(3) between centers
    # Arrange in a compact pattern
    
    # Try a 3-4-3-2 row arrangement or similar
    # For flat-topped hexagons with spacing s = sqrt(3):
    # Row offsets: even rows aligned, odd rows shifted by s/2
    
    s = np.sqrt(3.0)
    
    # Initial positions: hexagonal cluster
    # Let's use a simple arrangement:
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons  
    # Row 2: 3 hexagons
    # Row 3: 2 hexagons
    # Total: 12
    
    positions = []
    # Row 0: y=0, 3 hexagons centered
    for i in range(3):
        positions.append([s * (i - 1), 0.0, 0.0])
    # Row 1: y=s*0.75 (since vertical spacing for flat-topped is s*sqrt(3)/2 = 1.5 for s=sqrt(3))
    # Actually for flat-topped hexagons, the vertical distance between rows is s * sqrt(3)/2
    # s = sqrt(3), so vertical spacing = sqrt(3) * sqrt(3)/2 = 3/2 = 1.5
    vy = 1.5
    for i in range(4):
        x_offset = s * (i - 1.5) + s/2  # shifted by half
        positions.append([x_offset, vy, 0.0])
    # Row 2: y=2*vy
    for i in range(3):
        positions.append([s * (i - 1), 2*vy, 0.0])
    # Row 3: y=3*vy
    for i in range(2):
        x_offset = s * (i - 0.5) + s/2
        positions.append([x_offset, 3*vy, 0.0])
    
    # Flatten to params
    params0 = []
    for x, y, ang in positions:
        params0.extend([x, y, ang])
    
    # Optimize
    best_params = params0
    best_side = compute_outer_side(positions, (0.0, 0.0), 0.0)
    
    elapsed = time.time() - START_TIME
    if elapsed < TIME_LIMIT - 30:  # leave 30s margin
        try:
            result = minimize(objective, params0, method='L-BFGS-B', 
                            options={'maxiter': 200, 'ftol': 1e-8})
            if result.fun < best_side:
                best_params = result.x
                best_inner = [[best_params[3*i], best_params[3*i+1], best_params[3*i+2]] for i in range(N)]
                best_side = compute_outer_side(best_inner, (0.0, 0.0), 0.0)
        except:
            pass
    
    # Try a few random restarts if time permits
    elapsed = time.time() - START_TIME
    while elapsed < TIME_LIMIT - 20:
        # Random perturbation
        np.random.seed(int(elapsed * 1000) % 10000)
        perturbed = [p + np.random.normal(0, 0.1) for p in params0]
        
        try:
            result = minimize(objective, perturbed, method='L-BFGS-B',
                            options={'maxiter': 100, 'ftol': 1e-7})
            if result.fun < best_side:
                best_params = result.x
                best_inner = [[best_params[3*i], best_params[3*i+1], best_params[3*i+2]] for i in range(N)]
                best_side = compute_outer_side(best_inner, (0.0, 0.0), 0.0)
        except:
            pass
        
        elapsed = time.time() - START_TIME
    
    # Construct final answer
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        ang = best_params[3*i+2]
        inner.append([x, y, ang])
    
    center = (0.0, 0.0)
    side = best_side
    angle_deg = 0.0
    
    # Verify feasibility
    if not check_feasible(inner, center, side, angle_deg):
        # Increase side slightly
        side *= 1.01
    
    return inner, center, side, angle_deg
