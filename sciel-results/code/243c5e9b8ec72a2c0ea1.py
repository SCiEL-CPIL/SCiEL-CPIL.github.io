"""Physics-based packing of 12 unit hexagons in a regular hexagon container."""
import time
import math
import numpy as np

N = 12
DEADLINE = 140.0  # safety margin before 150s kill

def hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    a = math.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = a + k * math.pi / 3.0
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return np.array(verts)

def convex_hull(points):
    """Andrew's monotone chain convex hull."""
    pts = sorted(set(map(tuple, points)))
    if len(pts) <= 1:
        return np.array(pts)
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    return np.array(hull)

def point_in_polygon(x, y, poly):
    """Ray casting point-in-convex-polygon test."""
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def polygon_collision(verts1, verts2, tol=0.001):
    """Check if two convex polygons overlap using SAT."""
    polys = [verts1, verts2]
    for poly in polys:
        for i in range(len(poly)):
            p1 = poly[i]
            p2 = poly[(i+1) % len(poly)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            proj1 = np.dot(verts1, normal)
            proj2 = np.dot(verts2, normal)
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            if max1 < min2 + tol or max2 < min1 + tol:
                return False
    return True

def separation_axis_push(verts1, verts2):
    """Find minimum translation vector to separate two convex polygons."""
    best_overlap = float('inf')
    best_normal = None
    polys = [verts1, verts2]
    for poly in polys:
        for i in range(len(poly)):
            p1 = poly[i]
            p2 = poly[(i+1) % len(poly)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal = normal / norm
            proj1 = np.dot(verts1, normal)
            proj2 = np.dot(verts2, normal)
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            overlap = min(max1, max2) - max(min1, min2)
            if overlap <= 0:
                return None
            if overlap < best_overlap:
                best_overlap = overlap
                # Direction: push verts1 away from verts2
                c1 = verts1.mean(axis=0)
                c2 = verts2.mean(axis=0)
                if np.dot(c1 - c2, normal) < 0:
                    normal = -normal
                best_normal = normal
    return best_normal * best_overlap

def check_validity(inner, center, side, angle_deg):
    """Check if all hexagons are disjoint and inside the container."""
    cx, cy = center
    container_verts = hex_vertices(cx, cy, side, angle_deg)
    hex_verts_list = []
    for (x, y, ang) in inner:
        v = hex_vertices(x, y, 1.0, ang)
        # Check inside container
        for vx, vy in v:
            if not point_in_polygon(vx, vy, container_verts):
                return False
        hex_verts_list.append(v)
    # Check pairwise disjointness
    for i in range(len(hex_verts_list)):
        for j in range(i+1, len(hex_verts_list)):
            if polygon_collision(hex_verts_list[i], hex_verts_list[j]):
                return False
    return True

def compute_outer_side(inner, center, angle_deg):
    """Compute minimum outer side length for given inner positions and container rotation."""
    cx, cy = center
    # Collect all vertices of inner hexagons
    all_verts = []
    for (x, y, ang) in inner:
        v = hex_vertices(x, y, 1.0, ang)
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    # The container is a regular hexagon centered at (cx, cy) with rotation angle_deg
    # For a regular hexagon, a point (px, py) is inside if for each of 6 edge normals,
    # the projection is within the apothem
    # We need to find minimum side such that all points are inside
    # A regular hexagon with side s has apothem s*sqrt(3)/2
    # The 6 edge normals are at angles angle_deg + k*60 deg
    # For point p, distance from center in direction of normal k must be <= apothem
    a = math.radians(angle_deg)
    max_required_apothem = 0.0
    for k in range(6):
        theta = a + k * math.pi / 3.0
        nx, ny = math.cos(theta), math.sin(theta)
        # Project all vertices onto this normal
        rel = all_verts - np.array([cx, cy])
        proj = rel @ np.array([nx, ny])
        max_proj = proj.max()
        max_required_apothem = max(max_required_apothem, max_proj)
    # apothem = side * sqrt(3) / 2, so side = 2 * apothem / sqrt(3)
    return 2.0 * max_required_apothem / math.sqrt(3.0)

def simulate():
    """Run the physics simulation."""
    t0 = time.time()
    # Initial positions: rough grid
    np.random.seed(42)
    positions = []
    angles = []
    # Start with a loose arrangement
    for i in range(N):
        r = 2.0 + 0.5 * i
        theta = 2 * math.pi * i / N
        positions.append([r * math.cos(theta), r * math.sin(theta)])
        angles.append(np.random.uniform(0, 60))
    positions = np.array(positions, dtype=float)
    angles = np.array(angles, dtype=float)
    velocities = np.zeros_like(positions)
    
    # Container parameters
    container_center = np.array([0.0, 0.0])
    container_angle = 0.0
    container_side = 10.0  # start large, shrink over time
    
    dt = 0.005
    max_steps = 50000
    best_side = float('inf')
    best_state = None
    
    for step in range(max_steps):
        if time.time() - t0 > DEADLINE:
            break
        # Shrink container gradually
        target_side = max(3.0, 10.0 - 0.002 * step)
        container_side = max(target_side, container_side - 0.01)
        
        # Compute forces
        forces = np.zeros_like(positions)
        
        # Pairwise repulsion
        for i in range(N):
            vi = hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
            for j in range(i+1, N):
                vj = hex_vertices(positions[j][0], positions[j][1], 1.0, angles[j])
                # Check if close enough to interact
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < 4.0:
                    if polygon_collision(vi, vj):
                        push = separation_axis_push(vi, vj)
                        if push is not None:
                            forces[i] += push * 50.0
                            forces[j] -= push * 50.0
                    else:
                        # Soft repulsion when close
                        if dist < 3.5:
                            direction = positions[i] - positions[j]
                            direction = direction / (dist + 1e-10)
                            forces[i] += direction * (3.5 - dist) * 10.0
                            forces[j] -= direction * (3.5 - dist) * 10.0
        
        # Container wall forces
        container_verts = hex_vertices(container_center[0], container_center[1], container_side, container_angle)
        for i in range(N):
            vi = hex_vertices(positions[i][0], positions[i][1], 1.0, angles[i])
            # Check each vertex against container
            for vx, vy in vi:
                if not point_in_polygon(vx, vy, container_verts):
                    # Push back toward center
                    direction = container_center - positions[i]
                    norm = np.linalg.norm(direction)
                    if norm > 1e-10:
                        direction = direction / norm
                    forces[i] += direction * 100.0
        
        # Damping
        velocities = velocities * 0.95 + forces * dt
        positions = positions + velocities * dt
        
        # Clamp velocities
        vmax = np.max(np.abs(velocities))
        if vmax > 5.0:
            velocities *= 5.0 / vmax
        
        # Every 100 steps, check and record
        if step % 100 == 0:
            inner = [[positions[i][0], positions[i][1], angles[i]] for i in range(N)]
            side = compute_outer_side(inner, container_center, container_angle)
            if side < best_side:
                best_side = side
                best_state = (inner, container_center.copy(), container_angle)
    
    return best_state, best_side

def construct():
    t_start = time.time()
    best_overall = None
    best_side_overall = float('inf')
    
    # Multiple restarts with different random seeds
    for seed in range(20):
        if time.time() - t_start > DEADLINE:
            break
        np.random.seed(seed)
        state, side = simulate()
        if state is not None and side < best_side_overall:
            best_side_overall = side
            best_overall = state
    
    if best_overall is None:
        # Fallback: seed-like solution
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    inner, center, angle = best_overall
    # Verify and compute final side
    side = compute_outer_side(inner, center, angle)
    return inner, (float(center[0]), float(center[1])), float(side), float(angle)
