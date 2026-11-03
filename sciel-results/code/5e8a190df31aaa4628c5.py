"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 12
# Deadline guard: aim to finish well before 150s
MAX_TIME = 145.0

def hex_vertices(side, angle_deg):
    """Return 6 vertices of a regular hexagon with given side and rotation."""
    # Regular hexagon vertices at angles 0, 60, 120, 180, 240, 300 degrees
    # But we want the flat-top or pointy-top? Let's use standard: vertices at angles
    # 0, 60, 120, 180, 240, 300 from center, radius = side (circumradius = side for hexagon)
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300])) + np.deg2rad(angle_deg)
    x = side * np.cos(angles)
    y = side * np.sin(angles)
    return np.column_stack([x, y])

def polygon_area(vertices):
    """Shoelace formula for polygon area."""
    x = vertices[:, 0]
    y = vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def point_in_polygon(point, vertices):
    """Check if a point is inside a convex polygon using cross products."""
    x0, y0 = point
    n = len(vertices)
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        cross = (x2 - x1) * (y0 - y1) - (y2 - y1) * (x0 - x1)
        if cross < 0:
            return False
    return True

def polygons_overlap(v1, v2, eps=1e-7):
    """Check if two convex polygons overlap using SAT (Separating Axis Theorem).
    Returns True if they overlap (including touching with tolerance)."""
    # For each edge of both polygons, check if there's a separating axis
    def project(vertices, axis):
        dots = vertices @ axis
        return dots.min(), dots.max()
    
    for v in [v1, v2]:
        for i in range(len(v)):
            p1 = v[i]
            p2 = v[(i + 1) % len(v)]
            edge = p2 - p1
            # Normal to edge
            normal = np.array([-edge[1], edge[0]])
            normal /= (np.linalg.norm(normal) + 1e-12)
            
            min1, max1 = project(v1, normal)
            min2, max2 = project(v2, normal)
            
            # Check for gap
            if max1 < min2 - eps or max2 < min1 - eps:
                return False
    
    return True

def hex_in_hex(inner_verts, outer_center, outer_side, outer_angle):
    """Check if inner hexagon (verts) is fully inside outer hexagon."""
    # Outer hexagon vertices
    outer_verts = hex_vertices(outer_side, outer_angle) + outer_center
    # Check each vertex of inner is inside outer
    for v in inner_verts:
        if not point_in_polygon(v, outer_verts):
            return False
    return True

def total_overlap_penalty(inner_positions, inner_angles, outer_center, outer_side, outer_angle, w_overlap=10.0, w_contain=10.0):
    """Compute penalty for overlaps and containment violations."""
    penalty = 0.0
    n = len(inner_positions)
    
    # Get all inner hexagon vertices
    inner_verts = []
    for i in range(n):
        cx, cy = inner_positions[i]
        ang = inner_angles[i]
        verts = hex_vertices(1.0, ang) + np.array([cx, cy])
        inner_verts.append(verts)
    
    # Check pairwise overlaps
    for i in range(n):
        for j in range(i + 1, n):
            if polygons_overlap(inner_verts[i], inner_verts[j]):
                penalty += w_overlap
    
    # Check containment in outer hexagon
    outer_verts = hex_vertices(outer_side, outer_angle) + outer_center
    for i in range(n):
        for v in inner_verts[i]:
            if not point_in_polygon(v, outer_verts):
                penalty += w_contain
                break
    
    return penalty

def construct():
    start_time = time.time()
    
    # Initial configuration: hexagonal packing
    # Place 12 hexagons in a hexagonal pattern around center
    # For n=12, we can use a hexagonal arrangement: center + 6 around + 6 more
    # Actually, for 12 unit hexagons, a good initial guess is a 3x4 grid or hexagonal
    
    # Let's use a hexagonal lattice arrangement
    # Hexagonal lattice: positions at (i * 3/2, j * sqrt(3) + (i % 2) * sqrt(3)/2)
    # But we need exactly 12
    
    # Simple initial: circular arrangement
    initial_positions = []
    initial_angles = [0.0] * N
    
    # Place 6 in a ring of radius 2, 6 in a ring of radius 4, or similar
    # Better: use a known good packing
    # For 12 unit hexagons, try a 3x4 rectangular grid with hexagonal offset
    
    # Let's use a hexagonal cluster: 1 center, 6 first ring, 5 from second ring
    # Actually, let's just use a simple grid and let optimization fix it
    
    # Grid arrangement
    cols = 4
    rows = 3
    for r in range(rows):
        for c in range(cols):
            x = c * 1.75  # hex spacing
            y = r * 1.5 + (c % 2) * 0.75
            initial_positions.append([x, y])
    
    # Center the configuration
    pos = np.array(initial_positions, dtype=float)
    pos -= pos.mean(axis=0)
    
    best_score = float('inf')
    best_result = None
    
    # Use a simple local search / gradient-free optimization
    # Since the problem is non-differentiable, use simulated annealing or random restarts
    
    np.random.seed(42)
    
    # Try multiple random restarts with local optimization
    n_restarts = 20
    
    for restart in range(n_restarts):
        if time.time() - start_time > MAX_TIME - 5:
            break
        
        # Initialize with perturbation
        positions = pos.copy()
        angles = np.zeros(N)
        
        # Add random perturbation
        if restart > 0:
            positions += np.random.randn(N, 2) * 0.5
            angles = np.random.uniform(0, 60, N)
        
        # Local search with random steps
        step_size = 0.5
        current_penalty = total_overlap_penalty(positions, angles, np.array([0.0, 0.0]), 5.0, 0.0)
        
        # Binary search on outer side length
        # First, find a feasible outer side length
        low_side = 3.0
        high_side = 10.0
        
        # Check if current config fits in high_side
        for _ in range(100):
            if time.time() - start_time > MAX_TIME - 5:
                break
            
            # Find minimal outer side for this configuration
            # Use bisection
            lo = low_side
            hi = high_side
            found = False
            for _ in range(30):
                mid = (lo + hi) / 2
                # Check if all inner hexagons fit in outer hexagon of side mid
                feasible = True
                for i in range(N):
                    cx, cy = positions[i]
                    ang = angles[i]
                    verts = hex_vertices(1.0, ang) + np.array([cx, cy])
                    outer_verts = hex_vertices(mid, 0.0)
                    for v in verts:
                        if not point_in_polygon(v, outer_verts):
                            feasible = False
                            break
                    if not feasible:
                        break
                
                if feasible:
                    hi = mid
                    found = True
                else:
                    lo = mid
            
            if found:
                # Try to improve by local search
                improved = False
                for _ in range(200):
                    if time.time() - start_time > MAX_TIME - 5:
                        break
                    
                    # Random perturbation
                    idx = np.random.randint(N)
                    delta = np.random.randn(2) * step_size
                    old_pos = positions[idx].copy()
                    positions[idx] += delta
                    
                    # Check if still feasible
                    feasible = True
                    for i in range(N):
                        cx, cy = positions[i]
                        ang = angles[i]
                        verts = hex_vertices(1.0, ang) + np.array([cx, cy])
                        outer_verts = hex_vertices(hi, 0.0)
                        for v in verts:
                            if not point_in_polygon(v, outer_verts):
                                feasible = False
                                break
                        if not feasible:
                            break
                    
                    # Check overlaps
                    if feasible:
                        for i in range(N):
                            for j in range(i + 1, N):
                                v1 = hex_vertices(1.0, angles[i]) + positions[i]
                                v2 = hex_vertices(1.0, angles[j]) + positions[j]
                                if polygons_overlap(v1, v2):
                                    feasible = False
                                    break
                            if not feasible:
                                break
                    
                    if feasible:
                        improved = True
                    else:
                        positions[idx] = old_pos
                
                if not improved:
                    break
            else:
                # Need larger outer hexagon
                low_side = hi
                high_side = hi * 1.5
                if high_side > 20:
                    break
        
        # Record best
        if found and hi < best_score:
            best_score = hi
            best_positions = positions.copy()
            best_angles = angles.copy()
            best_outer_center = (0.0, 0.0)
            best_outer_angle = 0.0
    
    # If no feasible solution found, return a safe default
    if best_result is None:
        if best_positions is not None and best_score < float('inf'):
            inner = [[best_positions[i][0], best_positions[i][1], best_angles[i]] for i in range(N)]
            return inner, (0.0, 0.0), best_score, 0.0
        else:
            # Fallback: large hexagon with hexagons in a grid
            inner = [[i * 2.0, (j % 2) * 1.0, 0.0] for j in range(3) for i in range(4)]
            return inner, (0.0, 0.0), 15.0, 0.0

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
