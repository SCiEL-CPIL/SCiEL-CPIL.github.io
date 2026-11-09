"""
Combined approach:
1. Structural idea from Inspiration 1: Use smooth, differentiable overlap penalties (SAT-based) and
   scipy.optimize.minimize (L-BFGS-B) to find tight packings. This is much more effective than
   discrete simulated annealing for this continuous problem.
2. Refinement machinery from Current Program:
   - Robust validity checking (though we rely on the optimizer to find valid states, we verify at the end).
   - Multi-start strategy with diverse topologies (Lattice, Random, Circular).
   - Explicit deadline handling.

Key improvements:
- The objective function minimizes the outer side length directly, using a smooth penalty for overlaps
  and containment violations. This allows the gradient-based optimizer to navigate the configuration space.
- We define the outer hexagon center and rotation as variables to be optimized as well, allowing the
  inner hexagons to be arranged optimally relative to the container.
"""

import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11
DEADLINE = 150.0
START_TIME = time.time()

def _remaining_time():
    return DEADLINE - (time.time() - START_TIME)

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with side length 'side'."""
    angle_rad = np.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + k * np.pi / 3
        verts.append([cx + side * np.cos(a), cy + side * np.sin(a)])
    return np.array(verts)

def _point_in_hex(px, py, cx, cy, side, angle_deg, tol=1e-9):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    dx = px - cx
    dy = py - cy
    angle_rad = np.radians(angle_deg)
    ca = np.cos(-angle_rad)
    sa = np.sin(-angle_rad)
    lx = dx * ca - dy * sa
    ly = dx * sa + dy * ca
    
    inradius = side * np.sqrt(3) / 2
    for k in range(6):
        a = np.radians(30 + k * 60)
        nx = np.cos(a)
        ny = np.sin(a)
        if lx * nx + ly * ny > inradius + tol:
            return False
    return True

def _hexagons_disjoint(verts1, verts2, tol=1e-7):
    """Check if two convex hexagons are disjoint using SAT."""
    n1 = len(verts1)
    n2 = len(verts2)
    
    def get_normals(verts):
        normals = []
        n = len(verts)
        for i in range(n):
            j = (i + 1) % n
            edge = verts[j] - verts[i]
            nx = -edge[1]
            ny = edge[0]
            norm = np.sqrt(nx*nx + ny*ny)
            if norm > 1e-12:
                normals.append([nx/norm, ny/norm])
        return normals
    
    normals1 = get_normals(verts1)
    normals2 = get_normals(verts2)
    all_normals = normals1 + normals2
    
    for nx, ny in all_normals:
        proj1 = np.array([v[0]*nx + v[1]*ny for v in verts1])
        proj2 = np.array([v[0]*nx + v[1]*ny for v in verts2])
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 - tol or max2 < min1 - tol:
            return True
    
    return False

def _min_enclosing_side(inner_positions, outer_center, outer_angle):
    """Find minimum side length of outer hexagon to contain all inner hexagons."""
    cx, cy = outer_center
    best_side = 0.0
    
    for (ix, iy, iangle) in inner_positions:
        iverts = _hex_vertices(ix, iy, 1.0, iangle)
        for v in iverts:
            dx = v[0] - cx
            dy = v[1] - cy
            angle_rad = np.radians(outer_angle)
            ca = np.cos(-angle_rad)
            sa = np.sin(-angle_rad)
            lx = dx * ca - dy * sa
            ly = dx * sa + dy * ca
            
            r = np.sqrt(lx*lx + ly*ly)
            if r < 1e-12:
                continue
            
            theta = np.arctan2(ly, lx)
            best_cos = 0.0
            for k in range(6):
                a = np.radians(30 + k * 60)
                d = abs(((theta - a) % (2*np.pi) + np.pi) % (2*np.pi))
                d = min(d, 2*np.pi - d)
                c = np.cos(d)
                if c > best_cos:
                    best_cos = c
            
            if best_cos < 1e-12:
                best_side = max(best_side, 1000.0)
            else:
                required_side = 2.0 * r * best_cos / np.sqrt(3)
                best_side = max(best_side, required_side)
    
    return best_side

def _check_validity(inner_positions, outer_center, outer_side, outer_angle, tol=1e-6):
    """Check if configuration is valid: all inner hexagons inside outer, no overlaps."""
    cx, cy = outer_center
    
    for (ix, iy, iangle) in inner_positions:
        iverts = _hex_vertices(ix, iy, 1.0, iangle)
        for v in iverts:
            if not _point_in_hex(v[0], v[1], cx, cy, outer_side, outer_angle, tol=tol):
                return False
    
    all_verts = [_hex_vertices(ix, iy, 1.0, iangle) for (ix, iy, iangle) in inner_positions]
    for i in range(N):
        for j in range(i+1, N):
            if not _hexagons_disjoint(all_verts[i], all_verts[j], tol=tol):
                return False
    
    return True

def _initial_lattice():
    """Create initial hexagonal lattice arrangement with correct spacing."""
    positions = []
    positions.append([0.0, 0.0, 0.0])
    dist = 2.0 * np.sqrt(3)  # Minimum distance for non-overlapping unit hexagons
    for k in range(6):
        a = np.radians(60 * k)
        positions.append([dist * np.cos(a), dist * np.sin(a), 0.0])
    # Add 4 more hexagons in a second ring
    dist2 = dist * np.sqrt(3)
    angles2 = [30, 90, 150, 210]
    for a_deg in angles2:
        a = np.radians(a_deg)
        positions.append([dist2 * np.cos(a), dist2 * np.sin(a), 0.0])
    
    return positions

def _initial_random():
    """Create a random initial configuration."""
    positions = []
    positions.append([0.0, 0.0, 0.0])
    for i in range(1, N):
        r = random.uniform(1.0, 4.0)
        a = random.uniform(0, 2*np.pi)
        angle = random.uniform(0, 360)
        positions.append([r * np.cos(a), r * np.sin(a), angle])
    return positions

def _initial_circular():
    """Create a circular initial configuration."""
    positions = []
    positions.append([0.0, 0.0, 0.0])
    for i in range(1, N):
        a = 2 * np.pi * i / N
        r = 2.0 * np.sqrt(3)
        positions.append([r * np.cos(a), r * np.sin(a), 0.0])
    return positions

def _objective(params):
    """
    Objective function for optimization.
    params: [x1, y1, a1, ..., xN, yN, aN, cx, cy, cangle]
    Returns: side_length + penalty
    """
    if _remaining_time() < 5.0:
        return 1e10 # Force stop if time is up
    
    # Unpack parameters
    inner = []
    for i in range(N):
        x = params[i*3]
        y = params[i*3 + 1]
        a = params[i*3 + 2]
        inner.append((x, y, a))
    
    cx = params[N*3]
    cy = params[N*3 + 1]
    cangle = params[N*3 + 2]
    
    # Calculate required side length
    side = _min_enclosing_side(inner, (cx, cy), cangle)
    
    # Calculate penalty for overlaps
    penalty = 0.0
    all_verts = [_hex_vertices(x, y, 1.0, a) for (x, y, a) in inner]
    for i in range(N):
        for j in range(i+1, N):
            # Use a simple distance-based penalty for speed in gradient steps
            # SAT is too expensive for every gradient step.
            # We check if they are likely overlapping by checking center distance
            # and then doing a quick SAT if close.
            ci = np.mean(all_verts[i], axis=0)
            cj = np.mean(all_verts[j], axis=0)
            d = np.linalg.norm(ci - cj)
            min_dist = 2.0 * np.sqrt(3) # Approx min center distance for non-overlap
            if d < min_dist:
                # Do a precise check
                if not _hexagons_disjoint(all_verts[i], all_verts[j], tol=1e-8):
                    penalty += (min_dist - d) ** 2 * 100.0
    
    # Penalty for containment violations (if side is too small, points are outside)
    # _min_enclosing_side already accounts for this by increasing side.
    # However, if the geometry is weird, we might need a hard penalty.
    # For now, the side calculation is the primary driver.
    
    return side + penalty

def _run_optimization(initial_positions, outer_angle=0.0, max_iters=1000):
    """Run scipy optimization starting from initial_positions."""
    # Flatten initial positions
    params = []
    for p in initial_positions:
        params.extend(p)
    
    # Initial outer center and angle
    cx = 0.0
    cy = 0.0
    params.extend([cx, cy, outer_angle])
    
    # Bounds
    bounds = []
    for i in range(N):
        bounds.append((0.0, 10.0)) # x
        bounds.append((0.0, 10.0)) # y
        bounds.append((0.0, 360.0)) # angle
    
    # Outer bounds
    bounds.append((-5.0, 5.0)) # cx
    bounds.append((-5.0, 5.0)) # cy
    bounds.append((0.0, 60.0)) # cangle (symmetry)
    
    # Center the initial positions to (0,0) roughly
    xs = [p[0] for p in initial_positions]
    ys = [p[1] for p in initial_positions]
    cx_init = sum(xs) / len(xs)
    cy_init = sum(ys) / len(ys)
    
    # Adjust params to be relative to center? No, keep absolute but start near 0.
    # Actually, let's shift the initial positions to be centered at 0,0.
    shifted_positions = []
    for p in initial_positions:
        shifted_positions.append([p[0] - cx_init, p[1] - cy_init, p[2]])
    
    params = []
    for p in shifted_positions:
        params.extend(p)
    params.extend([0.0, 0.0, outer_angle])
    
    try:
        result = minimize(_objective, params, method='L-BFGS-B', bounds=bounds, 
                          options={'maxiter': max_iters, 'maxfun': max_iters*10})
        
        if result.success or result.fun < 100:
            # Extract results
            res_params = result.x
            inner = []
            for i in range(N):
                x = res_params[i*3]
                y = res_params[i*3 + 1]
                a = res_params[i*3 + 2]
                inner.append([x, y, a])
            
            cx = res_params[N*3]
            cy = res_params[N*3 + 1]
            cangle = res_params[N*3 + 2]
            
            side = _min_enclosing_side(inner, (cx, cy), cangle)
            
            # Verify validity
            if _check_validity(inner, (cx, cy), side, cangle, tol=1e-5):
                return inner, (cx, cy), side, cangle
            else:
                # Try to fix by slightly increasing side
                side += 0.01
                if _check_validity(inner, (cx, cy), side, cangle, tol=1e-5):
                    return inner, (cx, cy), side, cangle
                else:
                    return None
        else:
            return None
    except Exception as e:
        return None

def construct():
    start_time = time.time()
    
    # Try multiple initializations
    initializations = [
        _initial_lattice,
        _initial_circular,
        _initial_random
    ]
    
    best_result = None
    best_side = float('inf')
    
    for init_func in initializations:
        if _remaining_time() < 10.0:
            break
        
        init_pos = init_func()
        
        # Try different outer angles
        for angle in [0.0, 30.0]:
            if _remaining_time() < 10.0:
                break
            
            result = _run_optimization(init_pos, outer_angle=angle, max_iters=500)
            
            if result:
                inner, center, side, cangle = result
                if side < best_side:
                    best_side = side
                    best_result = result
    
    # Fallback if no valid result found
    if best_result is None:
        # Simple valid configuration: Line of hexagons
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        center = (0.0, 0.0)
        side = 40.0
        angle = 0.0
        return inner, center, side, angle
    
    inner, center, side, angle = best_result
    
    # Final verification
    if not _check_validity(inner, center, side, angle, tol=1e-5):
        # If invalid, try to expand side slightly
        side += 0.1
        if _check_validity(inner, center, side, angle, tol=1e-5):
            return inner, center, side, angle
        else:
            # Fallback
            inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
            center = (0.0, 0.0)
            side = 40.0
            angle = 0.0
            return inner, center, side, angle
    
    return inner, center, side, angle

# EVOLVE-BLOCK-END
