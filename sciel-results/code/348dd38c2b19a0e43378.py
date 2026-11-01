import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 11

def hex_vertices(cx, cy, s, angle_deg=0.0):
    """Get vertices of regular hexagon centered at (cx,cy) with side s, rotated by angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.pi * np.arange(6) / 3.0
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def point_in_hexagon(px, py, cx, cy, s, angle_deg=0.0):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with side s."""
    # Transform to hexagon's local frame
    dx = px - cx
    dy = py - cy
    a = np.radians(angle_deg)
    # Rotate point back
    cos_a = np.cos(-a)
    sin_a = np.sin(-a)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    # Hexagon with flat top has sides at angles 0, 60, 120, 180, 240, 300
    # The distance from center to each side (apothem) = s * sqrt(3)/2
    apothem = s * np.sqrt(3) / 2
    # Check all 6 sides
    for i in range(6):
        theta = np.pi * i / 3.0
        nx = np.cos(theta)
        ny = np.sin(theta)
        dist = lx * nx + ly * ny
        if dist > apothem:
            return False
    return True

def hexagon_in_hexagon(inner_cx, inner_cy, inner_angle, outer_cx, outer_cy, outer_s, outer_angle):
    """Check if inner unit hexagon is fully inside outer hexagon."""
    # Get inner hexagon vertices
    inner_verts = hex_vertices(inner_cx, inner_cy, 1.0, inner_angle)
    # Check all vertices are inside outer hexagon
    for vx, vy in inner_verts:
        if not point_in_hexagon(vx, vy, outer_cx, outer_cy, outer_s, outer_angle):
            return False
    return True

def containment_penalty(inner, outer_cx, outer_cy, outer_s, outer_angle):
    """Penalty for how much inner hexagons stick out of outer hexagon."""
    penalty = 0.0
    for ix, iy, iang in inner:
        inner_verts = hex_vertices(ix, iy, 1.0, iang)
        for vx, vy in inner_verts:
            dx = vx - outer_cx
            dy = vy - outer_cy
            a = np.radians(outer_angle)
            cos_a = np.cos(-a)
            sin_a = np.sin(-a)
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            apothem = outer_s * np.sqrt(3) / 2
            for i in range(6):
                theta = np.pi * i / 3.0
                nx = np.cos(theta)
                ny = np.sin(theta)
                dist = lx * nx + ly * ny
                if dist > apothem:
                    penalty += (dist - apothem) ** 2
    return penalty

def total_penalty(inner, outer_cx, outer_cy, outer_s, outer_angle):
    """Total penalty: containment + overlap between inner hexagons."""
    pen = containment_penalty(inner, outer_cx, outer_cy, outer_s, outer_angle)
    # Check overlap between inner hexagons
    for i in range(len(inner)):
        for j in range(i+1, len(inner)):
            ix1, iy1, _ = inner[i]
            ix2, iy2, _ = inner[j]
            dist = np.sqrt((ix1-ix2)**2 + (iy1-iy2)**2)
            min_dist = 2.0 * np.sqrt(3) / 2 * 2  # 2 * inradius = sqrt(3) for touching
            # Actually for two unit hexagons to not overlap, center distance >= 2*inradius = sqrt(3)
            if dist < np.sqrt(3) - 1e-9:
                pen += (np.sqrt(3) - dist) ** 2 * 100
    return pen

def objective(params, N):
    """Optimization objective."""
    # params: positions (2*N), angles (N), outer center (2), outer side, outer angle
    pos = params[:2*N].reshape(N, 2)
    angles = params[2*N:3*N]
    outer_cx, outer_cy = params[3*N:3*N+2]
    outer_s = params[3*N+2]
    outer_angle = params[3*N+3]
    
    inner = [list(pos[i]) + [angles[i]] for i in range(N)]
    
    pen = total_penalty(inner, outer_cx, outer_cy, outer_s, outer_angle)
    # We want to minimize outer_s, but with penalty for violations
    # Use: outer_s + large * penalty
    return outer_s + 1000.0 * pen

def generate_hexagonal_lattice(N):
    """Generate hexagonal lattice arrangement for N hexagons."""
    # Try to arrange in hexagonal pattern
    positions = []
    # Hexagonal numbers: 1, 7, 19, 37, ...
    # For N=11, use a roughly hexagonal arrangement
    # Center + 6 around + 4 more
    
    # Simple approach: spiral or grid-like
    hex_spacing = np.sqrt(3)  # distance between centers of touching hexagons
    
    # Generate candidate positions in a hexagonal grid
    candidates = []
    max_r = 3
    for q in range(-max_r, max_r+1):
        for r in range(-max_r, max_r+1):
            x = hex_spacing * (q + r/2.0)
            y = hex_spacing * np.sqrt(3)/2 * r
            candidates.append((x, y))
    
    # Sort by distance from origin
    candidates.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # Take first N
    positions = candidates[:N]
    
    return positions

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    N_local = N
    
    # Generate initial hexagonal lattice
    init_positions = generate_hexagonal_lattice(N_local)
    
    # Initial guess
    init_inner = [[x, y, 0.0] for x, y in init_positions]
    
    # Initial outer hexagon: estimate needed size
    # Find max distance from origin to any inner hex center
    max_r = max(np.sqrt(x**2 + y**2) for x, y in init_positions)
    init_outer_s = max_r + 2.0
    init_outer_angle = 0.0
    
    # Build initial params
    init_params = []
    for x, y in init_positions:
        init_params.extend([x, y])
    for _ in range(N_local):
        init_params.append(0.0)
    init_params.extend([0.0, 0.0, init_outer_s, init_outer_angle])
    
    best_params = np.array(init_params)
    best_score = objective(best_params, N_local)
    
    # Try multiple starting points and local optimizations
    n_starts = 5
    for s in range(n_starts):
        if time.time() > deadline - 20.0:
            break
        
        # Perturb initial positions
        if s == 0:
            params0 = np.array(init_params)
        else:
            params0 = np.array(init_params) + np.random.normal(0, 0.5, size=len(init_params))
        
        # Local optimization using Nelder-Mead
        try:
            result = minimize(
                objective, params0, args=(N_local,),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x
        except Exception:
            pass
    
    # Extract best solution
    pos = best_params[:2*N_local].reshape(N_local, 2)
    angles = best_params[2*N_local:3*N_local]
    outer_cx, outer_cy = best_params[3*N_local:3*N_local+2]
    outer_s = best_params[3*N_local+2]
    outer_angle = best_params[3*N_local+3]
    
    inner = [list(pos[i]) + [angles[i]] for i in range(N_local)]
    
    # Verify and if there are violations, try to fix
    pen = total_penalty(inner, outer_cx, outer_cy, outer_s, outer_angle)
    if pen > 1e-6:
        # Try to expand outer hexagon slightly
        outer_s += pen * 10 + 0.1
    
    # Final check: ensure all hexagons are actually contained
    # If not, expand outer_s until valid
    max_expand = 0
    while max_expand < 50:
        valid = True
        for ix, iy, iang in inner:
            inner_verts = hex_vertices(ix, iy, 1.0, iang)
            for vx, vy in inner_verts:
                if not point_in_hexagon(vx, vy, outer_cx, outer_cy, outer_s, outer_angle):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            break
        outer_s += 0.01
        max_expand += 1
    
    return inner, (float(outer_cx), float(outer_cy)), float(outer_s), float(outer_angle)

