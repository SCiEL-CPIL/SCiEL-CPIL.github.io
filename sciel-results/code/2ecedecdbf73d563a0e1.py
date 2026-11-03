"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START

def _hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx, cy) with given side length and rotation."""
    angle = np.radians(angle_deg)
    verts = []
    for i in range(6):
        a = angle + i * np.pi / 3.0
        verts.append((cx + side * np.cos(a), cy + side * np.sin(a)))
    return np.array(verts)

def _poly_overlap_area(v1, v2):
    """Compute area of intersection of two convex polygons using Sutherland-Hodgman clipping."""
    def clip(subject, clipper):
        output = list(subject)
        for i in range(len(clipper)):
            if not output:
                break
            input_list = output
            output = []
            start = clipper[i]
            end = clipper[(i + 1) % len(clipper)]
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            
            def inside(p):
                return (dx * (p[1] - start[1]) - dy * (p[0] - start[0])) >= -1e-10
            
            def intersect(p1, p2):
                # Line p1-p2 with line start-end
                x1, y1 = p1
                x2, y2 = p2
                x3, y3 = start
                x4, y4 = end
                denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if abs(denom) < 1e-12:
                    return p1
                t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
                return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
            
            for j in range(len(input_list)):
                cur = input_list[j]
                prev = input_list[j - 1]
                if inside(cur):
                    if not inside(prev):
                        output.append(intersect(prev, cur))
                    output.append(cur)
                elif inside(prev):
                    output.append(intersect(prev, cur))
        return output
    
    if len(v1) < 3 or len(v2) < 3:
        return 0.0
    
    clipped = clip(list(v1), list(v2))
    if len(clipped) < 3:
        return 0.0
    
    # Shoelace formula
    area = 0.0
    n = len(clipped)
    for i in range(n):
        x1, y1 = clipped[i]
        x2, y2 = clipped[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0

def _penalty(x, outer_side, outer_center, outer_angle):
    """Compute penalty based on overlaps and containment."""
    cx, cy = outer_center
    inner_hexes = []
    for i in range(N):
        ix, iy, iangle = x[3*i], x[3*i+1], x[3*i+2]
        inner_hexes.append(_hex_vertices(ix, iy, 1.0, iangle))
    
    outer_hex = _hex_vertices(cx, cy, outer_side, outer_angle)
    
    penalty = 0.0
    
    # Check containment: each inner hex must be inside outer hex
    # Approximate by checking all vertices of inner are inside outer
    # and all vertices of outer are outside inner (or use overlap)
    for i in range(N):
        # Overlap between inner i and outer should be area of inner (if contained)
        # If not contained, overlap < area of inner
        area_inner = _poly_overlap_area(inner_hexes[i], outer_hex)
        expected_area = np.sqrt(3) * 1.5  # area of unit hexagon
        if area_inner < expected_area - 1e-6:
            penalty += (expected_area - area_inner) ** 2
    
    # Check pairwise disjointness
    for i in range(N):
        for j in range(i+1, N):
            overlap = _poly_overlap_area(inner_hexes[i], inner_hexes[j])
            if overlap > 1e-6:
                penalty += overlap ** 2
    
    return penalty

def construct():
    # Initialize with a reasonable configuration
    # Place hexagons in a hexagonal pattern
    # For 12 hexagons, we can use a pattern with some symmetry
    
    # Let's try a configuration where we place them around a center
    # or in two rows
    
    # Initial guess: place in a grid-like pattern
    x0 = np.zeros(3 * N)
    
    # Place in a hexagonal arrangement
    # Center at (0,0), place 12 around
    # Actually, let's try a simpler initial config: two rows of 6
    positions = []
    for row in range(2):
        for col in range(6):
            # Hex grid spacing
            x_pos = col * 2.0 + (row % 2) * 1.0
            y_pos = row * np.sqrt(3)
            positions.append((x_pos - 5.0, y_pos - np.sqrt(3)/2, 0.0))
    
    for i, (px, py, pa) in enumerate(positions):
        x0[3*i] = px
        x0[3*i+1] = py
        x0[3*i+2] = pa
    
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    outer_side = 8.0  # initial guess
    
    # Optimize outer side length and inner positions
    def objective(params):
        # params: [inner positions (3N), outer_side]
        inner_params = params[:3*N]
        os = params[3*N]
        
        if os < 1.0:
            return 1e10
        
        return os + 100.0 * _penalty(inner_params, os, outer_center, outer_angle)
    
    initial_params = np.concatenate([x0, [outer_side]])
    
    # Use Nelder-Mead or BFGS
    # Nelder-Mead is more robust for this non-smooth problem
    result = minimize(
        objective,
        initial_params,
        method='Nelder-Mead',
        options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6, 'disp': False}
    )
    
    if result.success or result.fun < 1e10:
        inner_params = result.x[:3*N]
        outer_side_opt = result.x[3*N]
        
        inner = []
        for i in range(N):
            inner.append([inner_params[3*i], inner_params[3*i+1], inner_params[3*i+2]])
        
        return inner, outer_center, max(outer_side_opt, 1.0), outer_angle
    else:
        # Fallback to initial
        inner = [[positions[i][0], positions[i][1], positions[i][2]] for i in range(N)]
        return inner, outer_center, outer_side, outer_angle

# EVOLVE-BLOCK-END
