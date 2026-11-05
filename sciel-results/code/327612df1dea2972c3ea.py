"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize_scalar

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Place N unit hexagons in a hexagonal grid pattern
    hex_width = np.sqrt(3)   # horizontal distance between adjacent centers (pointy-top)
    hex_v_offset = 1.5       # vertical offset for adjacent rows
    
    # Generate hexagonal grid positions using axial coordinates
    axial_coords = [(0, 0)]
    ring = 1
    while len(axial_coords) < N:
        for q in range(-ring, ring + 1):
            for r in range(-ring, ring + 1):
                if max(abs(q), abs(r), abs(q + r)) == ring:
                    if (q, r) not in axial_coords:
                        axial_coords.append((q, r))
        ring += 1
    
    axial_coords = axial_coords[:N]
    
    # Convert to pixel coordinates (pointy-top hexagons)
    pixel_positions = []
    for q, r in axial_coords:
        x = hex_width * (q + r / 2.0)
        y = hex_v_offset * r
        pixel_positions.append((x, y))
    
    # Center the arrangement at origin
    xs = [p[0] for p in pixel_positions]
    ys = [p[1] for p in pixel_positions]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    
    inner = []
    centered_points = []
    for (x, y) in pixel_positions:
        px = x - cx
        py = y - cy
        inner.append([px, py, 0.0])
        centered_points.append((px, py))
    
    def max_projection(px, py, theta):
        max_proj = 0.0
        for i in range(6):
            angle = theta + i * np.pi / 3.0
            nx = np.cos(angle)
            ny = np.sin(angle)
            proj = abs(px * nx + py * ny)
            if proj > max_proj:
                max_proj = proj
        return max_proj
    
    def compute_required_side(theta):
        max_proj = 0.0
        for (px, py) in centered_points:
            proj = max_projection(px, py, theta)
            if proj > max_proj:
                max_proj = proj
        return 2.0 * max_proj / np.sqrt(3.0)
    
    # Optimize rotation angle
    best_s = float('inf')
    best_theta = 0.0
    for theta in np.linspace(0, np.pi / 6, 37):
        s = compute_required_side(theta)
        if s < best_s:
            best_s = s
            best_theta = theta
    
    result = minimize_scalar(lambda t: compute_required_side(t),
                            bounds=(best_theta - np.pi / 36, best_theta + np.pi / 36),
                            method='bounded')
    if result.success:
        best_theta = result.x
        best_s = compute_required_side(best_theta)
    
    return inner, (0.0, 0.0), best_s, best_theta * 180.0 / np.pi
# EVOLVE-BLOCK-END

