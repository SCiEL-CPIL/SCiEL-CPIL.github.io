"""Packing 12 unit hexagons in a regular hexagon, minimizing outer side length."""
import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    sqrt3 = np.sqrt(3)
    
    # Initial arrangement: hexagonal grid
    positions = []
    for row in range(3):
        for col in range(4):
            x = col * sqrt3
            y = row * 1.5
            if row % 2 == 1:
                x += sqrt3 / 2
            positions.append([x, y, 0.0])
    
    # Center
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    for p in positions:
        p[0] -= cx
        p[1] -= cy
    
    def compute_outer_side(positions):
        """Compute minimum outer hexagon side length to contain all inner hexagons."""
        inradius_factor = np.sqrt(3) / 2
        max_s = 0
        for (px, py, _) in positions:
            for k in range(6):
                vx = px + np.cos(k * np.pi / 3)
                vy = py + np.sin(k * np.pi / 3)
                max_proj = 0
                for m in range(6):
                    a = np.pi / 6 + m * np.pi / 3
                    val = vx * np.cos(a) + vy * np.sin(a)
                    max_proj = max(max_proj, val)
                s_needed = max_proj / inradius_factor
                max_s = max(max_s, s_needed)
        return max_s
    
    # Simple local optimization: try small perturbations
    best_positions = [p[:] for p in positions]
    best_s = compute_outer_side(positions)
    
    # Try rotating the whole arrangement
    for angle_deg in [0, 30]:
        angle = np.radians(angle_deg)
        rotated = []
        for p in positions:
            x, y = p[0], p[1]
            rx = x * np.cos(angle) - y * np.sin(angle)
            ry = x * np.sin(angle) + y * np.cos(angle)
            rotated.append([rx, ry, 0.0])
        s = compute_outer_side(rotated)
        if s < best_s:
            best_s = s
            best_positions = [p[:] for p in rotated]
    
    return best_positions, (0.0, 0.0), best_s, 0.0
# EVOLVE-BLOCK-END
