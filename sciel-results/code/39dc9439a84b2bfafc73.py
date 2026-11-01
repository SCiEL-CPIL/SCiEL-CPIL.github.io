"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Generate hexagonal lattice positions for N unit hexagons
    def hex_count(m):
        return 3 * m * (m + 1) + 1
    
    m = 0
    while hex_count(m) < N:
        m += 1
    
    positions_axial = []
    for q in range(-m, m + 1):
        for r in range(-m, m + 1):
            if abs(q + r) <= m:
                positions_axial.append((q, r))
    
    positions_axial = positions_axial[:N]
    
    size = 1.0
    sqrt3 = np.sqrt(3.0)
    positions = []
    for q, r in positions_axial:
        x = size * 1.5 * q
        y = size * sqrt3 * (r + q / 2.0)
        positions.append((x, y))
    
    inner = [[x, y, 0.0] for x, y in positions]
    
    if positions:
        cx = float(np.mean([p[0] for p in positions]))
        cy = float(np.mean([p[1] for p in positions]))
    else:
        cx, cy = 0.0, 0.0
    
    def hexagon_contains(cx, cy, s, theta, px, py):
        dx = px - cx
        dy = py - cy
        ct = np.cos(-theta)
        st = np.sin(-theta)
        rx = dx * ct - dy * st
        ry = dx * st + dy * ct
        apothem = s * np.sqrt(3) / 2
        for i in range(6):
            angle = np.pi / 3 * i
            nx = np.cos(angle)
            ny = np.sin(angle)
            proj = rx * nx + ry * ny
            if proj > apothem:
                return False
        return True
    
    def inner_hex_vertices(x, y, theta_deg):
        theta = np.radians(theta_deg)
        verts = []
        for i in range(6):
            angle = theta + np.pi / 3 * i
            vx = x + np.cos(angle)
            vy = y + np.sin(angle)
            verts.append((vx, vy))
        return verts
    
    best_side = float('inf')
    best_angle = 0.0
    
    for angle_deg in [0, 30, 60, 90, 120, 150]:
        angle = np.radians(angle_deg)
        lo = 0.0
        hi = 50.0
        for _ in range(60):
            mid = (lo + hi) / 2
            valid = True
            for (px, py, ptheta) in inner:
                verts = inner_hex_vertices(px, py, ptheta)
                for (vx, vy) in verts:
                    if not hexagon_contains(cx, cy, mid, angle, vx, vy):
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                hi = mid
            else:
                lo = mid
        if hi < best_side:
            best_side = hi
            best_angle = angle_deg
    
    return inner, (cx, cy), best_side, best_angle
# EVOLVE-BLOCK-END
