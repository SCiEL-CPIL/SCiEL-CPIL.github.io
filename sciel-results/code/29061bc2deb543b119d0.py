"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    d = np.sqrt(3.0)  # center-to-center distance for adjacent unit hexagons
    
    # Compact 3-4-4 hexagonal cluster
    positions = []
    # Row 0 (y=0): 3 hexagons
    for i in range(3):
        positions.append(((i - 1) * d, 0.0))
    # Row 1 (y=1.5): 4 hexagons
    for i in range(4):
        positions.append(((i - 1.5) * d, 1.5))
    # Row 2 (y=3.0): 4 hexagons
    for i in range(4):
        positions.append(((i - 1.5) * d, 3.0))
    
    # Center the pattern
    cx = sum(p[0] for p in positions) / N
    cy = sum(p[1] for p in positions) / N
    positions = [(x - cx, y - cy) for x, y in positions]
    
    # Vertices of a unit hexagon with rotation 0
    hex_verts = [(np.cos(v * np.pi / 3), np.sin(v * np.pi / 3)) for v in range(6)]
    
    best_s = float('inf')
    best_angle = 0.0
    
    for theta_deg in range(0, 30):
        theta = np.radians(theta_deg)
        normals = [(np.cos(theta + k * np.pi / 3), np.sin(theta + k * np.pi / 3)) for k in range(6)]
        
        required_apothem = 0.0
        for px, py in positions:
            for nx, ny in normals:
                # Check both directions
                max_pos = -float('inf')
                max_neg = -float('inf')
                for vx, vy in hex_verts:
                    wx = px + vx
                    wy = py + vy
                    proj = wx * nx + wy * ny
                    max_pos = max(max_pos, proj)
                    max_neg = max(max_neg, -proj)
                required_apothem = max(required_apothem, max_pos, max_neg)
        
        s = required_apothem * 2.0 / np.sqrt(3.0)
        if s < best_s:
            best_s = s
            best_angle = theta_deg
    
    # Add small margin for floating point
    best_s = np.ceil(best_s * 10000) / 10000.0 + 0.0001
    
    inner = [[x, y, 0.0] for x, y in positions]
    return inner, (0.0, 0.0), float(best_s), float(best_angle)
# EVOLVE-BLOCK-END
