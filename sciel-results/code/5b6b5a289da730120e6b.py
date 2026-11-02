import numpy as np
import time

N = 11

def _hexagon_vertices(cx, cy, side, angle_deg):
    angle_rad = np.radians(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * np.pi / 3
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        vertices.append((x, y))
    return np.array(vertices)

def _min_enclosing_hexagon_side(inner_positions):
    best_L = float('inf')
    best_phi = 0.0
    for phi_deg in np.linspace(0, 60, 241):
        phi = np.radians(phi_deg)
        max_extent = 0.0
        for k in range(6):
            angle = phi + k * np.pi / 3
            nx, ny = np.cos(angle), np.sin(angle)
            max_proj = float('-inf')
            min_proj = float('inf')
            for (cx, cy, ang) in inner_positions:
                verts = _hexagon_vertices(cx, cy, 1.0, ang)
                projs = verts[:, 0] * nx + verts[:, 1] * ny
                max_proj = max(max_proj, float(np.max(projs)))
                min_proj = min(min_proj, float(np.min(projs)))
            extent = max(max_proj, -min_proj)
            max_extent = max(max_extent, extent)
        L = max_extent / (np.sqrt(3) / 2)
        if L < best_L:
            best_L = L
            best_phi = phi_deg
    return best_L, best_phi

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130

    d = np.sqrt(3)
    
    # Generate hex grid positions
    positions = []
    for q in range(-4, 5):
        for r in range(-4, 5):
            if abs(q + r) <= 4:
                x = d * (q + r / 2)
                y = d * (np.sqrt(3) / 2) * r
                positions.append((x, y))
    
    positions_with_dist = sorted([(x*x + y*y, x, y) for (x, y) in positions])
    
    arrangements = []
    
    # Top 11 closest to origin
    top11 = [(x, y, 0.0) for (_, x, y) in positions_with_dist[:11]]
    arrangements.append(top11)
    
    # With 30 deg rotation
    arrangements.append([(x, y, 30.0) for (_, x, y) in positions_with_dist[:11]])
    
    # With 15 deg rotation
    arrangements.append([(x, y, 15.0) for (_, x, y) in positions_with_dist[:11]])
    
    # 2-row arrangement
    row2 = []
    for i in range(6):
        row2.append((i * d, 0, 0.0))
    for i in range(5):
        row2.append((i * d + d/2, d * np.sqrt(3)/2, 0.0))
    arrangements.append(row2[:11])
    
    best_side = float('inf')
    best_arr = None
    
    for arr in arrangements:
        if len(arr) != N:
            continue
        side, phi = _min_enclosing_hexagon_side(arr)
        if side < best_side:
            best_side = side
            best_arr = arr
        if time.time() > deadline:
            break
    
    if best_arr is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    inner = [[float(x), float(y), float(ang)] for (x, y, ang) in best_arr]
    return inner, (0.0, 0.0), float(best_side), 0.0
# EVOLVE-BLOCK-END
