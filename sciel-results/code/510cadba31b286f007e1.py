"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    """
    Pack N unit regular hexagons inside a regular hexagon, minimizing outer side length.
    
    Returns:
        inner: list of [x, y, angle_degrees] for each inner hexagon
        center: (x, y) of outer hexagon
        side: side length of outer hexagon
        angle: rotation angle of outer hexagon in degrees
    """
    d = np.sqrt(3.0)
    
    def axial_to_cartesian(q, r):
        x = d * (q + r / 2.0)
        y = d * (3.0 / 2.0) * r
        return x, y
    
    # Unit hexagon vertices (pointy-top, vertex at (1,0))
    unit_vertices = [(np.cos(np.radians(a)), np.sin(np.radians(a))) for a in [0, 60, 120, 180, 240, 300]]
    
    def compute_min_hexagon_side(positions):
        S = 0.0
        for (px, py) in positions:
            for (vx, vy) in unit_vertices:
                for a in [0, 60, 120, 180, 240, 300]:
                    rad = np.radians(a)
                    nx, ny = np.cos(rad), np.sin(rad)
                    proj = (px + vx) * nx + (py + vy) * ny
                    required_S = abs(proj) / (np.sqrt(3) / 2.0)
                    if required_S > S:
                        S = required_S
        return S
    
    # Try several structured arrangements
    arrangements = [
        # Hex cluster 7 + 4
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(1,1),(0,2),(-1,2)],
        # More symmetric
        [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)],
        # 3-4-4 rows
        [(-1,0),(0,0),(1,0),(-2,1),(-1,1),(0,1),(1,1),(-2,-1),(-1,-1),(0,-1),(1,-1)],
        # Linear
        [(k,0) for k in range(-5,6)],
        # 2-3-3-2-1
        [(0,0),(0,1),(-1,1),(1,1),(-1,2),(0,2),(1,2),(-2,-1),(-1,-1),(0,-1),(1,-1)],
    ]
    
    best_S = float('inf')
    best_axial = None
    
    for arr in arrangements:
        if len(arr) != N:
            continue
        positions = [axial_to_cartesian(q, r) for q, r in arr]
        S = compute_min_hexagon_side(positions)
        if S < best_S:
            best_S = S
            best_axial = arr
    
    # Local search
    if best_axial is not None:
        for _ in range(200):
            new_arr = list(best_axial)
            idx = np.random.randint(0, N)
            dq = np.random.randint(-2, 3)
            dr = np.random.randint(-2, 3)
            new_arr[idx] = (new_arr[idx][0] + dq, new_arr[idx][1] + dr)
            positions = [axial_to_cartesian(q, r) for q, r in new_arr]
            S = compute_min_hexagon_side(positions)
            if S < best_S - 1e-9:
                best_S = S
                best_axial = new_arr
    
    positions = [axial_to_cartesian(q, r) for q, r in best_axial]
    inner = [[x, y, 0.0] for x, y in positions]
    center = (0.0, 0.0)
    side = best_S
    angle = 0.0
    
    return inner, center, side, angle
# EVOLVE-BLOCK-END
