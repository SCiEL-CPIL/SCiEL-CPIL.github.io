"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def _hex_to_cartesian(q, r):
    """Convert axial hex coordinates (pointy-top) to Cartesian."""
    x = np.sqrt(3) * (q + r / 2.0)
    y = 1.5 * r
    return x, y

def _find_min_enclosing_hexagon(points, n_theta=13, n_center=15):
    """Find minimal enclosing regular hexagon for a set of 2D points.
    Returns (center_x, center_y, side_length, angle_degrees)."""
    best = None
    best_s = float('inf')
    
    for theta_deg in np.linspace(0, 60, n_theta):
        theta = np.radians(theta_deg)
        normals = []
        for i in range(6):
            angle = theta + i * np.pi / 3
            normals.append((np.cos(angle), np.sin(angle)))
        
        centroid = np.mean(points, axis=0)
        spread = np.max(points, axis=0) - np.min(points, axis=0)
        scale = max(spread[0], spread[1], 1.0)
        
        for cx in np.linspace(centroid[0] - scale, centroid[0] + scale, n_center):
            for cy in np.linspace(centroid[1] - scale, centroid[1] + scale, n_center):
                apothem = 0.0
                for p in points:
                    for nx, ny in normals:
                        d = abs((p[0] - cx) * nx + (p[1] - cy) * ny)
                        if d > apothem:
                            apothem = d
                s = apothem * 2.0 / np.sqrt(3)
                if s < best_s:
                    best_s = s
                    best = (float(cx), float(cy), float(s), float(theta_deg))
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_result = None
    best_side = float('inf')
    
    # Generate candidate hexagonal grid arrangements
    # Base: hexagonal cluster of radius 1 (7 hexagons)
    base = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    
    # Ring 2 positions
    ring2 = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if max(abs(q), abs(r), abs(-q-r)) == 2:
                ring2.append((q, r))
    
    # Try several patterns of 11 hexagons
    candidates = []
    
    # Pattern 1: base + 4 from ring2 (symmetric-ish)
    for i in range(0, len(ring2), 3):
        if i + 3 < len(ring2):
            candidates.append(base + [ring2[i], ring2[i+1], ring2[i+2], ring2[i+3]])
    
    # Pattern 2: More spread out
    candidates.append([(0,0), (1,0), (2,0), (3,0), (4,0),
                       (-1,0), (-2,0), (1,1), (0,1), (-1,1), (2,1)])
    
    # Pattern 3: Compact
    candidates.append([(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1),
                       (2,0), (2,-1), (1,-2), (-2,1)])
    
    for positions in candidates:
        if time.time() > deadline:
            break
        if len(positions) != N:
            continue
        
        points = np.array([_hex_to_cartesian(q, r) for q, r in positions])
        result = _find_min_enclosing_hexagon(points)
        if result:
            cx, cy, s, theta = result
            if s < best_side:
                best_side = s
                inner = [[float(p[0]), float(p[1]), 0.0] for p in points]
                best_result = (inner, (cx, cy), s, theta)
    
    if best_result is None:
        # Fallback
        inner = [[np.sqrt(3) * k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_result
# EVOLVE-BLOCK-END
