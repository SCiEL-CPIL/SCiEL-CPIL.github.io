import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    sqrt3 = np.sqrt(3)
    
    # Compact hexagonal cluster: 7-center + 4 extensions
    axial = [
        (0, 0),
        (1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1),
        (2, -1), (2, -2), (1, -2), (0, -2)
    ]
    
    centers = []
    for q, r in axial:
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        centers.append((x, y))
    
    # All vertices of inner hexagons (flat-top, angle 0)
    all_verts = []
    for cx, cy in centers:
        for k in range(6):
            theta = k * np.pi / 3
            all_verts.append((cx + np.cos(theta), cy + np.sin(theta)))
    all_verts = np.array(all_verts)
    
    # Centroid as center
    centroid = all_verts.mean(axis=0)
    
    # Try rotations to minimize outer hexagon side
    best_S = float('inf')
    best_rot = 0.0
    for rot_deg in np.arange(0, 30, 1.0):
        a = np.radians(rot_deg)
        normals = np.array([
            [np.cos(a + np.pi/6 + k * np.pi/3), np.sin(a + np.pi/6 + k * np.pi/3)]
            for k in range(6)
        ])
        vs = all_verts - centroid
        dots = vs @ normals.T
        max_dot = dots.max()
        S = max_dot * 2.0 / sqrt3
        if S < best_S:
            best_S = S
            best_rot = rot_deg
    
    inner = [[float(cx), float(cy), 0.0] for cx, cy in centers]
    return inner, (float(centroid[0]), float(centroid[1])), float(best_S), float(best_rot)
# EVOLVE-BLOCK-END
