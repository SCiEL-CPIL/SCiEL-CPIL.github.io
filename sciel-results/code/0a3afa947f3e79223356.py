import numpy as np
from scipy.optimize import minimize_scalar

N = 11

# EVOLVE-BLOCK-START
def construct():
    # 1. Define lattice positions for 11 hexagons
    axial = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)
    ]
    axial += [
        (1, 1), (-1, 2), (-2, 1), (-1, -1)
    ]
    
    centers = []
    for q, r in axial:
        x = 1.5 * q
        y = (np.sqrt(3) / 2) * (q + 2 * r)
        centers.append((x, y))
    
    centers = np.array(centers)
    
    # Center the cluster at the origin
    cm = np.mean(centers, axis=0)
    centers = centers - cm
    
    # Vertices of a unit hexagon (flat-top) centered at origin
    angles = np.deg2rad(np.arange(0, 360, 60))
    offsets = np.array([(np.cos(a), np.sin(a)) for a in angles])
    
    # All vertices of all 11 hexagons
    verts = centers[:, np.newaxis, :] + offsets[np.newaxis, :, :]
    verts = verts.reshape(-1, 2)
    
    def min_side(angle):
        normal_angles = angle + np.pi/6 + np.arange(6) * np.pi/3
        normals = np.array([(np.cos(a), np.sin(a)) for a in normal_angles])
        
        proj = verts @ normals.T
        max_proj = np.max(proj, axis=0)
        
        S = np.max(max_proj) * 2 / np.sqrt(3)
        return S

    res = minimize_scalar(min_side, bounds=(0, np.pi/3), method='bounded')
    best_angle = res.x
    best_side = res.fun
    
    inner = [[c[0], c[1], 0.0] for c in centers]
    
    return inner, (0.0, 0.0), best_side, np.rad2deg(best_angle)
# EVOLVE-BLOCK-END
