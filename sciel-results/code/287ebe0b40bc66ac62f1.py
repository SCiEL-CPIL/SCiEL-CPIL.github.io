import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    # Generate 11 hexagons in a honeycomb pattern
    d = np.sqrt(3)
    centers = []
    
    # Row 0: 3 hexes
    for j in [-1, 0, 1]:
        centers.append((j * d, 0.0))
        
    # Row 1: 3 hexes
    y1 = 1.5
    for j in [-1, 0, 1]:
        centers.append((j * d + d/2, y1))
        
    # Row 2: 3 hexes
    y2 = 3.0
    for j in [-1, 0, 1]:
        centers.append((j * d, y2))
        
    # Row 3: 2 hexes
    y3 = 4.5
    for j in [0, 1]:
        centers.append((j * d + d/2, y3))
        
    centers = np.array(centers)
    
    # Center the cluster
    center = np.mean(centers, axis=0)
    centers = centers - center
    
    # Generate vertices for all inner hexagons
    # Inner hexagons are aligned with angle 0 (pointy top)
    angles = np.arange(6) * np.pi / 3
    verts = np.array([[np.cos(a), np.sin(a)] for a in angles])
    
    all_verts = []
    for c in centers:
        all_verts.append(c + verts)
    all_verts = np.vstack(all_verts)
    
    # Outer hexagon normals (pointy top alignment)
    norm_angles = np.arange(6) * np.pi / 3
    normals = np.array([[np.cos(a), np.sin(a)] for a in norm_angles])
    
    # Calculate max projection
    # proj = all_verts @ normals.T
    # max_proj = np.max(proj)
    # L = (2 / np.sqrt(3)) * max_proj
    
    # We can just compute this quickly
    max_proj = 0
    for n in normals:
        projs = all_verts @ n
        max_proj = max(max_proj, np.max(projs))
        
    L = (2 / np.sqrt(3)) * max_proj
    
    # Add small epsilon for safety
    L += 1e-6
    
    # Construct inner list: [x, y, angle_degrees]
    inner = []
    for c in centers:
        inner.append([c[0], c[1], 0.0])
        
    return inner, (0.0, 0.0), L, 0.0
# EVOLVE-BLOCK-END
