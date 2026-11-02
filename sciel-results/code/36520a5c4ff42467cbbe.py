import numpy as np

N = 12

# EVOLVE-BLOCK-START
def construct():
    # Generate 3x4 honeycomb lattice centers
    u = np.array([np.sqrt(3), 0.0])
    v = np.array([np.sqrt(3)/2.0, 1.5])
    
    centers = []
    for i in range(3):
        for j in range(4):
            c = i * u + j * v
            centers.append(c)
            
    centers = np.array(centers)
    
    # Generate vertices for all inner hexagons
    # Hexagon side 1, oriented with flat bottom (vertices at 30, 90, ...)
    # This orientation has horizontal top/bottom sides.
    angles = np.array([np.pi/6 + k * np.pi/3 for k in range(6)])
    verts_rel = np.column_stack((np.cos(angles), np.sin(angles)))
    
    all_verts = []
    for c in centers:
        verts = c + verts_rel
        all_verts.append(verts)
        
    all_verts = np.vstack(all_verts)
    
    # Center the configuration
    centroid = np.mean(all_verts, axis=0)
    all_verts -= centroid
    
    # Optimize outer hexagon orientation
    # Outer hexagon side S, inradius R = S * sqrt(3) / 2
    # Constraint: | v dot n_k | <= R
    # S >= 2/sqrt(3) * max(|v dot n_k|)
    
    best_S = float('inf')
    best_alpha = 0.0
    
    # Coarse search
    alphas = np.linspace(0, np.pi/3, 181)
    for a in alphas:
        max_proj = 0.0
        for k in range(6):
            n_angle = a + k * np.pi/3
            n = np.array([np.cos(n_angle), np.sin(n_angle)])
            projs = np.abs(all_verts @ n)
            m = np.max(projs)
            if m > max_proj:
                max_proj = m
        S = (2.0 / np.sqrt(3)) * max_proj
        if S < best_S:
            best_S = S
            best_alpha = a
            
    # Fine search
    if best_alpha > 0 and best_alpha < np.pi/3:
        fine_alphas = np.linspace(max(0, best_alpha - 0.05), min(np.pi/3, best_alpha + 0.05), 101)
        for a in fine_alphas:
            max_proj = 0.0
            for k in range(6):
                n_angle = a + k * np.pi/3
                n = np.array([np.cos(n_angle), np.sin(n_angle)])
                projs = np.abs(all_verts @ n)
                m = np.max(projs)
                if m > max_proj:
                    max_proj = m
            S = (2.0 / np.sqrt(3)) * max_proj
            if S < best_S:
                best_S = S
                best_alpha = a

    # Construct output
    inner = []
    for c in centers:
        # Angle 0.0 corresponds to the orientation used (flat bottom)
        inner.append([c[0], c[1], 0.0])
        
    outer_center = (centroid[0], centroid[1])
    outer_side = best_S
    outer_angle = np.degrees(best_alpha)
    
    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
