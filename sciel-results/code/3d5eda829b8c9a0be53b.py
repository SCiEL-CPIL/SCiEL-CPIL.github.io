import numpy as np
import science_chemistry

N = 55

def construct():
    # Generate icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2.0
    # 12 vertices of a regular icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    
    # Normalize to unit length
    norms = np.linalg.norm(verts, axis=1)
    verts_unit = verts / norms[:, np.newaxis]
    
    # Find edges (pairs of vertices with minimum distance)
    dists = np.linalg.norm(verts_unit[:, np.newaxis] - verts_unit[np.newaxis, :], axis=2)
    min_dist = np.min(dists)
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            if abs(dists[i, j] - min_dist) < 1e-10:
                edges.append((i, j))
                
    # Generate midpoints of edges
    mids = []
    for i, j in edges:
        mid = (verts_unit[i] + verts_unit[j]) / 2.0
        norm_mid = np.linalg.norm(mid)
        mids.append(mid / norm_mid)
    mids_unit = np.array(mids)
    
    # Initial radii
    # r1 is the distance to the first shell (12 atoms)
    # r2 is the distance to the second shell (42 atoms)
    r1 = 1.0
    r2 = 1.7
    
    # Construct positions
    pos = np.zeros((55, 3))
    
    # Center
    pos[0] = [0, 0, 0]
    
    # Inner shell (12 atoms)
    pos[1:13] = r1 * verts_unit
    
    # Outer shell (42 atoms)
    # 12 "corner" atoms (same directions as inner vertices)
    pos[13:25] = r2 * verts_unit
    
    # 30 "edge" atoms (same directions as edge midpoints)
    pos[25:55] = r2 * mids_unit
    
    # Relax the structure
    try:
        relaxed_pos = science_chemistry.relax(pos)
        return relaxed_pos
    except Exception:
        return pos
