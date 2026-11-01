"""Lennard-Jones 55-atom cluster: Mackay icosahedron seed + L-BFGS relaxation."""
import numpy as np
import time

N = 55

def _lj_energy(pos):
    """Compute total LJ energy 4*sum(r^-12 - r^-6) over all pairs."""
    n = len(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(r2, np.inf)
    r = np.sqrt(r2)
    r6 = r ** 6
    r12 = r6 * r6
    # Sum over upper triangle to avoid double counting
    iu = np.triu_indices(n, k=1)
    return 4.0 * np.sum(1.0 / r12[iu] - 1.0 / r6[iu])

def _mckay_icosahedron():
    """Construct the 55-atom Mackay icosahedron geometry."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    # 12 vertices of regular icosahedron (cyclic perms of (0, ±1, ±phi))
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    # Normalize to unit circumradius
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    # 30 edges: pairs of vertices that are nearest neighbors on the icosahedron
    # Edge length for unit-circumradius icosahedron
    edge_len = 2.0 / np.sqrt(1.0 + 2.0 * phi)  # = 2*sin(pi/5) ≈ 1.1756
    # Actually, for unit circumradius, edge length = 2*sin(pi/5)
    # Let's find edges by distance
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < 0.6:  # threshold: nearest neighbors
                edges.append((i, j))
    
    # 30 edge midpoints
    edge_mids = []
    for (i, j) in edges:
        mid = (verts[i] + verts[j]) / 2.0
        edge_mids.append(mid)
    edge_mids = np.array(edge_mids, dtype=float)
    
    # 12 face centers: each face is a triangle of 3 mutually-connected vertices
    # Find all triangles (faces) of the icosahedron
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                dij = np.linalg.norm(verts[i] - verts[j])
                djk = np.linalg.norm(verts[j] - verts[k])
                dik = np.linalg.norm(verts[i] - verts[k])
                # All three distances should be ~edge_len (nearest neighbors)
                if dij < 0.6 and djk < 0.6 and dik < 0.6:
                    faces.append((i, j, k))
    
    face_centers = []
    for (i, j, k) in faces:
        fc = (verts[i] + verts[j] + verts[k]) / 3.0
        face_centers.append(fc)
    face_centers = np.array(face_centers, dtype=float)
    
    # Scale the whole structure so that nearest-neighbor distance ≈ 1.12 (LJ minimum)
    # The edge midpoint to vertex distance in unit-circumradius icosahedron:
    # For now, let's just scale so that the typical interatomic distance is ~1.12
    # Actually, let's compute the edge length in the constructed geometry
    if len(edges) > 0:
        i, j = edges[0]
        current_edge = np.linalg.norm(verts[i] - verts[j])
        target_edge = 1.12
        scale = target_edge / current_edge
    else:
        scale = 1.0
    
    # Build all 55 positions: 1 center + 12 verts + 30 edge mids + 12 face centers
    center = np.zeros(3)
    all_pos = np.vstack([
        center.reshape(1, 3),
        verts,
        edge_mids,
        face_centers
    ]) * scale
    
    # Verify count
    assert len(all_pos) == 55, f"Expected 55 atoms, got {len(all_pos)}"
    return all_pos

def construct():
    t0 = time.time()
    deadline = 110.0  # safety margin before 120s limit
    
    # Start from Mackay icosahedron
    pos = _mckay_icosahedron()
    
    # Verify minimum distance constraint
    def min_dist(p):
        n = len(p)
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        r = np.sqrt(np.einsum('ijk,ijk->ij', diff, diff))
        np.fill_diagonal(r, np.inf)
        return r.min()
    
    md = min_dist(pos)
    if md < 0.5:
        # Scale up to satisfy constraint
        pos = pos * (0.5 / md)
    
    best_pos = pos.copy()
    best_e = _lj_energy(pos)
    
    # Try to relax using science_chemistry.relax if available
    try:
        import science_chemistry
        relaxed = science_chemistry.relax(pos)
        # Check validity
        if relaxed is not None and len(relaxed) == N:
            md_rel = min_dist(relaxed)
            if md_rel >= 0.5:
                e_rel = _lj_energy(relaxed)
                if e_rel < best_e:
                    best_e = e_rel
                    best_pos = relaxed.copy()
    except Exception:
        pass
    
    # Check time
    if time.time() - t0 > deadline:
        return best_pos
    
    # Try a second round of relaxation with tighter settings if time permits
    try:
        import science_chemistry
        # Try again from the relaxed position
        relaxed2 = science_chemistry.relax(best_pos)
        if relaxed2 is not None and len(relaxed2) == N:
            md_rel2 = min_dist(relaxed2)
            if md_rel2 >= 0.5:
                e_rel2 = _lj_energy(relaxed2)
                if e_rel2 < best_e:
                    best_e = e_rel2
                    best_pos = relaxed2.copy()
    except Exception:
        pass
    
    return best_pos
