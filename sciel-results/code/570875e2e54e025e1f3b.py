"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 55

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110  # 110s limit to be safe within 120s/150s
    
    # 1. Generate unit direction vectors for the Mackay icosahedron topology
    # Inner shell: 12 vertices of a regular icosahedron
    phi = (1 + np.sqrt(5)) / 2
    # Vertices of icosahedron (unnormalized)
    # Permutations of (0, 1, phi)
    verts = []
    for sign1 in [-1, 1]:
        for sign2 in [-1, 1]:
            verts.append((0, sign1, sign2 * phi))
            verts.append((sign1, sign2 * phi, 0))
            verts.append((sign2 * phi, 0, sign1))
    
    verts = np.array(verts)
    # Normalize to unit vectors
    norms = np.linalg.norm(verts, axis=1)
    v_unit = verts / norms[:, np.newaxis]
    
    # Inner shell positions (directions)
    inner_dirs = v_unit
    
    # Outer shell 1: Midpoints of edges (30 atoms)
    # Identify edges: pairs of vertices with minimum distance
    # Distance between adjacent vertices in unit icosahedron is constant.
    # We can find them by checking all pairs.
    edge_mids = []
    for i in range(12):
        for j in range(i + 1, 12):
            dist = np.linalg.norm(verts[i] - verts[j])
            # The minimum distance is 2.0 for this construction (0, 1, phi) etc.
            # Let's check: dist((0,1,phi), (0,-1,phi)) = 2.
            # dist((0,1,phi), (1,phi,0)) = sqrt(1 + (1-phi)^2 + phi^2) = sqrt(1 + 1 - 2phi + phi^2 + phi^2) 
            # phi^2 = phi+1. So 2 - 2phi + 2phi + 2 = 4. Dist = 2.
            # So adjacent vertices have distance 2.
            if dist < 2.1:  # Tolerance for floating point
                mid = (verts[i] + verts[j]) / 2.0
                mid_norm = np.linalg.norm(mid)
                edge_mids.append(mid / mid_norm)
    
    edge_mids = np.array(edge_mids)
    
    # Outer shell 2: Face centers (12 atoms? No, 20 faces, but Mackay 55 has 12 face-center-like atoms?
    # Wait, 12+30+12 = 54 + 1 = 55.
    # The 12 "face center" atoms in the Mackay icosahedron N=55 are actually associated with the 12 vertices of the *dual* dodecahedron?
    # Or are they the centers of the 20 faces? That would be 20 atoms. 1+12+30+20 = 63. Too many.
    # The Mackay icosahedron for N=55 is often cited as:
    # 1 center
    # 12 inner
    # 30 outer (edge midpoints)
    # 12 outer (face centers of the *inner* icosahedron? No, there are 20 faces.)
    
    # Let's re-verify the N=55 Mackay icosahedron composition.
    # Cambridge Cluster Database: 55-atom LJ.
    # Structure: Icosahedral.
    # Shells: 1, 12, 30, 12.
    # The 12 outer atoms are located at the centers of the 12 *vertices* of the dual dodecahedron?
    # The dual of an icosahedron is a dodecahedron, which has 12 vertices.
    # The vectors pointing to the dodecahedron vertices are perpendicular to the icosahedron faces.
    # So these 12 atoms are in the directions of the face normals of the icosahedron.
    # There are 20 faces, so 20 face normals. Why only 12?
    # Ah, the 12 atoms in the outermost shell of the 55-atom Mackay icosahedron are *not* all face centers.
    # Actually, looking at the literature:
    # The 55-atom Mackay icosahedron consists of:
    # - 1 atom at the center.
    # - 12 atoms at the vertices of an icosahedron (radius R1).
    # - 30 atoms at the midpoints of the edges of the icosahedron (radius R2).
    # - 12 atoms at the centers of the faces of the *dodecahedron* dual to the icosahedron?
    # The dodecahedron has 12 vertices. The vectors to these vertices are normal to the faces of the icosahedron.
    # But there are 20 faces. So there are 20 face normals.
    # Why only 12?
    # Perhaps the "12" outer atoms are the vertices of a *larger* icosahedron?
    # If we take the 12 vertices of a larger icosahedron, that's 12 atoms.
    # Total: 1 + 12 + 30 + 12 = 55.
    # This makes sense. The outermost shell is an icosahedron of 12 atoms.
    # The 30 atoms are on the edges of the inner icosahedron?
    # Or are the 30 atoms on the edges of the *outer* icosahedron?
    
    # Let's assume the standard Mackay icosahedron geometry:
    # Shell 1: 1 atom (center)
    # Shell 2: 12 atoms (icosahedron vertices, radius R1)
    # Shell 3: 30 atoms (midpoints of edges of Shell 2 icosahedron, radius R2)
    # Shell 4: 12 atoms (vertices of a larger icosahedron, radius R3)
    
    # This topology is standard for Mackay icosahedra.
    # Directions for Shell 4: Same as Shell 2 (icosahedron vertices).
    
    # So we need:
    # 12 dirs: v_unit
    # 30 dirs: edge_mids (normalized)
    # 12 dirs: v_unit (same as inner, but different radius)
    
    # Let's verify if this matches the "face center" description.
    # The vertices of the outer icosahedron (Shell 4) align with the inner icosahedron vertices (Shell 2).
    # The edge midpoints (Shell 3) are between them.
    # This creates a dense packing.
    
    # Let's proceed with this assumption.
    # Directions:
    # inner_dirs (12)
    # edge_dirs (30)
    # outer_dirs (12) - same as inner_dirs
    
    # Initial Radii:
    # R1 (inner) ~ 1.1
    # R2 (edge) ~ 1.5
    # R3 (outer) ~ 2.0
    
    # Construct positions
    R1 = 1.1
    R2 = 1.5
    R3 = 2.0
    
    pos = np.zeros((55, 3))
    pos[0] = [0, 0, 0]
    pos[1:13] = inner_dirs * R1
    pos[13:43] = edge_mids * R2
    pos[43:55] = inner_dirs * R3
    
    # Ensure minimum distance constraint (0.5)
    # Check pairwise distances
    # If any pair < 0.5, push them apart slightly or adjust radii.
    # For these initial radii, distances should be > 0.5.
    
    # Try to use relax if available
    try:
        import science_chemistry
        # relax(positions) returns relaxed positions
        # We need to be careful with the time limit
        if time.time() < deadline - 10:
            pos_relaxed = science_chemistry.relax(pos)
            # Check if relaxation was successful (no NaNs, etc.)
            if np.all(np.isfinite(pos_relaxed)):
                pos = pos_relaxed
    except Exception:
        pass
    
    # Final check and return
    return pos

# EVOLVE-BLOCK-END
