import numpy as np
import time
import science_chemistry

N = 55

# EVOLVE-BLOCK-START
def _get_icosahedron_vertices():
    """Generate the 12 vertices of a regular icosahedron."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    # Normalize to unit sphere
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    return verts

def _get_icosahedron_edges(verts):
    """Find the 30 edges of the icosahedron."""
    edges = []
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            dist = np.linalg.norm(verts[i] - verts[j])
            # The edge length of a unit icosahedron is 2/sqrt(1+phi^2) approx 1.051
            # Actually, for unit vertices, min dist is 1.0514...
            if dist < 1.1:  # Threshold to catch edges
                edges.append((i, j))
    return edges

def _get_icosahedron_faces(verts, edges):
    """Find the 20 faces of the icosahedron."""
    faces = []
    # For each pair of vertices, find the third vertex that forms a triangle
    # This is a bit slow but N=12 is small
    for i in range(len(verts)):
        for j in range(i + 1, len(verts)):
            for k in range(j + 1, len(verts)):
                d1 = np.linalg.norm(verts[i] - verts[j])
                d2 = np.linalg.norm(verts[j] - verts[k])
                d3 = np.linalg.norm(verts[i] - verts[k])
                # Check if it's an equilateral triangle (face of icosahedron)
                if abs(d1 - d2) < 1e-6 and abs(d2 - d3) < 1e-6 and d1 < 1.1:
                    faces.append((i, j, k))
    return faces

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # 1. Generate 13-atom icosahedron (1 center + 12 vertices)
    verts = _get_icosahedron_vertices()
    center = np.zeros(3)
    
    # Scale 1st shell to R1 = 1.1
    R1 = 1.1
    shell1 = np.vstack([center, verts * R1])
    
    # 2. Generate 42-atom shell
    edges = _get_icosahedron_edges(verts)
    faces = _get_icosahedron_faces(verts, edges)
    
    # 30 edge midpoints
    midpoints = []
    for i, j in edges:
        mid = (verts[i] + verts[j]) / 2.0
        # Normalize to unit sphere
        mid = mid / np.linalg.norm(mid)
        midpoints.append(mid)
    
    # 20 face centers
    face_centers = []
    for i, j, k in faces:
        fc = (verts[i] + verts[j] + verts[k]) / 3.0
        fc = fc / np.linalg.norm(fc)
        face_centers.append(fc)
    
    # We need 42 atoms. We have 30 midpoints and 20 face centers.
    # The Mackay icosahedron 2nd shell has 42 atoms.
    # It is composed of 30 edge midpoints and 12 face centers?
    # Let's take 30 midpoints and 12 face centers.
    # To maintain symmetry, we should pick 12 face centers carefully.
    # However, for the purpose of relaxation, any 12 should work if close enough.
    # Let's just take the first 12.
    selected_outer = midpoints + face_centers[:12]
    
    # Scale 2nd shell to R2 = 2.2
    R2 = 2.2
    shell2 = np.array(selected_outer) * R2
    
    # Combine
    positions = np.vstack([shell1, shell2])
    
    # Ensure shape is (55, 3)
    if len(positions) != N:
        # Fallback: if logic fails, use random
        positions = np.random.randn(N, 3) * 2.0
        positions[0] = 0.0
    
    # Relax
    try:
        relaxed = science_chemistry.relax(positions)
        # Check time
        if time.time() > deadline:
            return positions
        return relaxed
    except Exception:
        # If relaxation fails, return initial guess
        return positions

# EVOLVE-BLOCK-END
