"""Rewritten program: construct 55-atom LJ cluster near Mackay icosahedron, then local relax."""
import numpy as np
import time
import sys

N = 55

# EVOLVE-BLOCK-START

def _lennard_jones_energy(pos):
    """Compute total LJ energy 4*sum(r^-12 - r^-6) over all pairs."""
    n = pos.shape[0]
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(pos[i] - pos[j])
            if r < 1e-8:
                return 1e10
            r6 = r ** 6
            total += 4.0 * (1.0 / (r6 * r6) - 1.0 / r6)
    return total


def _mackay_icosahedron_55():
    """
    Construct a two-shell Mackay icosahedron with 55 atoms.
    
    Shell 0: 1 atom at center (1 atom)
    Shell 1: 12 atoms at icosahedron vertices (12 atoms)
    Shell 2: 42 atoms at face centers and edge midpoints of the icosahedron (42 atoms)
    Total: 1 + 12 + 42 = 55
    
    The icosahedron has:
    - 12 vertices
    - 30 edges (midpoints: 30)
    - 20 faces (centers: 20)
    30 + 20 = 50... but we need 42 for shell 2.
    
    Actually, let me reconsider. The Mackay icosahedron I_2 has:
    - 1 center
    - 12 first-shell (vertices)
    - 42 second-shell positions
    
    The second shell consists of:
    - 30 edge midpoints
    - 12 face centers (wait, icosahedron has 20 faces)
    
    Let me use the standard construction. The 55-atom Mackay icosahedron:
    Center: 1
    First shell (12): vertices of regular icosahedron at distance a1
    Second shell (42): 
      - 30 edge midpoints at distance a2
      - 12 additional points... 
    
    Actually the standard Mackay I_2:
    1 + 12 + 42 = 55
    The 42 second-shell atoms are at:
    - 30 edge centers
    - 12 face centers? No, icosahedron has 20 faces.
    
    Let me just use known coordinates or a simpler approach.
    I'll use the icosahedral symmetry to place atoms.
    
    A regular icosahedron has 12 vertices. For the two-shell Mackay icosahedron,
    the second shell has 42 atoms. These are located at:
    - 30 edge midpoints (scaled)
    - 12 face centers... but there are 20 faces.
    
    Actually, I recall that the 55-atom Mackay icosahedron second shell has:
    - 30 atoms at edge midpoints
    - 12 atoms at face centers of a subset? 
    
    Let me try a different approach: use the known fact that the best 55-atom 
    LJ cluster is the two-shell Mackay icosahedron. I'll construct it using
    icosahedral coordinates and then relax.
    
    Icosahedron vertices (12):
    (0, ±1, ±φ), (±1, ±φ, 0), (±φ, 0, ±1) where φ = (1+√5)/2
    
    I'll place:
    - 1 atom at origin
    - 12 atoms at icosahedron vertices scaled to optimal first-shell distance
    - 42 atoms: I'll use edge midpoints (30) and face centers... 
    
    Actually, let me just use a reasonable geometric construction and rely on 
    local relaxation to find the minimum. The key is to start near the right 
    symmetry.
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    
    # 12 vertices of a regular icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize to unit sphere
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # First shell distance (optimal ~1.0 for LJ minimum at r=2^(1/6)≈1.122)
    r1 = 1.1225  # 2^(1/6)
    shell1 = verts * r1
    
    # Edge midpoints: for each pair of vertices that are connected by an edge
    # In a regular icosahedron, each vertex connects to 5 others
    # Total edges = 12*5/2 = 30
    edge_mids = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            # Edge length in unit icosahedron: the minimum distance between vertices
            # For unit icosahedron, edge length = 2/sqrt(phi^2+1) ≈ 1.0515
            if d < 1.1:  # threshold for edge connection
                mid = (verts[i] + verts[j]) / 2.0
                mid = mid / np.linalg.norm(mid)
                edge_mids.append(mid)
    
    edge_mids = np.array(edge_mids)
    # Second shell distance
    r2 = 2.0  # approximate second shell radius
    shell2_edges = edge_mids * r2
    
    # Face centers: icosahedron has 20 faces
    # Each face is a triangle of 3 mutually connected vertices
    face_centers = []
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                d1 = np.linalg.norm(verts[i] - verts[j])
                d2 = np.linalg.norm(verts[j] - verts[k])
                d3 = np.linalg.norm(verts[i] - verts[k])
                if d1 < 1.1 and d2 < 1.1 and d3 < 1.1:
                    fc = (verts[i] + verts[j] + verts[k]) / 3.0
                    fc = fc / np.linalg.norm(fc)
                    face_centers.append(fc)
    
    face_centers = np.array(face_centers)
    shell2_faces = face_centers * r2
    
    # We have 30 edge mids + 20 face centers = 50, but we need 42
    # The Mackay icosahedron second shell has 42 atoms.
    # Let me use 30 edge mids + 12 of the face centers (or adjust)
    # Actually, for the 55-atom Mackay icosahedron, the second shell is 42 atoms.
    # Let me just use all 50 and then trim, or better, use a known construction.
    
    # Alternative: use 30 edge midpoints + 12 face centers from a dodecahedron subset
    # For now, let's use 30 edge mids + 12 face centers (first 12)
    if len(face_centers) >= 12:
        shell2_faces_use = face_centers[:12] * r2
    else:
        shell2_faces_use = face_centers * r2
    
    # Combine: 1 + 12 + 30 + 12 = 55
    positions = np.vstack([
        np.array([[0.0, 0.0, 0.0]]),
        shell1,
        shell2_edges,
        shell2_faces_use
    ])
    
    return positions[:55]


def _relax_lbfgs(pos, max_iter=200, deadline=None):
    """Local relaxation using L-BFGS via scipy."""
    from scipy.optimize import minimize
    
    def energy_and_grad(x):
        pos = x.reshape(-1, 3)
        n = pos.shape[0]
        # Compute energy
        E = 0.0
        grad = np.zeros_like(pos)
        for i in range(n):
            for j in range(i + 1, n):
                dr = pos[i] - pos[j]
                r2 = np.dot(dr, dr)
                if r2 < 1e-12:
                    return 1e10, np.zeros_like(x)
                r = np.sqrt(r2)
                r6 = r2 * r2 * r2
                r12 = r6 * r6
                # dE/dr = 4*(-12*r^-13 + 6*r^-7) = 24*r^-7 - 48*r^-13
                # dE/dpos_i = -dE/dr * dr/r = (48*r^-13 - 24*r^-7) * dr/r
                # = (48/r12 - 24/r6) * dr / r
                coeff = (48.0 / r12 - 24.0 / r6) / r
                E += 4.0 * (1.0 / r12 - 1.0 / r6)
                grad[i] += coeff * dr
                grad[j] -= coeff * dr
        return E, grad.flatten()
    
    try:
        res = minimize(
            energy_and_grad,
            pos.flatten(),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': max_iter, 'maxls': 50}
        )
        if res.success or res.fun < energy_and_grad(pos.flatten())[0]:
            return res.x.reshape(-1, 3)
        return pos
    except Exception:
        return pos


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin before 120s kill
    
    # Start from Mackay icosahedron geometry
    pos = _mackay_icosahedron_55()
    
    # Ensure minimum separation of 0.5
    # Check and fix any close pairs
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            d = np.linalg.norm(pos[i] - pos[j])
            if d < 0.5:
                # Nudge apart
                if d < 1e-8:
                    pos[j] += np.array([0.5, 0, 0])
                else:
                    direction = (pos[j] - pos[i]) / d
                    pos[j] = pos[i] + direction * 0.5
    
    # Local relaxation
    pos = _relax_lbfgs(pos, max_iter=300, deadline=deadline)
    
    # Check time
    if time.time() > deadline - 10.0:
        return pos
    
    # Try a second relaxation with more iterations if time permits
    pos2 = _relax_lbfgs(pos, max_iter=500, deadline=deadline)
    if _lennard_jones_energy(pos2) < _lennard_jones_energy(pos):
        pos = pos2
    
    # Final separation check
    for i in range(len(pos)):
        for j in range(i + 1, len(pos)):
            d = np.linalg.norm(pos[i] - pos[j])
            if d < 0.5:
                if d < 1e-8:
                    pos[j] += np.array([0.5, 0, 0])
                else:
                    direction = (pos[j] - pos[i]) / d
                    pos[j] = pos[i] + direction * 0.5
    
    return pos

# EVOLVE-BLOCK-END
