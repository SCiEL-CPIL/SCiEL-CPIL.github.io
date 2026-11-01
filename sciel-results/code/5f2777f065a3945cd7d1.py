"""Lennard-Jones cluster global minimum, 55 atoms - Mackay icosahedron construction."""
import numpy as np
import time
import sys

N = 55


def _icosahedron_vertices():
    """Generate 12 icosahedral vertices (unit sphere)."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    # Normalize to unit sphere
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    return verts / norms


def _icosahedron_edges():
    """Generate 30 edges of the icosahedron (pairs of vertex indices)."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # Find all pairs with the edge length
    edge_len = np.linalg.norm(verts[0] - verts[4])  # distance between adjacent vertices
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if np.isclose(d, edge_len, atol=1e-10):
                edges.append((i, j))
    return edges


def _icosahedron_faces():
    """Generate 20 faces of the icosahedron (triples of vertex indices)."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms
    
    # Find all triangles (faces) - each face is an equilateral triangle
    # on the unit sphere with the correct edge length
    edge_len = np.linalg.norm(verts[0] - verts[4])
    faces = []
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                d1 = np.linalg.norm(verts[i] - verts[j])
                d2 = np.linalg.norm(verts[j] - verts[k])
                d3 = np.linalg.norm(verts[k] - verts[i])
                if (np.isclose(d1, edge_len, atol=1e-10) and
                    np.isclose(d2, edge_len, atol=1e-10) and
                    np.isclose(d3, edge_len, atol=1e-10)):
                    faces.append((i, j, k))
    return faces


def construct():
    """Construct 55-atom Mackay icosahedron and relax to local minimum."""
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s limit
    
    # Generate icosahedral geometry
    verts = _icosahedron_vertices()
    edges = _icosahedron_edges()
    faces = _icosahedron_faces()
    
    # Mackay icosahedron: 1 + 12 + 42 = 55 atoms
    # Shell 0: 1 atom at center
    # Shell 1: 12 atoms at icosahedral vertices, radius r1
    # Shell 2: 42 atoms at icosahedral vertices (12) + edge midpoints (30) + face centers (20) = 62... 
    # Wait, that's too many. Let me reconsider.
    # Actually for N=55: 1 + 12 + 42. The 42 in the outer shell are:
    # 12 at vertices (same directions as inner shell), 
    # 30 at edge midpoints
    # That gives 1+12+12+30 = 55. Yes!
    
    # Optimal radii for LJ: inner shell at r1, outer shell at r2
    # For LJ minimum, pair distance should be 2^(1/6) ≈ 1.1225
    # Inner shell radius: r1 = 2^(1/6) ≈ 1.1225
    # Outer shell: atoms at vertices are at distance r2 from center
    # Distance from outer vertex to inner vertex: should be ~2^(1/6)
    # If both are along same direction, distance = r2 - r1
    # So r2 - r1 ≈ 2^(1/6), giving r2 ≈ 2 * 2^(1/6) ≈ 2.245
    
    r1 = 2 ** (1 / 6)  # ≈ 1.1225
    r2 = 2 * r1        # ≈ 2.245
    
    # Build positions
    positions = []
    
    # Center atom
    positions.append(np.array([0.0, 0.0, 0.0]))
    
    # Inner shell: 12 atoms at icosahedral vertices
    for v in verts:
        positions.append(r1 * v)
    
    # Outer shell: 12 atoms at icosahedral vertices (same directions)
    for v in verts:
        positions.append(r2 * v)
    
    # Outer shell: 30 atoms at edge midpoints
    for (i, j) in edges:
        mid_dir = (verts[i] + verts[j])
        mid_norm = np.linalg.norm(mid_dir)
        mid_unit = mid_dir / mid_norm
        positions.append(r2 * mid_unit)
    
    # Total: 1 + 12 + 12 + 30 = 55
    assert len(positions) == N, f"Expected {N} atoms, got {len(positions)}"
    
    positions = np.array(positions, dtype=float)
    
    # Verify minimum distance constraint (atoms must be at least 0.5 apart)
    # Compute all pair distances
    from scipy.spatial.distance import pdist
    dists = pdist(positions)
    min_dist = np.min(dists)
    
    if min_dist < 0.5:
        # If any atoms are too close, scale up slightly
        scale = 0.5 / min_dist * 1.001
        positions = positions * scale
    
    # Try to relax using science_chemistry if available
    best_positions = positions.copy()
    best_energy = _compute_energy(positions)
    
    try:
        import science_chemistry
        relaxed = science_chemistry.relax(positions.copy())
        if relaxed is not None and len(relaxed) == N:
            # Verify validity
            dists_relaxed = pdist(relaxed)
            if np.min(dists_relaxed) >= 0.5:
                energy_relaxed = _compute_energy(relaxed)
                if energy_relaxed < best_energy:
                    best_positions = relaxed
                    best_energy = energy_relaxed
    except Exception:
        pass
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return best_positions
    
    # Try a few random perturbations + relax to see if we can do better
    rng = np.random.default_rng(42)
    for trial in range(10):
        if time.time() - start_time > deadline - 5:
            break
        
        # Small random perturbation
        perturbed = best_positions + rng.normal(0, 0.05, size=best_positions.shape)
        
        try:
            import science_chemistry
            relaxed = science_chemistry.relax(perturbed.copy())
            if relaxed is not None and len(relaxed) == N:
                dists_relaxed = pdist(relaxed)
                if np.min(dists_relaxed) >= 0.5:
                    energy_relaxed = _compute_energy(relaxed)
                    if energy_relaxed < best_energy:
                        best_positions = relaxed
                        best_energy = energy_relaxed
        except Exception:
            pass
    
    return best_positions


def _compute_energy(positions):
    """Compute total Lennard-Jones energy."""
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            if r > 1e-10:
                r6 = r ** 6
                energy += 4.0 * (1.0 / (r6 * r6) - 1.0 / r6)
    return energy
