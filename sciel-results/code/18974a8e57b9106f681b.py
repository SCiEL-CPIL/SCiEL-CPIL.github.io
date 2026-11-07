"""Program for Sutton-Chen TM cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

_START_TIME = time.time()
_DEADLINE = 290.0  # Safety margin before 300s kill


def _time_left():
    return _DEADLINE - (time.time() - _START_TIME)


def _is_valid(positions, min_dist=0.35):
    """Check if positions are valid (finite, no atoms closer than min_dist)."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    for i in range(n - 1):
        d = np.linalg.norm(positions[i + 1:] - positions[i], axis=1)
        if np.any(d < min_dist):
            return False
    return True


def _generate_fcc(n):
    """Generate a cluster of n atoms in an FCC-like structure."""
    max_r = int(np.ceil(n ** (1 / 3))) + 3
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                if (i + j + k) % 2 == 0:
                    p = np.array([i, j, k]) * (1.0 / np.sqrt(2.0))
                    candidates.append((np.linalg.norm(p), p))
    
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)


def _generate_bcc(n):
    """Generate a cluster of n atoms in a BCC-like structure."""
    max_r = int(np.ceil(n ** (1 / 3))) + 3
    candidates = []
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p1 = np.array([i, j, k])
                p2 = np.array([i + 0.5, j + 0.5, k + 0.5])
                candidates.append((np.linalg.norm(p1), p1))
                candidates.append((np.linalg.norm(p2), p2))
    
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)


def _generate_hcp(n):
    """Generate a cluster of n atoms in an HCP-like structure."""
    # HCP lattice: two interpenetrating triangular lattices
    a = 1.0
    c = np.sqrt(8.0 / 3.0) * a
    pts = []
    max_r = int(np.ceil(n ** (1 / 3))) + 3
    candidates = []
    
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                # Layer A
                x = a * (i + 0.5 * j)
                y = a * np.sqrt(3) / 2 * j
                z = c * k
                p1 = np.array([x, y, z])
                candidates.append((np.linalg.norm(p1), p1))
                
                # Layer B (offset)
                x2 = a * (i + 0.5 * j + 1.0 / np.sqrt(3.0))
                y2 = a * np.sqrt(3) / 2 * (j + 1.0 / 3.0)
                z2 = c * (k + 1.0 / 3.0)
                p2 = np.array([x2, y2, z2])
                candidates.append((np.linalg.norm(p2), p2))
    
    candidates.sort(key=lambda x: x[0])
    pts = [p for _, p in candidates[:n]]
    return np.array(pts)


def _generate_icosahedron(n):
    """Generate a cluster of n atoms with icosahedral symmetry."""
    # Start with icosahedron vertices and add layers
    phi = (1 + np.sqrt(5)) / 2
    # Icosahedron vertices (12)
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])  # Normalize to unit sphere
    
    pts = [np.zeros(3)]  # Center atom
    pts.extend(verts.tolist())  # 12 vertices
    
    # Add second shell (30 atoms at edges)
    # Midpoints of edges of icosahedron
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < 0.5:  # Close enough to be an edge
                mid = (verts[i] + verts[j]) / 2
                mid = mid / np.linalg.norm(mid) * 1.05  # Slightly outward
                pts.append(mid)
    
    # Add third shell if needed
    while len(pts) < n:
        # Add atoms on faces or further out
        for i in range(12):
            for j in range(i + 1, 12):
                for k in range(j + 1, 12):
                    if len(pts) >= n:
                        break
                    # Check if this forms a triangle
                    d1 = np.linalg.norm(verts[i] - verts[j])
                    d2 = np.linalg.norm(verts[j] - verts[k])
                    d3 = np.linalg.norm(verts[k] - verts[i])
                    if d1 < 0.5 and d2 < 0.5 and d3 < 0.5:
                        centroid = (verts[i] + verts[j] + verts[k]) / 3
                        centroid = centroid / np.linalg.norm(centroid) * 1.15
                        pts.append(centroid)
                if len(pts) >= n:
                    break
            if len(pts) >= n:
                break
        if len(pts) < n:
            # Add more outer shell
            for v in verts:
                if len(pts) >= n:
                    break
                outer = v * 1.3
                pts.append(outer)
    
    return np.array(pts[:n])


def _generate_random_compact(n, rng, radius=2.0):
    """Generate n random points within a sphere, compactly distributed."""
    pts = []
    # Use rejection sampling for uniform sphere
    while len(pts) < n:
        # Sample in cube and reject outside sphere
        p = rng.uniform(-1, 1, size=(10, 3))
        norms = np.linalg.norm(p, axis=1)
        valid = p[norms <= 1.0]
        for v in valid:
            if len(pts) < n:
                pts.append(v * radius)
    return np.array(pts[:n])


def _generate_deformed(n, base, rng, scale=0.3):
    """Generate a deformed version of a base structure."""
    pos = base.copy()
    # Apply random deformation
    for i in range(len(pos)):
        pos[i] += rng.normal(0, scale, 3)
    return pos


def _polish(positions, kind):
    """High-precision polish pass with tighter tolerance."""
    try:
        # First, do a standard relax
        energy, pos = relax(positions, 'sutton_chen', kind)
        if not np.isfinite(energy):
            return None, np.inf
        
        # Then do additional refinement steps with tighter tolerances
        for _ in range(5):
            e, g = sutton_chen_energy_gradient(pos, kind)
            if not np.isfinite(e):
                break
            # Use gradient descent with small step
            lr = 0.001
            pos = pos - lr * g
            # Check validity
            if not _is_valid(pos):
                pos = pos + 0.01 * rng_global.normal(0, 1, pos.shape)
        
        # Final relax
        energy, pos = relax(pos, 'sutton_chen', kind)
        if np.isfinite(energy) and _is_valid(pos):
            return pos, energy
    except Exception:
        pass
    return None, np.inf


rng_global = np.random.default_rng(12345)


def construct(instance):
    n = instance["n"]
    kind = instance["kind"]
    
    best_energy = np.inf
    best_pos = None
    
    rng = np.random.default_rng(42 + n)
    
    # Base structures to try
    base_structures = []
    base_structures.append(_generate_fcc(n))
    base_structures.append(_generate_bcc(n))
    base_structures.append(_generate_hcp(n))
    base_structures.append(_generate_icosahedron(n))
    base_structures.append(_generate_random_compact(n, rng, radius=2.0))
    
    # Relax each base structure first to get better starting points
    relaxed_bases = []
    for base in base_structures:
        if _time_left() < 30.0:
            break
        try:
            energy, relaxed_pos = relax(base, 'sutton_chen', kind)
            if np.isfinite(energy) and _is_valid(relaxed_pos):
                relaxed_bases.append(relaxed_pos)
                if energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed_pos
        except Exception:
            continue
    
    # Now do multi-start from relaxed bases with various perturbations
    max_restarts = 200
    
    for i in range(max_restarts):
        if _time_left() < 20.0:
            break
        
        # Choose a base structure
        if relaxed_bases:
            base = relaxed_bases[i % len(relaxed_bases)].copy()
        else:
            base = base_structures[i % len(base_structures)].copy()
        
        # Vary noise level - use a range of scales
        noise_scale = 0.05 + 0.15 * ((i // 5) % 4)
        noise = rng.normal(0, noise_scale, base.shape)
        pos = base + noise
        
        # Relax
        try:
            energy, relaxed_pos = relax(pos, 'sutton_chen', kind)
            
            if not np.isfinite(energy):
                continue
                
            if _is_valid(relaxed_pos):
                if energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed_pos
        except Exception:
            continue
    
    # If we have time, do a polish pass on the best result
    if best_pos is not None and _time_left() > 30.0:
        polished_pos, polished_energy = _polish(best_pos, kind)
        if polished_pos is not None and polished_energy < best_energy:
            best_energy = polished_energy
            best_pos = polished_pos
    
    # If no valid position found, try more random starts
    if best_pos is None:
        for i in range(100):
            if _time_left() < 15.0:
                break
            pos = _generate_random_compact(n, rng, radius=2.5)
            try:
                energy, relaxed_pos = relax(pos, 'sutton_chen', kind)
                if np.isfinite(energy) and _is_valid(relaxed_pos):
                    if energy < best_energy:
                        best_energy = energy
                        best_pos = relaxed_pos
            except Exception:
                continue
    
    # Final fallback: return FCC structure
    if best_pos is None:
        best_pos = _generate_fcc(n)
    
    return best_pos

# EVOLVE-BLOCK-END
