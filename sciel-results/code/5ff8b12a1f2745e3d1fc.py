"""Program for Sutton-Chen TM cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution
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


def _generate_icosahedron(n):
    """Generate an icosahedral cluster with Mackay shells."""
    # Icosahedral directions
    phi = (1 + np.sqrt(5)) / 2
    # 12 vertices of icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])
    
    # 30 edge midpoints (12-6 shell)
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if 1.0 < d < 1.5:  # approximate edge length
                edges.append((verts[i] + verts[j]) / 2)
    edges = np.array(edges)
    if len(edges) > 0:
        edges /= np.linalg.norm(edges[0])
    
    # 20 face centers
    faces = []
    # For a regular icosahedron, we can approximate face centers
    # This is complex, so let's just use the vertices and edges
    
    pts = [np.zeros(3)]  # center
    for v in verts:
        pts.append(v)
    for e in edges:
        pts.append(e)
    
    # If we need more, add second shell (scaled)
    while len(pts) < n:
        for v in verts:
            if len(pts) >= n:
                break
            pts.append(v * 2.0)
        for e in edges:
            if len(pts) >= n:
                break
            pts.append(e * 2.0)
    
    return np.array(pts[:n])


def _generate_random_compact(n, rng, radius=2.5):
    """Generate n random points within a sphere, compacted."""
    pts = []
    for _ in range(n):
        while True:
            p = rng.normal(0, 1, 3)
            norm = np.linalg.norm(p)
            if norm <= 1.0:
                break
        pts.append(p * radius)
    return np.array(pts)


def _flatten(positions):
    """Flatten (n,3) array to (3n,) for DE."""
    return positions.flatten()


def _unflatten(x, n):
    """Unflatten (3n,) array to (n,3)."""
    return x.reshape(n, 3)


def _de_energy_wrapper(x, n, kind):
    """Wrapper for DE: compute energy, penalize invalid configurations."""
    pos = _unflatten(x, n)
    if not np.all(np.isfinite(pos)):
        return 1e10
    if not _is_valid(pos):
        return 1e10
    try:
        energy, _ = sutton_chen_energy_gradient(pos, kind)
        if not np.isfinite(energy):
            return 1e10
        return energy
    except Exception:
        return 1e10


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
    base_structures.append(_generate_icosahedron(n))
    base_structures.append(_generate_random_compact(n, rng, radius=2.5))
    
    # First pass: relax each base structure
    for base in base_structures:
        if _time_left() < 20.0:
            break
        try:
            energy, relaxed_pos = relax(base, 'sutton_chen', kind)
            if np.isfinite(energy) and _is_valid(relaxed_pos):
                if energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed_pos
        except Exception:
            continue
    
    # Second pass: Differential Evolution
    if _time_left() > 60.0:
        # Determine bounds based on cluster size
        radius = 1.5 * n ** (1/3)
        bounds = [(-radius, radius)] * (3 * n)
        
        # Use the best base structure as initial population hint
        pop_size = min(20, max(10, n // 3))
        
        try:
            de_result = differential_evolution(
                _de_energy_wrapper,
                bounds,
                args=(n, kind),
                popsize=pop_size,
                maxiter=30,
                tol=1e-4,
                mutation=(0.3, 1.0),
                recombination=0.8,
                seed=42 + n,
                polish=False,  # We'll do our own polish
                init='latinhypercube',
                x0=_flatten(base_structures[0])
            )
            
            if de_result.success and de_result.fun < 1e9:
                de_pos = _unflatten(de_result.x, n)
                # Polish with local relaxation
                try:
                    energy, relaxed_pos = relax(de_pos, 'sutton_chen', kind)
                    if np.isfinite(energy) and _is_valid(relaxed_pos):
                        if energy < best_energy:
                            best_energy = energy
                            best_pos = relaxed_pos
                except Exception:
                    pass
        except Exception:
            pass
    
    # Third pass: Multiple restarts with noise on best structure
    if best_pos is not None and _time_left() > 30.0:
        max_restarts = 50
        for i in range(max_restarts):
            if _time_left() < 15.0:
                break
            
            noise_scale = 0.05 + 0.1 * (i % 3)
            noise = rng.normal(0, noise_scale, best_pos.shape)
            pos = best_pos + noise
            
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
