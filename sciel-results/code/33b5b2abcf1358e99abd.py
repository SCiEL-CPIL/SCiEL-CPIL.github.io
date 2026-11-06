"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = time.time()
TIME_LIMIT = 285.0  # Leave 15s safety margin for 330s limit


def _check_time():
    return time.time() - _START_TIME < TIME_LIMIT


def _get_icosahedral_points():
    """
    Returns the 12 vertices, 20 face centers, and 30 edge midpoints of a regular icosahedron.
    All points are normalized to unit radius.
    """
    phi = (1 + np.sqrt(5)) / 2
    
    # 12 Vertices of regular icosahedron
    v = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    
    # Normalize to unit radius
    norm = np.linalg.norm(v[0])
    v = v / norm
    
    # Find edges: pairs of vertices with minimum distance
    dists = np.linalg.norm(v[:, np.newaxis, :] - v[np.newaxis, :, :], axis=2)
    min_dist = np.min(dists)
    
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if np.isclose(dists[i, j], min_dist):
                edges.append((i, j))
    
    # Face centers: triangles of edges
    face_centers = []
    seen_faces = set()
    for i, j in edges:
        for k in range(12):
            if k == i or k == j:
                continue
            edge_ik = (i, k) in edges or (k, i) in edges
            edge_jk = (j, k) in edges or (k, j) in edges
            if edge_ik and edge_jk:
                face_key = tuple(sorted([i, j, k]))
                if face_key not in seen_faces:
                    seen_faces.add(face_key)
                    center = (v[i] + v[j] + v[k]) / 3.0
                    center = center / np.linalg.norm(center)
                    face_centers.append(center)
    
    # Edge midpoints
    edge_centers = []
    for i, j in edges:
        center = (v[i] + v[j]) / 2.0
        center = center / np.linalg.norm(center)
        edge_centers.append(center)
    
    return v, np.array(face_centers), np.array(edge_centers)


# Precompute the special points
_V, _F, _E = _get_icosahedral_points()


def _generate_icosahedral(n, rng, variant=0, scale=1.0):
    """
    Generate n points with icosahedral symmetry.
    variant 0: Center + 12 vertices + structured outer (face/edge centers)
    variant 1: Center + 12 vertices + random outer shell
    variant 2: Random compact with minimum spacing
    """
    # Equilibrium nearest-neighbor distance for Dzugutov ~1.0-1.1
    r1 = 1.05 * scale
    
    points = [np.array([0.0, 0.0, 0.0])]
    shell1 = _V * r1
    points.extend(shell1.tolist())
    
    if n <= len(points):
        return np.array(points[:n])
    
    needed = n - len(points)
    
    if variant == 0:
        # Structured outer shell using face centers and edge midpoints
        # In Mackay icosahedron, second shell radius is larger
        r2 = r1 * 1.55
        
        # Pool: face centers (20) + edge midpoints (30) = 50 candidates
        pool = np.vstack([_F, _E])
        idx = rng.permutation(len(pool))
        selected = pool[idx[:needed]] * r2
        for p in selected:
            points.append(p.tolist())
            
    elif variant == 1:
        # Random points on sphere of radius r2
        r2 = r1 * 1.5
        for _ in range(needed):
            direction = rng.normal(size=3)
            direction = direction / np.linalg.norm(direction)
            p = direction * r2
            points.append(p.tolist())
            
    else:
        # Random compact with minimum spacing 0.55
        radius = 1.2 * (n / (4.0 / 3.0 * np.pi)) ** (1.0 / 3.0)
        pts = []
        attempts = 0
        while len(pts) < n and attempts < 5000:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                ok = True
                for q in pts:
                    if np.linalg.norm(p - np.array(q)) < 0.55:
                        ok = False
                        break
                if ok:
                    pts.append(p.tolist())
            attempts += 1
        # Fallback if not enough points
        while len(pts) < n:
            p = rng.uniform(-radius, radius, 3)
            pts.append(p.tolist())
        return np.array(pts)
    
    return np.array(points)


def _relax_with_fallback(positions, potential, parameter):
    """Relax with error handling."""
    try:
        energy, relaxed = relax(positions, potential, parameter)
        return energy, relaxed
    except Exception:
        return np.inf, positions


def construct(instance):
    """
    Find low-energy Dzugutov cluster using icosahedral starting points
    with multiple structured variants and seeds.
    """
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    best_energy = np.inf
    best_positions = None
    
    # Multiple seeds for diversity
    seeds = [42, 123, 456, 789, 101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111]
    
    for seed in seeds:
        if not _check_time():
            break
            
        rng = np.random.default_rng(seed)
        
        # Variant 0: Structured Icosahedral (best aligned with Dzugutov preference)
        try:
            pos = _generate_icosahedral(size, rng, variant=0)
            if len(pos) == size:
                energy, positions = _relax_with_fallback(pos, potential, parameter)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
        except Exception:
            pass
        
        # Variant 1: Random Icosahedral Shell
        if _check_time():
            try:
                pos = _generate_icosahedral(size, rng, variant=1)
                if len(pos) == size:
                    energy, positions = _relax_with_fallback(pos, potential, parameter)
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = positions
            except Exception:
                pass
        
        # Variant 2: Random Compact (diversity)
        if _check_time() and (best_energy > 0 or seed < 3):
            try:
                pos = _generate_icosahedral(size, rng, variant=2)
                if len(pos) == size:
                    energy, positions = _relax_with_fallback(pos, potential, parameter)
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = positions
            except Exception:
                pass
    
    # Fallback if no valid result
    if best_positions is None:
        rng = np.random.default_rng(size)
        best_positions = rng.uniform(-1, 1, (size, 3))
        try:
            _, best_positions = relax(best_positions, potential, parameter)
        except Exception:
            pass
    
    return best_positions
# EVOLVE-BLOCK-END
