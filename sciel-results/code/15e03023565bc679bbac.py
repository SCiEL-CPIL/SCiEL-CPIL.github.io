"""Combined program: FCC lattice fragments + multi-start relaxation for SC clusters."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _time_left(start_time, deadline=290.0):
    return deadline - (time.time() - start_time)


def _fcc_shells(n, spacing=1.0):
    """Generate n points from FCC lattice shells, centered."""
    if n == 0:
        return np.zeros((0, 3))
    
    points = []
    max_coord = int(np.ceil(n ** (1/3) * 2)) + 2
    
    for ix in range(-max_coord, max_coord + 1):
        for iy in range(-max_coord, max_coord + 1):
            for iz in range(-max_coord, max_coord + 1):
                for bx, by, bz in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    x = (ix + bx) * spacing
                    y = (iy + by) * spacing
                    z = (iz + bz) * spacing
                    if x*x + y*y + z*z <= (n ** (1/3) * spacing * 1.2) ** 2:
                        points.append([x, y, z])
    
    points = np.array(points)
    if len(points) > n:
        dists = np.sum(points**2, axis=1)
        idx = np.argsort(dists)[:n]
        points = points[idx]
    elif len(points) < n:
        rng = np.random.default_rng(42)
        while len(points) < n:
            p = rng.uniform(-n**(1/3)*spacing, n**(1/3)*spacing, 3)
            points = np.vstack([points, [p]])
    
    points -= np.mean(points, axis=0)
    return points


def _hcp_shells(n, spacing=1.0):
    """Generate n points from HCP lattice shells."""
    if n == 0:
        return np.zeros((0, 3))
    
    points = []
    a = spacing
    c = np.sqrt(8/3) * a
    max_coord = int(np.ceil(n ** (1/3) * 2)) + 2
    
    for ix in range(-max_coord, max_coord + 1):
        for iy in range(-max_coord, max_coord + 1):
            for iz in range(-max_coord, max_coord + 1):
                if iz % 2 == 0:
                    x = ix * a + 0.5 * iy * a * 0.5
                    y = iy * a * np.sqrt(3)/2
                else:
                    x = ix * a + 0.5 * a + 0.5 * iy * a * 0.5
                    y = iy * a * np.sqrt(3)/2
                z = iz * c / 2
                if x*x + y*y + z*z <= (n ** (1/3) * spacing * 1.2) ** 2:
                    points.append([x, y, z])
    
    points = np.array(points)
    if len(points) > n:
        dists = np.sum(points**2, axis=1)
        idx = np.argsort(dists)[:n]
        points = points[idx]
    elif len(points) < n:
        rng = np.random.default_rng(42)
        while len(points) < n:
            p = rng.uniform(-n**(1/3)*spacing, n**(1/3)*spacing, 3)
            points = np.vstack([points, [p]])
    
    points -= np.mean(points, axis=0)
    return points


def _icosahedron_based(n, spacing=1.0):
    """Generate points based on icosahedral symmetry with nested shells."""
    phi = (1 + 5 ** 0.5) / 2
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    vertices = vertices / np.linalg.norm(vertices[0]) * spacing
    
    points = [[0.0, 0.0, 0.0]]
    points.extend([v.tolist() for v in vertices])
    
    if n > 13:
        dodec = np.array([
            [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
            [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
            [0, phi, 1/phi], [0, phi, -1/phi], [0, -phi, 1/phi], [0, -phi, -1/phi],
            [1/phi, 0, phi], [1/phi, 0, -phi], [-1/phi, 0, phi], [-1/phi, 0, -phi],
            [phi, 1/phi, 0], [phi, -1/phi, 0], [-phi, 1/phi, 0], [-phi, -1/phi, 0]
        ])
        dodec = dodec / np.linalg.norm(dodec[0]) * spacing * 1.8
        for v in dodec:
            if len(points) >= n:
                break
            points.append(v.tolist())
    
    rng = np.random.default_rng(42)
    while len(points) < n:
        base = points[rng.integers(0, len(points))]
        p = np.array(base) + rng.normal(0, 0.3 * spacing, 3)
        points.append(p.tolist())
    
    points = np.array(points[:n])
    points -= np.mean(points, axis=0)
    return points


def _random_compact(n, spacing=1.0, seed=None):
    """Generate a random compact cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.9
    min_dist = 0.5 * spacing
    
    points = []
    attempts = 0
    max_attempts = n * 200
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - np.array(q)) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    points = np.array(points)
    points -= np.mean(points, axis=0)
    return points


def _perturb(positions, sigma=0.1, seed=None):
    """Add small random perturbations to positions."""
    rng = np.random.default_rng(seed)
    return positions + rng.normal(0, sigma, positions.shape)


def _check_validity(positions, min_dist=0.35):
    """Check if positions are valid (finite and no atoms too close)."""
    if not np.all(np.isfinite(positions)):
        return False
    n = len(positions)
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(positions[i] - positions[j]) < min_dist:
                return False
    return True


def construct(instance):
    """Multi-start relaxation with diverse initial structures for SC clusters."""
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))
    
    start_time = time.time()
    deadline = 290.0
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse starting structures
    starters = []
    
    # FCC-like (most important for SC clusters)
    starters.append(_fcc_shells(size, spacing=1.0))
    
    # HCP-like
    starters.append(_hcp_shells(size, spacing=1.0))
    
    # Icosahedron-based
    starters.append(_icosahedron_based(size, spacing=1.0))
    
    # Multiple random compact structures with different seeds
    for seed in [size, size + 1, size + 2, size + 3, size + 4, size + 5]:
        starters.append(_random_compact(size, spacing=1.0, seed=seed))
    
    # Relax each starter
    for i, starter in enumerate(starters):
        if _time_left(start_time, deadline) < 10.0:
            break
        
        try:
            energy, positions = relax(starter, potential, kind)
            
            if np.isfinite(energy) and energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            continue
    
    # If we have time, try additional random starts and perturbations
    if best_positions is not None:
        rng = np.random.default_rng(size + 100)
        
        # Try perturbing the best solution
        for attempt in range(3):
            if _time_left(start_time, deadline) < 15.0:
                break
            
            perturbed = _perturb(best_positions, sigma=0.05, seed=size + 100 + attempt)
            try:
                energy, positions = relax(perturbed, potential, kind)
                
                if np.isfinite(energy) and energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()
            except Exception:
                continue
        
        # Try additional random starts
        for attempt in range(5):
            if _time_left(start_time, deadline) < 10.0:
                break
            
            starter = _random_compact(size, spacing=1.0, seed=size + 200 + attempt)
            try:
                energy, positions = relax(starter, potential, kind)
                
                if np.isfinite(energy) and energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()
            except Exception:
                continue
    
    # Fallback: return a simple valid configuration if nothing worked
    if best_positions is None:
        best_positions = _random_compact(size, spacing=1.0, seed=size)
    
    # Final validity check
    if not _check_validity(best_positions):
        # Try to fix by spreading out atoms
        rng = np.random.default_rng(size + 500)
        for _ in range(10):
            fixed = best_positions + rng.normal(0, 0.1, best_positions.shape)
            if _check_validity(fixed):
                best_positions = fixed
                break
    
    return best_positions
# EVOLVE-BLOCK-END
