"""Dzugutov cluster global minima: portfolio of structural heuristics + parallel relaxation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = None
_DEADLINE = 320.0  # safety margin before 330s kill


def _get_time_remaining():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _DEADLINE - (time.time() - _START_TIME)


def _generate_mackay_icosahedron(n, rng):
    """Generate a Mackay icosahedral cluster of n atoms."""
    # Icosahedral vertices and edges for shell generation
    # Use golden ratio
    phi = (1 + np.sqrt(5)) / 2
    
    # 12 vertices of icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])  # normalize to unit radius
    
    # Build positions shell by shell
    # Shell 0: center (1 atom)
    # Shell 1: 12 vertices of icosahedron
    # Shell 2: 42 atoms (12 vertices + 30 edge midpoints)
    # Shell 3: 92 atoms (12 vertices + 30 edges + 60 face centers + 30... actually 92)
    # Mackay shells: 1, 12, 42, 92, ...
    # Cumulative: 1, 13, 55, 147, ...
    
    # For n up to 60, we need shells up to shell 2 (55 atoms)
    # If n > 55, add extra atoms near the surface
    
    positions = [np.zeros(3)]  # center
    
    if n <= 1:
        return np.array(positions[:n])
    
    # Add icosahedron vertices
    for v in verts:
        if len(positions) >= n:
            break
        positions.append(v * 1.0)
    
    if n <= 13:
        return np.array(positions[:n])
    
    # Add edge midpoints (30 edges in icosahedron)
    # Icosahedron edges: connect vertices that are ~1.05 apart (in unit sphere)
    # Let's find all pairs with distance < 1.2 (in normalized coords)
    edges = []
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(verts[i] - verts[j])
            if d < 1.2:  # threshold for edges
                edges.append((i, j))
    
    for i, j in edges:
        if len(positions) >= n:
            break
        mid = (verts[i] + verts[j]) / 2
        positions.append(mid * 1.0)
    
    if n <= 55:
        return np.array(positions[:n])
    
    # For n > 55, add face centers and more
    # Icosahedron has 20 faces
    # Find faces (triples of vertices forming equilateral triangles)
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                d1 = np.linalg.norm(verts[i] - verts[j])
                d2 = np.linalg.norm(verts[j] - verts[k])
                d3 = np.linalg.norm(verts[i] - verts[k])
                if d1 < 1.2 and d2 < 1.2 and d3 < 1.2:
                    faces.append((i, j, k))
    
    for i, j, k in faces:
        if len(positions) >= n:
            break
        centroid = (verts[i] + verts[j] + verts[k]) / 3
        centroid = centroid / np.linalg.norm(centroid)  # normalize to unit sphere
        positions.append(centroid * 1.0)
    
    # If still not enough, add random points on/near the surface
    while len(positions) < n:
        # Pick a random direction and place at radius ~1.0
        p = rng.normal(size=3)
        p /= np.linalg.norm(p)
        positions.append(p * 1.0)
    
    return np.array(positions[:n])


def _generate_fcc(n, rng):
    """Generate an FCC lattice chunk roughly spherical."""
    # Equilibrium distance for Dzugutov is around 1.8-2.0
    a = 1.9  # lattice constant
    
    # Generate FCC points in a cube
    points = []
    # Estimate cube size
    # n atoms in FCC: density = 4/a^3 per unit cell
    # volume ~ n * a^3/4, so side ~ (n*a^3/4)^(1/3)
    side = int((n * a**3 / 4) ** (1/3)) + 2
    radius = side * a / 2
    
    for i in range(-side, side+1):
        for j in range(-side, side+1):
            for k in range(-side, side+1):
                # FCC basis: (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)
                for dx, dy, dz in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    if np.linalg.norm(p) <= radius:
                        points.append(p)
    
    if len(points) >= n:
        # Select n closest to center
        points = np.array(points)
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:n]
        return points[idx]
    else:
        # Add random points to fill
        points = np.array(points)
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                points = np.vstack([points, p])
        return points[:n]


def _generate_random_cluster(n, rng):
    """Generate a random compact cluster."""
    spacing = 1.1
    radius = spacing * n ** (1.0 / 3.0) * 0.75
    points = []
    max_attempts = 10000
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            min_dist = min([np.linalg.norm(p - q) for q in points], default=float('inf'))
            if min_dist >= 0.5:
                points.append(p)
        attempts += 1
    
    # If we couldn't place all atoms, add them anyway (may violate min distance but relax will fix)
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points)


def _relax_worker(args):
    """Worker function for parallel relaxation."""
    positions, potential, parameter = args
    try:
        energy, relaxed = relax(positions, potential, parameter)
        return energy, relaxed
    except Exception:
        return float('inf'), positions


def construct(instance):
    """Portfolio of structural heuristics with parallel relaxation and deadline management."""
    global _START_TIME
    _START_TIME = time.time()
    
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Generate diverse starting structures
    rng = np.random.default_rng(size)
    
    generators = [
        _generate_mackay_icosahedron,
        _generate_fcc,
        _generate_random_cluster,
    ]
    
    # Create initial structures
    initial_structures = []
    for gen in generators:
        try:
            pos = gen(size, rng)
            if len(pos) == size:
                initial_structures.append(pos)
        except Exception:
            pass
    
    # Add a few more random variants
    for _ in range(2):
        try:
            pos = _generate_random_cluster(size, rng)
            if len(pos) == size:
                initial_structures.append(pos)
        except Exception:
            pass
    
    if not initial_structures:
        # Fallback: pure random
        pos = _generate_random_cluster(size, rng)
        initial_structures = [pos]
    
    # Parallel relaxation with deadline checking
    best_energy = float('inf')
    best_positions = initial_structures[0]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for pos in initial_structures:
            if _get_time_remaining() < 30:  # need at least 30s for relaxation
                break
            futures.append(executor.submit(_relax_worker, (pos, potential, parameter)))
        
        for future in concurrent.futures.as_completed(futures):
            if _get_time_remaining() < 10:
                break
            try:
                energy, positions = future.result(timeout=60)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except Exception:
                pass
    
    # If we have time, try some perturbation + re-relaxation
    if _get_time_remaining() > 60:
        for _ in range(3):
            if _get_time_remaining() < 30:
                break
            # Perturb the best structure
            perturbed = best_positions + rng.normal(0, 0.3, size=(size, 3))
            try:
                energy, positions = relax(perturbed, potential, parameter)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except Exception:
                pass
    
    return best_positions
# EVOLVE-BLOCK-END
