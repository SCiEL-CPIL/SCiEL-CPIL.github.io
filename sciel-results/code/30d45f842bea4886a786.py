"""Rewritten program for Dzugutov cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient


# Dzugutov potential parameters
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16


def _dzugutov_energy(positions):
    """Compute total Dzugutov energy for a set of positions."""
    n = len(positions)
    if n < 2:
        return 0.0
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            if r < 1e-10:
                continue
            if r < a:
                v = (A * (r ** (-m) - B) * np.exp(c / (r - a)))
            elif r < b:
                v = B * np.exp(d / (r - b))
            else:
                v = 0.0
            total += v
    return total


def _get_icosahedron_vertices():
    """Return the 12 vertices of a regular icosahedron centered at origin."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    # Normalize to unit radius
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, None]
    return verts


def _get_dodecahedron_vertices():
    """Return the 20 vertices of a regular dodecahedron centered at origin."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [0, 1/phi, phi], [0, 1/phi, -phi], [0, -1/phi, phi], [0, -1/phi, -phi],
        [1/phi, phi, 0], [1/phi, -phi, 0], [-1/phi, phi, 0], [-1/phi, -phi, 0],
        [phi, 0, 1/phi], [phi, 0, -1/phi], [-phi, 0, 1/phi], [-phi, 0, -1/phi]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, None]
    return verts


def _generate_icosahedral_cluster(n, spacing=1.1):
    """
    Generate an icosahedral cluster with n atoms.
    Shell structure:
      - Core: 1 atom at center
      - Shell 1 (12 atoms): icosahedron vertices at radius r1
      - Shell 2 (42 atoms): second shell at radius r2
      - Shell 3 (92 atoms): third shell at radius r3
    """
    icosa = _get_icosahedron_vertices()
    dodeca = _get_dodecahedron_vertices()
    
    # Radii for shells (tuned for Dzugutov potential)
    r1 = spacing * 0.95
    r2 = spacing * 1.65
    r3 = spacing * 2.35
    
    positions = [np.zeros(3)]  # Core atom
    
    # Shell 1: 12 atoms at icosahedron vertices
    if n > 1:
        shell1 = icosa * r1
        for v in shell1:
            if len(positions) < n:
                positions.append(v.copy())
    
    # Shell 2: 42 atoms (mix of dodecahedron and other positions)
    if n > 13:
        shell2 = dodeca * r2
        # Add additional points to fill up to 42
        # Use icosahedron vertices scaled and rotated
        for v in shell2:
            if len(positions) < n:
                positions.append(v.copy())
        # Add more if needed
        for v in icosa * r2 * 1.1:
            if len(positions) < n:
                positions.append(v.copy())
    
    # Shell 3: fill remaining
    if n > 55:
        shell3 = dodeca * r3
        for v in shell3:
            if len(positions) < n:
                positions.append(v.copy())
        for v in icosa * r3 * 1.1:
            if len(positions) < n:
                positions.append(v.copy())
    
    # If still not enough, add random points on sphere
    if len(positions) < n:
        rng = np.random.default_rng(42)
        while len(positions) < n:
            p = rng.uniform(-1, 1, 3)
            p = p / np.linalg.norm(p) * r3 * 1.2
            positions.append(p)
    
    return np.array(positions[:n])


def _generate_fcc_cluster(n, spacing=1.1):
    """Generate an FCC-like cluster with n atoms."""
    # FCC lattice points
    lattice = []
    # Generate FCC points in a cubic region
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                # FCC basis
                lattice.append(np.array([i, j, k]) * spacing)
                lattice.append((np.array([i, j, k]) + 0.5 * np.array([1, 1, 0])) * spacing)
                lattice.append((np.array([i, j, k]) + 0.5 * np.array([1, 0, 1])) * spacing)
                lattice.append((np.array([i, j, k]) + 0.5 * np.array([0, 1, 1])) * spacing)
    
    lattice = np.array(lattice)
    
    # Center the cluster
    center = np.mean(lattice, axis=0)
    lattice = lattice - center
    
    # Select n closest points to center
    dists = np.linalg.norm(lattice, axis=1)
    order = np.argsort(dists)
    selected = lattice[order[:n]]
    
    return selected


def _generate_random_compact(n, spacing=1.1, seed=0):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.75
    points = []
    min_dist = 0.5 * spacing
    
    attempts = 0
    while len(points) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    
    # If not enough points, fill with random
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points[:n])


def _relax_with_retry(positions, potential, parameter, max_restarts=3):
    """Relax positions with multiple restarts if needed."""
    best_energy = np.inf
    best_positions = positions.copy()
    
    for restart in range(max_restarts):
        try:
            energy, relaxed = relax(positions.copy(), potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed
        except Exception:
            pass
    
    return best_energy, best_positions


def construct(instance):
    """Find low-energy Dzugutov cluster structures using diverse initial guesses."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    deadline = start_time + 280  # Safety margin
    
    # Generate diverse initial structures
    initial_structures = []
    
    # 1. Icosahedral structure
    try:
        icos = _generate_icosahedral_cluster(size, spacing=1.1)
        initial_structures.append(icos)
    except Exception:
        pass
    
    # 2. FCC-like structure
    try:
        fcc = _generate_fcc_cluster(size, spacing=1.1)
        initial_structures.append(fcc)
    except Exception:
        pass
    
    # 3. Random compact structures with different seeds
    for seed in [1, 2, 3, 4, 5]:
        try:
            rand = _generate_random_compact(size, spacing=1.1, seed=seed)
            initial_structures.append(rand)
        except Exception:
            pass
    
    # 4. Icosahedral with slightly different spacing
    try:
        icos2 = _generate_icosahedral_cluster(size, spacing=1.05)
        initial_structures.append(icos2)
    except Exception:
        pass
    
    try:
        icos3 = _generate_icosahedral_cluster(size, spacing=1.15)
        initial_structures.append(icos3)
    except Exception:
        pass
    
    # Relax each initial structure and keep the best
    best_energy = np.inf
    best_positions = None
    
    for i, init_pos in enumerate(initial_structures):
        if time.time() > deadline - 30:  # Stop if we're running out of time
            break
        
        # Add small perturbation to avoid exact duplicates
        if i > 0:
            rng = np.random.default_rng(i)
            perturbed = init_pos + rng.normal(0, 0.05, size=(size, 3))
        else:
            perturbed = init_pos
        
        try:
            energy, relaxed = relax(perturbed, potential, parameter)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed
        except Exception:
            pass
    
    # If we still have time, try more perturbations of the best structure
    if best_positions is not None and time.time() < deadline - 60:
        for perturb_seed in range(10):
            if time.time() > deadline - 30:
                break
            
            rng = np.random.default_rng(100 + perturb_seed)
            perturbed = best_positions + rng.normal(0, 0.1, size=(size, 3))
            
            try:
                energy, relaxed = relax(perturbed, potential, parameter)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed
            except Exception:
                pass
    
    # Final fallback if no valid result
    if best_positions is None:
        best_positions = _generate_random_compact(size, spacing=1.1, seed=99)
        try:
            _, best_positions = relax(best_positions, potential, parameter)
        except Exception:
            pass
    
    return best_positions
# EVOLVE-BLOCK-END
