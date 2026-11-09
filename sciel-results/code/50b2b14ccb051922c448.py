"""Dzugutov cluster global minima via GA with polytetrahedral templates and local relaxation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = time.time()
_HARD_LIMIT = 310.0  # safety margin before 330s kill

A_DZ = 5.82
B_DZ = 1.28
a_DZ = 1.87
b_DZ = 1.94
c_DZ = 1.10
d_DZ = 0.27
m_DZ = 16


def _elapsed():
    return time.time() - _START_TIME


def _remaining():
    return _HARD_LIMIT - _elapsed()


def _dzugutov_energy(positions):
    """Compute Dzugutov potential energy using vectorized operations."""
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(r, np.inf)
    E = 0.0
    mask1 = r < a_DZ
    mask2 = (r >= a_DZ) & (r < b_DZ)
    if np.any(mask1):
        r1 = r[mask1]
        E += np.sum(A_DZ * (r1 ** (-m_DZ) - B_DZ) * np.exp(c_DZ / (r1 - a_DZ)))
    if np.any(mask2):
        r2 = r[mask2]
        E += np.sum(B_DZ * np.exp(d_DZ / (r2 - b_DZ)))
    return float(E)


def _dzugutov_gradient(positions):
    """Compute gradient of Dzugutov potential energy."""
    n = len(positions)
    grad = np.zeros_like(positions)
    if n < 2:
        return grad
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(r, np.inf)
    mask1 = r < a_DZ
    mask2 = (r >= a_DZ) & (r < b_DZ)
    if np.any(mask1):
        r1 = r[mask1]
        dr1 = c_DZ / (r1 - a_DZ) ** 2
        dV1 = A_DZ * (-m_DZ * r1 ** (-m_DZ - 1) - B_DZ * dr1) * np.exp(c_DZ / (r1 - a_DZ))
        # dV/dr, then grad_i = -sum_{j!=i} dV/dr * (r_i - r_j)/r
        force_mag = dV1 * diff[mask1] / r1[:, np.newaxis]
        for i in range(n):
            for j in range(n):
                if i != j and mask1[i, j]:
                    grad[i] -= force_mag[i, j]
    if np.any(mask2):
        r2 = r[mask2]
        dr2 = -d_DZ / (r2 - b_DZ) ** 2
        dV2 = B_DZ * dr2 * np.exp(d_DZ / (r2 - b_DZ))
        force_mag = dV2 * diff[mask2] / r2[:, np.newaxis]
        for i in range(n):
            for j in range(n):
                if i != j and mask2[i, j]:
                    grad[i] -= force_mag[i, j]
    return grad


def _icosahedron_vertices(scale=1.0):
    """Return 12 vertices of a regular icosahedron scaled so edge length ~ scale."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    d = np.linalg.norm(verts[0] - verts[4])
    verts = verts * (scale / d)
    return verts


def _face_centers(vertices):
    """Return centers of all triangular faces of an icosahedron."""
    faces = [
        (0, 1, 4), (0, 1, 5), (0, 4, 8), (0, 8, 5), (0, 2, 6),
        (0, 2, 7), (0, 6, 9), (0, 9, 7), (1, 4, 10), (1, 10, 5),
        (1, 5, 10), (2, 6, 11), (2, 11, 7), (2, 7, 11), (4, 8, 3),
        (4, 3, 10), (5, 8, 3), (5, 3, 10), (6, 9, 11), (6, 11, 3),
        (7, 9, 11), (7, 11, 3), (9, 11, 3), (10, 3, 8)
    ]
    centers = []
    for f in faces:
        c = (vertices[f[0]] + vertices[f[1]] + vertices[f[2]]) / 3.0
        centers.append(c)
    return np.array(centers)


def _generate_mackay_icosahedron(n, base_spacing=1.1):
    """Generate a Mackay-like icosahedral cluster with n atoms."""
    ico = _icosahedron_vertices(base_spacing)
    points = [ico]
    
    if n <= 12:
        return ico[:n].copy()
    
    face_centers = _face_centers(ico)
    points.append(face_centers)
    
    all_points = np.vstack(points)
    if len(all_points) >= n:
        return all_points[:n].copy()
    
    # Third shell: edge midpoints
    edges = [
        (0, 1), (0, 4), (0, 5), (0, 2), (0, 6), (0, 7),
        (1, 4), (1, 5), (1, 10), (2, 6), (2, 7), (2, 11),
        (3, 8), (3, 9), (3, 10), (3, 11), (4, 8), (4, 10),
        (5, 8), (5, 10), (6, 9), (6, 11), (7, 9), (7, 11),
        (8, 9), (8, 10), (9, 11), (10, 11)
    ]
    edge_points = []
    for e in edges:
        mid = (ico[e[0]] + ico[e[1]]) / 2.0
        edge_points.append(mid)
    edge_points = np.array(edge_points)
    all_points = np.vstack([all_points, edge_points])
    
    if len(all_points) >= n:
        return all_points[:n].copy()
    
    # Fourth shell: outer icosahedron
    outer_ico = _icosahedron_vertices(base_spacing * 1.7)
    all_points = np.vstack([all_points, outer_ico])
    
    if len(all_points) >= n:
        return all_points[:n].copy()
    
    # Fallback: add random compact points
    radius = base_spacing * n ** (1.0/3.0) * 0.8
    rng = np.random.default_rng(n)
    while len(all_points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            all_points = np.vstack([all_points, p])
    
    return all_points[:n].copy()


def _generate_decahedral(n, base_spacing=1.1):
    """Generate a decahedral (5-fold symmetric) cluster."""
    # Start with icosahedron but distort toward decahedral
    ico = _icosahedron_vertices(base_spacing)
    # Apply slight distortion
    distorted = ico.copy()
    for i in range(12):
        distorted[i] *= 1.0 + 0.05 * np.sin(i * 0.5)
    
    face_centers = _face_centers(distorted)
    all_points = np.vstack([distorted, face_centers])
    
    if len(all_points) >= n:
        return all_points[:n].copy()
    
    edges = [
        (0, 1), (0, 4), (0, 5), (0, 2), (0, 6), (0, 7),
        (1, 4), (1, 5), (1, 10), (2, 6), (2, 7), (2, 11),
        (3, 8), (3, 9), (3, 10), (3, 11), (4, 8), (4, 10),
        (5, 8), (5, 10), (6, 9), (6, 11), (7, 9), (7, 11),
        (8, 9), (8, 10), (9, 11), (10, 11)
    ]
    edge_points = []
    for e in edges:
        mid = (distorted[e[0]] + distorted[e[1]]) / 2.0
        edge_points.append(mid)
    edge_points = np.array(edge_points)
    all_points = np.vstack([all_points, edge_points])
    
    if len(all_points) >= n:
        return all_points[:n].copy()
    
    radius = base_spacing * n ** (1.0/3.0) * 0.8
    rng = np.random.default_rng(n + 100)
    while len(all_points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            all_points = np.vstack([all_points, p])
    
    return all_points[:n].copy()


def _generate_compact_random(n, seed=42):
    """Generate a compact random cluster."""
    rng = np.random.default_rng(seed)
    radius = 1.1 * n ** (1.0/3.0) * 0.75
    points = []
    for _ in range(200):
        if len(points) >= n:
            break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            points.append(p)
    points = np.array(points)
    if len(points) < n:
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points = np.vstack([points, p])
    return points[:n].copy()


def _generate_fcc(n, spacing=1.1):
    """Generate an FCC lattice fragment."""
    a = spacing
    points = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                # FCC basis
                for bx, by, bz in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    p = np.array([i*a + bx*a, j*a + by*a, k*a + bz*a])
                    if np.linalg.norm(p) < a * n**(1/3) * 0.8:
                        points.append(p)
    points = np.array(points)
    if len(points) < n:
        rng = np.random.default_rng(n)
        radius = a * n ** (1.0/3.0) * 0.8
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points = np.vstack([points, p])
    return points[:n].copy()


def _relax_with_timeout(positions, potential, parameter, timeout=15.0):
    """Relax positions with a timeout guard."""
    start = time.time()
    try:
        energy, pos = relax(positions.copy(), potential, parameter)
        return energy, pos
    except Exception:
        pass
    # Fallback: gradient descent
    pos = positions.copy()
    lr = 0.01
    for step in range(200):
        if time.time() - start > timeout:
            break
        try:
            grad = dzugutov_energy_gradient(pos)
        except Exception:
            break
        norm = np.linalg.norm(grad)
        if norm < 1e-10:
            break
        pos = pos - lr * grad / (norm + 1e-12) * 0.1
        lr = max(0.001, lr * 0.99)
    energy = _dzugutov_energy(pos)
    return energy, pos


def _blend_crossover(parent1, parent2, rng):
    """Blend crossover: child = alpha*p1 + (1-alpha)*p2 with per-atom alpha."""
    n = len(parent1)
    alphas = rng.uniform(0.2, 0.8, n)[:, np.newaxis]
    child = alphas * parent1 + (1.0 - alphas) * parent2
    return child


def _mutate(positions, rng, sigma=0.1):
    """Gaussian mutation."""
    return positions + rng.normal(0, sigma, positions.shape)


def _ga_search(n, potential, parameter, pop_size=8, generations=15, time_budget=40.0):
    """Genetic algorithm with polytetrahedral templates."""
    start_time = time.time()
    
    # Initialize population with diverse templates
    templates = []
    templates.append(_generate_mackay_icosahedron(n, base_spacing=1.1))
    templates.append(_generate_mackay_icosahedron(n, base_spacing=1.15))
    templates.append(_generate_mackay_icosahedron(n, base_spacing=1.05))
    templates.append(_generate_decahedral(n, base_spacing=1.1))
    templates.append(_generate_compact_random(n, seed=42))
    templates.append(_generate_compact_random(n, seed=123))
    templates.append(_generate_fcc(n, spacing=1.1))
    templates.append(_generate_mackay_icosahedron(n, base_spacing=1.2))
    
    # Relax all templates and evaluate
    population = []
    energies = []
    for t in templates:
        if time.time() - start_time > time_budget * 0.5:
            break
        try:
            e, pos = _relax_with_timeout(t, potential, parameter, timeout=8.0)
            population.append(pos)
            energies.append(e)
        except Exception:
            population.append(t)
            energies.append(_dzugutov_energy(t))
    
    if not population:
        return templates[0], _dzugutov_energy(templates[0])
    
    best_idx = np.argmin(energies)
    best_pos = population[best_idx].copy()
    best_energy = energies[best_idx]
    
    # GA evolution
    rng = np.random.default_rng(n)
    for gen in range(generations):
        if time.time() - start_time > time_budget:
            break
        
        # Selection: keep top half
        sorted_idx = np.argsort(energies)
        keep = sorted_idx[:pop_size // 2]
        parents = [population[i] for i in keep]
        parent_energies = [energies[i] for i in keep]
        
        # Generate offspring
        new_population = []
        new_energies = []
        
        # Elitism: keep best
        new_population.append(population[sorted_idx[0]])
        new_energies.append(energies[sorted_idx[0]])
        
        for _ in range(pop_size - 1):
            if time.time() - start_time > time_budget:
                break
            
            # Crossover
            i = rng.randint(0, len(parents))
            j = rng.randint(0, len(parents))
            if i == j:
                j = (j + 1) % len(parents)
            child = _blend_crossover(parents[i], parents[j], rng)
            
            # Mutation
            if rng.random() < 0.5:
                child = _mutate(child, rng, sigma=0.08)
            
            # Relax
            try:
                e, pos = _relax_with_timeout(child, potential, parameter, timeout=5.0)
                new_population.append(pos)
                new_energies.append(e)
            except Exception:
                new_population.append(child)
                new_energies.append(_dzugutov_energy(child))
        
        population = new_population
        energies = new_energies
        
        # Update best
        gen_best_idx = np.argmin(energies)
        if energies[gen_best_idx] < best_energy:
            best_energy = energies[gen_best_idx]
            best_pos = population[gen_best_idx].copy()
    
    return best_pos, best_energy


def construct(instance):
    """Find low-energy Dzugutov cluster via GA with polytetrahedral templates."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Check remaining time
    remaining = _remaining()
    if remaining < 10:
        # Emergency fallback
        try:
            seed = _generate_mackay_icosahedron(size, base_spacing=1.1)
            energy, pos = _relax_with_timeout(seed, potential, parameter, timeout=5.0)
            return pos
        except Exception:
            return _generate_mackay_icosahedron(size, base_spacing=1.1)
    
    # Allocate time budget for GA
    ga_budget = min(remaining * 0.6, 45.0)
    
    # Run GA
    try:
        best_pos, best_energy = _ga_search(size, potential, parameter, 
                                           pop_size=8, generations=12, 
                                           time_budget=ga_budget)
    except Exception:
        # Fallback to simple seed
        try:
            seed = _generate_mackay_icosahedron(size, base_spacing=1.1)
            best_energy, best_pos = _relax_with_timeout(seed, potential, parameter, timeout=10.0)
        except Exception:
            best_pos = _generate_mackay_icosahedron(size, base_spacing=1.1)
            best_energy = _dzugutov_energy(best_pos)
    
    # Final refinement: perturb and relax around best
    remaining2 = _remaining()
    if remaining2 > 15:
        rng2 = np.random.default_rng(42)
        for _ in range(5):
            if _remaining() < 10:
                break
            perturbed = best_pos + rng2.normal(0, 0.04, best_pos.shape)
            try:
                energy, positions = _relax_with_timeout(perturbed, potential, parameter, timeout=8.0)
                if energy < best_energy:
                    best_energy = energy
                    best_pos = positions
            except Exception:
                continue
    
    return best_pos
# EVOLVE-BLOCK-END
