"""Dzugutov cluster global minima: gradient-based simulated annealing + multi-start."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_START_TIME = None
_DEADLINE = 318.0  # safety margin before 330s kill


def _get_time_remaining():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    return _DEADLINE - (time.time() - _START_TIME)


def _build_icosahedral_points():
    """Generate the 12 vertices of a regular icosahedron."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1],
        [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    norm = np.linalg.norm(verts[0])
    return verts / norm


def _generate_polytetrahedral_seed(n, rng, ico_verts):
    """
    Generate a seed with polytetrahedral character using icosahedral shells.
    """
    points = [np.zeros(3)]  # center atom
    
    # Shell 1: icosahedral (12 atoms)
    shell1_r = 1.0
    for v in ico_verts:
        points.append(v * shell1_r)
    
    # Shell 2: 30 edge midpoints scaled out
    if n > 13:
        edges = []
        for i in range(12):
            for j in range(i + 1, 12):
                d = np.linalg.norm(ico_verts[i] - ico_verts[j])
                if d < 1.5:  # adjacent vertices
                    mid = (ico_verts[i] + ico_verts[j]) / 2
                    edges.append(mid)
        edges = np.array(edges)
        if len(edges) > 0:
            edge_norms = np.linalg.norm(edges, axis=1)
            edges = edges / edge_norms[:, None]
            shell2_r = 1.55
            for e in edges:
                if len(points) < n:
                    points.append(e * shell2_r)
    
    # Shell 3: face centers (20 faces)
    if n > 43:
        n_remaining = n - len(points)
        for _ in range(n_remaining):
            theta = rng.uniform(0, 2 * np.pi)
            z = rng.uniform(-1, 1)
            r = np.sqrt(1 - z * z)
            p = np.array([r * np.cos(theta), r * np.sin(theta), z])
            points.append(p * 2.1)
    
    # Fill remaining with outer shell
    while len(points) < n:
        theta = rng.uniform(0, 2 * np.pi)
        z = rng.uniform(-1, 1)
        r = np.sqrt(1 - z * z)
        p = np.array([r * np.cos(theta), r * np.sin(theta), z])
        points.append(p * (2.4 + rng.uniform(0, 0.3)))
    
    points = np.array(points[:n])
    points += rng.normal(0, 0.05, points.shape)
    
    _enforce_min_distance(points, 0.51)
    
    return points


def _enforce_min_distance(points, min_dist):
    """Iteratively push apart any atoms closer than min_dist."""
    for _ in range(10):
        moved = False
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                d = np.linalg.norm(points[i] - points[j])
                if d < min_dist:
                    diff = points[i] - points[j]
                    if d < 1e-10:
                        diff = np.array([1.0, 0.0, 0.0])
                    else:
                        diff /= d
                    push = (min_dist - d) / 2 + 0.01
                    points[i] += diff * push
                    points[j] -= diff * push
                    moved = True
        if not moved:
            break


def _generate_compact_random(n, rng):
    """Generate a compact random cluster."""
    radius = 0.9 * n ** (1.0 / 3.0)
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 200:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.55:
                    valid = False
                    break
            if valid:
                points.append(p)
        attempts += 1
    
    while len(points) < n:
        theta = rng.uniform(0, 2 * np.pi)
        z = rng.uniform(-1, 1)
        r = np.sqrt(1 - z * z)
        p = np.array([r * np.cos(theta), r * np.sin(theta), z])
        points.append(p * (radius + rng.uniform(0, 0.5)))
    
    return np.array(points[:n])


def _generate_bcc_seed(n, rng):
    """Generate a BCC-like seed structure."""
    points = []
    a = 1.1
    body_center = np.array([0.5, 0.5, 0.5]) * a
    
    side = int(np.ceil((n / 2) ** (1/3)))
    for ix in range(side):
        for iy in range(side):
            for iz in range(side):
                corner = np.array([ix, iy, iz]) * a
                if len(points) < n:
                    points.append(corner)
                if len(points) < n:
                    points.append(corner + body_center)
    
    while len(points) < n:
        theta = rng.uniform(0, 2 * np.pi)
        z = rng.uniform(-1, 1)
        r = np.sqrt(1 - z * z)
        p = np.array([r * np.cos(theta), r * np.sin(theta), z])
        points.append(p * (side * 0.6))
    
    points = np.array(points[:n])
    points += rng.normal(0, 0.03, points.shape)
    return points


def _simulated_annealing(positions, potential, rng, time_budget):
    """
    Gradient-based simulated annealing.
    Uses dzugutov_energy_gradient for efficient optimization.
    Returns best positions and energy found.
    """
    n_atoms = len(positions)
    
    # Initial relaxation
    current_energy, current_positions = relax(positions, potential)
    best_energy = current_energy
    best_positions = current_positions.copy()
    
    # Annealing schedule
    T_start = 5.0
    T_end = 0.01
    n_steps = 500
    start_time = time.time()
    
    for step in range(n_steps):
        # Check time budget
        if time.time() - start_time > time_budget:
            break
        
        # Cooling schedule (exponential)
        alpha = step / n_steps
        T = T_start * (T_end / T_start) ** alpha
        
        # Get gradient
        grad = dzugutov_energy_gradient(current_positions)
        
        # Gradient step with noise
        lr = 0.01 * (1.0 + 10.0 * T / T_start)
        noise_scale = 0.1 * np.sqrt(T)
        
        new_positions = current_positions - lr * grad + noise_scale * rng.normal(0, 1, current_positions.shape)
        
        # Enforce minimum distance
        _enforce_min_distance(new_positions, 0.51)
        
        # Relax the new position
        new_energy, relaxed_positions = relax(new_positions, potential)
        
        # Metropolis acceptance
        delta_e = new_energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / max(T, 1e-10)):
            current_energy = new_energy
            current_positions = relaxed_positions.copy()
            
            if new_energy < best_energy:
                best_energy = new_energy
                best_positions = relaxed_positions.copy()
    
    return best_positions, best_energy


def construct(instance):
    """
    Find low-energy Dzugutov clusters using gradient-based simulated annealing
    with multiple diverse starting seeds.
    """
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    n = instance["n"]
    potential = instance["potential"]
    rng = np.random.default_rng(n * 7919 + 42)
    
    ico_verts = _build_icosahedral_points()
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse seeds
    seeds = []
    
    # Primary: polytetrahedral icosahedral seed
    seeds.append(_generate_polytetrahedral_seed(n, rng, ico_verts))
    
    # Secondary: BCC seed
    seeds.append(_generate_bcc_seed(n, rng))
    
    # Tertiary: compact random
    seeds.append(_generate_compact_random(n, rng))
    
    # Additional random seeds
    for _ in range(4):
        seeds.append(_generate_compact_random(n, rng))
    
    # Check time before starting SA
    time_remaining = _get_time_remaining()
    
    # Allocate time per seed: reserve 30s for final processing
    available_time = max(10, time_remaining - 30)
    time_per_seed = available_time / len(seeds)
    
    for seed in seeds:
        if _get_time_remaining() < 35:
            break
        
        # Run simulated annealing from this seed
        sa_positions, sa_energy = _simulated_annealing(
            seed, potential, rng, time_budget=time_per_seed
        )
        
        if sa_energy < best_energy:
            best_energy = sa_energy
            best_positions = sa_positions.copy()
    
    # Final relaxation
    if best_positions is not None:
        final_energy, final_positions = relax(best_positions, potential)
        if final_energy < best_energy:
            best_energy = final_energy
            best_positions = final_positions
    
    if best_positions is None:
        # Fallback: linear chain
        best_positions = np.array([[i * 1.1, 0.0, 0.0] for i in range(n)])
    
    return best_positions
# EVOLVE-BLOCK-END
