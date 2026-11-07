"""Sutton-Chen transition-metal cluster global minima search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _fibonacci_sphere(n, radius, rng=None):
    """Generate n points on a sphere of given radius using Fibonacci spiral."""
    if n <= 0:
        return np.empty((0, 3))
    if n == 1:
        return np.array([[0.0, 0.0, radius]])
    rng = rng or np.random.default_rng(0)
    indices = np.arange(n)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    theta = golden_angle * indices
    phi = np.arccos(1.0 - 2.0 * (indices + 0.5) / n)
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.column_stack([x, y, z])


def _icosahedron_positions(n, seed=0):
    """Generate positions based on icosahedral shells."""
    rng = np.random.default_rng(seed)
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico_vertices = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1]
    ], dtype=np.float64)
    ico_vertices = ico_vertices / np.linalg.norm(ico_vertices, axis=1, keepdims=True)
    
    positions = [np.zeros(3)]  # center atom
    remaining = n - 1
    
    # First shell: 12 icosahedron vertices
    shell1_radius = 1.0
    if remaining >= 12:
        positions.extend(ico_vertices * shell1_radius)
        remaining -= 12
    
    # Second shell: 30 positions (edges of icosahedron)
    if remaining >= 30:
        # Generate 30 points on second shell using fibonacci sphere
        shell2_radius = np.sqrt(2.0)  # distance to edge midpoints scaled
        shell2_pts = _fibonacci_sphere(30, shell2_radius, rng)
        positions.extend(shell2_pts)
        remaining -= 30
    
    # Additional shells
    if remaining > 0:
        shell3_radius = 2.0
        shell3_pts = _fibonacci_sphere(remaining, shell3_radius, rng)
        positions.extend(shell3_pts)
    
    return np.array(positions[:n])


def _fcc_positions(n, seed=0):
    """Generate FCC-like cluster positions."""
    rng = np.random.default_rng(seed)
    # FCC lattice vectors
    a = 1.0
    basis = np.array([
        [0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]
    ], dtype=np.float64) * a
    
    positions = []
    # Generate points in a cubic region
    max_coord = int(np.ceil((n / 4) ** (1.0/3.0))) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for b in basis:
                    p = np.array([i, j, k]) * a + b
                    positions.append(p)
    
    positions = np.array(positions)
    # Sort by distance from center and take first n
    distances = np.linalg.norm(positions, axis=1)
    order = np.argsort(distances)
    positions = positions[order[:n]]
    
    # Center the cluster
    positions -= np.mean(positions, axis=0)
    
    # Add small perturbation to break symmetry
    positions += rng.normal(0, 0.01, positions.shape)
    
    return positions


def _hcp_positions(n, seed=0):
    """Generate HCP-like cluster positions."""
    rng = np.random.default_rng(seed)
    a = 1.0
    c = a * np.sqrt(2.0 / 3.0)
    
    # HCP basis
    basis = np.array([
        [0, 0, 0], [0, 0, c/2],
        [a, 0, 0], [a, 0, c/2],
        [a/2, a*np.sqrt(3)/2, 0], [a/2, a*np.sqrt(3)/2, c/2]
    ], dtype=np.float64)
    
    positions = []
    max_coord = int(np.ceil((n / 6) ** (1.0/3.0))) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for b in basis:
                    p = np.array([i, j, k]) * a + b
                    positions.append(p)
    
    positions = np.array(positions)
    distances = np.linalg.norm(positions, axis=1)
    order = np.argsort(distances)
    positions = positions[order[:n]]
    
    positions -= np.mean(positions, axis=0)
    positions += rng.normal(0, 0.01, positions.shape)
    
    return positions


def _random_compact(n, seed=0):
    """Generate random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = 0.72 * n ** (1.0/3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 100:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
        attempts += 1
    
    # If we couldn't place enough points, fill with random positions
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points[:n])


def construct(instance):
    """Search for global minimum of Sutton-Chen cluster using multiple starting templates."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    deadline = start_time + 280  # Leave 50s safety margin
    
    # Generate diverse starting positions
    templates = [
        ("icosahedral", _icosahedron_positions(size, seed=0)),
        ("fcc", _fcc_positions(size, seed=0)),
        ("hcp", _hcp_positions(size, seed=0)),
        ("random1", _random_compact(size, seed=42)),
        ("random2", _random_compact(size, seed=123)),
        ("random3", _random_compact(size, seed=456)),
        ("ico_perturbed", _icosahedron_positions(size, seed=1) + np.random.default_rng(7).normal(0, 0.05, (size, 3))),
        ("fcc_perturbed", _fcc_positions(size, seed=1) + np.random.default_rng(8).normal(0, 0.05, (size, 3))),
    ]
    
    best_energy = np.inf
    best_positions = None
    
    for name, positions in templates:
        if time.time() > deadline:
            break
        
        # Validate positions
        if not np.all(np.isfinite(positions)):
            continue
        min_dist = np.inf
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                d = np.linalg.norm(positions[i] - positions[j])
                if d < min_dist:
                    min_dist = d
        if min_dist < 0.35:
            # Try to fix by spreading out
            positions = positions * 1.5
            continue
        
        try:
            energy, relaxed_pos = relax(positions, potential, parameter)
            if np.isfinite(energy) and energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pos.copy()
        except Exception:
            continue
    
    # If no valid result, fall back to a simple random start
    if best_positions is None:
        positions = _random_compact(size, seed=999)
        try:
            best_energy, best_positions = relax(positions, potential, parameter)
        except Exception:
            best_positions = positions
    
    return best_positions
# EVOLVE-BLOCK-END
