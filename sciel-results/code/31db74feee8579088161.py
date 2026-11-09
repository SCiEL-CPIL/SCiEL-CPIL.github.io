"""Morse cluster global minima search with multi-start, structured initialization, and time-budget guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline: evaluator kills after 330s wall-clock.
# We use a shared deadline across all six construct() calls.
# The first call sets the deadline; subsequent calls check against it.
_DEADLINE = time.time() + 310.0  # 20s safety margin

def _time_left():
    return _DEADLINE - time.time()

def _morse_energy(pos, rho):
    """Compute total Morse energy: sum_{i<j} exp(rho(1-r_ij))(exp(rho(1-r_ij))-2)."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    u = rho * (1.0 - dist)
    e = np.exp(u)
    pair_e = e * (e - 2.0)
    return 0.5 * np.sum(pair_e)

def _generate_icosahedral_shell(n, radius=1.0, rng=None):
    """Generate n points distributed on a spherical shell using Fibonacci sphere."""
    if rng is None:
        rng = np.random.default_rng(0)
    
    # Fibonacci sphere for even distribution
    i = np.arange(n)
    phi = np.arccos(1 - 2 * (i + 0.5) / n)
    theta = np.pi * (1 + 5**0.5) * i  # Golden angle
    
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    
    return np.column_stack([x, y, z])

def _generate_fcc_like(n, spacing=1.0, rng=None):
    """Generate points in an FCC-like lattice arrangement."""
    if rng is None:
        rng = np.random.default_rng(0)
    
    # Generate points in a cubic grid and keep the first n
    side = int(np.ceil(n ** (1.0/3.0))) + 2
    points = []
    for i in range(-side, side):
        for j in range(-side, side):
            for k in range(-side, side):
                if len(points) >= n:
                    break
                points.append([i * spacing, j * spacing, k * spacing])
            if len(points) >= n:
                break
        if len(points) >= n:
            break
    
    points = np.array(points[:n])
    
    # Center the cluster
    points -= np.mean(points, axis=0)
    
    # Add small random perturbation to break symmetry
    points += rng.normal(0, 0.05, points.shape)
    
    return points

def _generate_compact_random(n, radius, rng=None):
    """Generate n points randomly within a sphere with minimum separation."""
    if rng is None:
        rng = np.random.default_rng(0)
    
    points = []
    min_dist = 0.6  # Minimum separation
    
    # Try to place points with rejection sampling
    max_attempts = n * 100
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            # Check minimum separation
            if points:
                dists = np.linalg.norm(np.array(points) - p, axis=1)
                if np.min(dists) >= min_dist:
                    points.append(p)
            else:
                points.append(p)
        attempts += 1
    
    # If we couldn't place all points with separation, just add random points
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            points.append(p)
    
    return np.array(points)

def _generate_layered(n, spacing=1.0, rng=None):
    """Generate points in concentric shells (core + shells)."""
    if rng is None:
        rng = np.random.default_rng(0)
    
    # Start with a central point
    points = [np.zeros(3)]
    
    # Add shells
    shell_radii = [spacing, 2*spacing, 3*spacing, 4*spacing, 5*spacing]
    for radius in shell_radii:
        if len(points) >= n:
            break
        # Generate points on this shell
        n_on_shell = min(n - len(points), max(1, int(4 * np.pi * radius**2 / (spacing**2 * 2))))
        shell_points = _generate_icosahedral_shell(n_on_shell, radius, rng)
        points.extend(shell_points.tolist())
    
    # If we still need more points, add random ones
    while len(points) < n:
        p = rng.uniform(-5*spacing, 5*spacing, 3)
        points.append(p)
    
    points = np.array(points[:n])
    return points

def construct(instance):
    """Find low-energy arrangement of n atoms with Morse potential."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    # Time budget for this instance
    time_for_instance = max(5.0, min(50.0, _time_left() / 6.0))  # Roughly equal split
    deadline = time.time() + time_for_instance
    
    rng = np.random.default_rng(size)
    spacing = 1.0
    
    # Generate diverse starting configurations
    candidates = []
    
    # 1. Icosahedral shell
    radius = spacing * size ** (1.0/3.0) * 0.8
    candidates.append(_generate_icosahedral_shell(size, radius, rng))
    
    # 2. FCC-like lattice
    candidates.append(_generate_fcc_like(size, spacing, rng))
    
    # 3. Compact random
    candidates.append(_generate_compact_random(size, radius, rng))
    
    # 4. Layered structure
    candidates.append(_generate_layered(size, spacing, rng))
    
    # 5-8. More random starts with different radii
    for scale in [0.6, 0.9, 1.2, 1.5]:
        candidates.append(_generate_compact_random(size, radius * scale, rng))
    
    best_energy = np.inf
    best_positions = None
    
    for i, pos in enumerate(candidates):
        # Check time
        if time.time() > deadline - 2.0:
            break
        
        # Relax
        try:
            energy, relaxed_pos = relax(pos, potential, parameter)
            
            # Verify minimum separation
            if len(relaxed_pos) >= 2:
                diff = relaxed_pos[:, np.newaxis, :] - relaxed_pos[np.newaxis, :, :]
                dist = np.sqrt(np.sum(diff * diff, axis=2))
                np.fill_diagonal(dist, np.inf)
                if np.min(dist) < 0.49:
                    continue
            
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pos.copy()
        except Exception:
            continue
    
    # If we have time, do more random restarts
    if best_positions is not None and time.time() < deadline - 5.0:
        for _ in range(10):
            if time.time() > deadline - 2.0:
                break
            
            pos = _generate_compact_random(size, radius, rng)
            try:
                energy, relaxed_pos = relax(pos, potential, parameter)
                
                if len(relaxed_pos) >= 2:
                    diff = relaxed_pos[:, np.newaxis, :] - relaxed_pos[np.newaxis, :, :]
                    dist = np.sqrt(np.sum(diff * diff, axis=2))
                    np.fill_diagonal(dist, np.inf)
                    if np.min(dist) < 0.49:
                        continue
                
                if energy < best_energy:
                    best_energy = energy
                    best_positions = relaxed_pos.copy()
            except Exception:
                continue
    
    # If no valid result, return a simple fallback
    if best_positions is None:
        best_positions = _generate_icosahedral_shell(size, radius, rng)
    
    return best_positions
# EVOLVE-BLOCK-END
