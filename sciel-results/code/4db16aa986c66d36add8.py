"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient


def _time_left(start_time, deadline=300.0):
    return deadline - (time.time() - start_time)


def _fcc_sphere(n, spacing=0.72):
    """Generate n points in an FCC-like spherical arrangement."""
    if n == 0:
        return np.zeros((0, 3))
    
    # Generate points on FCC lattice shells
    points = []
    shell = 0
    while len(points) < n:
        if shell == 0:
            points.append(np.array([0.0, 0.0, 0.0]))
        else:
            # Add points on the shell
            # Use a simple approach: generate points on a sphere of radius shell * spacing
            radius = shell * spacing
            # Number of points to add from this shell
            remaining = n - len(points)
            
            # Generate points on the sphere using golden spiral
            k = min(remaining, 12 * shell * shell)  # approximate number of points on shell
            for i in range(k):
                # Fibonacci sphere
                phi = np.arccos(1 - 2 * (i + 0.5) / k)
                theta = np.pi * (1 + 5 ** 0.5) * i
                x = radius * np.sin(phi) * np.cos(theta)
                y = radius * np.sin(phi) * np.sin(theta)
                z = radius * np.cos(phi)
                points.append(np.array([x, y, z]))
        shell += 1
    
    points = np.array(points[:n])
    # Center the cluster
    points -= np.mean(points, axis=0)
    return points


def _random_compact(n, spacing=0.72, seed=None):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    max_attempts = n * 100
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            # Check minimum distance
            if all(np.linalg.norm(p - q) >= 0.6 * spacing for q in points):
                points.append(p)
        attempts += 1
    
    # If we couldn't place all points, just add random ones
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    points = np.array(points)
    points -= np.mean(points, axis=0)
    return points


def _icosahedron_based(n, spacing=0.72):
    """Generate points based on icosahedral symmetry."""
    # Start with icosahedron vertices
    phi = (1 + 5 ** 0.5) / 2
    vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]) * spacing / 2
    
    points = [vertices[0]]  # center
    # Add vertices
    for v in vertices:
        if len(points) >= n:
            break
        points.append(v)
    
    # Fill remaining with random points near the cluster
    rng = np.random.default_rng(42)
    while len(points) < n:
        base = points[rng.integers(0, len(points))]
        p = base + rng.normal(0, 0.3 * spacing, 3)
        points.append(p)
    
    points = np.array(points[:n])
    points -= np.mean(points, axis=0)
    return points


def construct(instance):
    """Multi-start relaxation with diverse initial structures for SC clusters."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    spacing = 0.72  # sutton_chen typical spacing
    
    start_time = time.time()
    deadline = 300.0
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse starting structures
    starters = []
    
    # FCC-like sphere
    starters.append(_fcc_sphere(size, spacing))
    
    # Icosahedron-based
    starters.append(_icosahedron_based(size, spacing))
    
    # Multiple random compact structures
    for seed in [size, size + 1, size + 2, size + 3]:
        starters.append(_random_compact(size, spacing, seed=seed))
    
    # Relax each starter
    for i, starter in enumerate(starters):
        if _time_left(start_time, deadline) < 5.0:
            break
        
        try:
            energy, positions = relax(starter, potential, parameter)
            
            if np.isfinite(energy) and energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            continue
    
    # If we have time, try additional random starts
    if best_positions is not None:
        rng = np.random.default_rng(size + 100)
        for attempt in range(5):
            if _time_left(start_time, deadline) < 10.0:
                break
            
            starter = _random_compact(size, spacing, seed=size + 100 + attempt)
            try:
                energy, positions = relax(starter, potential, parameter)
                
                if np.isfinite(energy) and energy < best_energy:
                    best_energy = energy
                    best_positions = positions
            except Exception:
                continue
    
    # Fallback: return a simple valid configuration if nothing worked
    if best_positions is None:
        best_positions = _random_compact(size, spacing, seed=size)
    
    return best_positions
# EVOLVE-BLOCK-END
