"""Improved program using structured initial configurations with time-budget management."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline: kill after 330s, use 310s as working limit for safety
_DEADLINE = time.time() + 310.0

def _time_left():
    return _DEADLINE - time.time()

def _morse_energy(pos, rho):
    """Compute total Morse energy efficiently."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    # Only sum upper triangle
    iu = np.triu_indices(n, k=1)
    r = dist[iu]
    x = rho * (1.0 - r)
    ex = np.exp(x)
    return np.sum(ex * (ex - 2.0))

def _build_icosahedral_shell(n_atoms, spacing=1.0):
    """Build an icosahedral shell configuration."""
    # Icosahedral vertices (12 vertices)
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico_verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts[0])
    
    points = [np.zeros(3)]  # center atom
    
    # Add icosahedral shell
    shell1 = ico_verts * spacing
    for v in shell1:
        if len(points) < n_atoms:
            points.append(v)
    
    # If need more, add second shell (octahedral)
    octa_verts = np.array([
        [1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]
    ], dtype=float)
    shell2 = octa_verts * spacing * 1.6
    for v in shell2:
        if len(points) < n_atoms:
            points.append(v)
    
    # Fill remaining with random points in a shell
    rng = np.random.default_rng(42)
    while len(points) < n_atoms:
        theta = rng.uniform(0, 2*np.pi)
        phi = rng.uniform(0, np.pi)
        r = spacing * (2.0 + rng.uniform(0, 0.5))
        p = np.array([r*np.sin(phi)*np.cos(theta), r*np.sin(phi)*np.sin(theta), r*np.cos(phi)])
        # Check minimum distance
        if all(np.linalg.norm(p - q) >= 0.5 for q in points):
            points.append(p)
    
    return np.array(points[:n_atoms])

def _build_fcc_like(n_atoms, spacing=1.0):
    """Build an FCC-like spherical cluster."""
    points = [np.zeros(3)]
    rng = np.random.default_rng(123)
    
    # FCC lattice directions
    fcc_dirs = np.array([
        [1,1,0],[1,1,2],[1,3,2],[3,1,2],[3,3,0],
        [1,-1,0],[1,-1,2],[1,-3,2],[3,-1,2],[3,-3,0],
        [-1,1,0],[-1,1,2],[-1,3,2],[-3,1,2],[-3,3,0],
        [-1,-1,0],[-1,-1,2],[-1,-3,2],[-3,-1,2],[-3,-3,0],
        [1,0,1],[1,2,1],[3,2,1],[1,0,-1],[1,2,-1],[3,2,-1],
        [-1,0,1],[-1,2,1],[-3,2,1],[-1,0,-1],[-1,2,-1],[-3,2,-1],
        [0,1,1],[2,1,1],[2,3,1],[0,1,-1],[2,1,-1],[2,3,-1],
        [0,-1,1],[2,-1,1],[2,-3,1],[0,-1,-1],[2,-1,-1],[2,-3,-1]
    ], dtype=float)
    fcc_dirs /= np.linalg.norm(fcc_dirs, axis=1, keepdims=True)
    
    for r in [spacing, spacing*1.7, spacing*2.4]:
        for d in fcc_dirs:
            if len(points) >= n_atoms:
                break
            p = d * r
            if all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
        if len(points) >= n_atoms:
            break
    
    # Fill remaining
    while len(points) < n_atoms:
        theta = rng.uniform(0, 2*np.pi)
        phi = rng.uniform(0, np.pi)
        r = spacing * (2.5 + rng.uniform(0, 1.0))
        p = np.array([r*np.sin(phi)*np.cos(theta), r*np.sin(phi)*np.sin(theta), r*np.cos(phi)])
        if all(np.linalg.norm(p - q) >= 0.5 for q in points):
            points.append(p)
    
    return np.array(points[:n_atoms])

def _build_random_compact(n_atoms, spacing=1.0, seed=0):
    """Build a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n_atoms ** (1.0/3.0) * 0.8
    points = []
    attempts = 0
    max_attempts = 10000
    while len(points) < n_atoms and attempts < max_attempts:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.5 for q in points):
            points.append(p)
    
    # If not enough, add more loosely
    while len(points) < n_atoms:
        theta = rng.uniform(0, 2*np.pi)
        phi = rng.uniform(0, np.pi)
        r = radius * (1.0 + rng.uniform(0, 0.5))
        p = np.array([r*np.sin(phi)*np.cos(theta), r*np.sin(phi)*np.sin(theta), r*np.cos(phi)])
        points.append(p)
    
    return np.array(points[:n_atoms])

def construct(instance):
    """Find low-energy Morse cluster using multiple structured starts and local relaxation."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    if potential != "morse":
        # Fallback for other potentials
        spacing = 1.0
        rng = np.random.default_rng(size)
        radius = spacing * size ** (1.0 / 3.0) * 0.75
        points = []
        while len(points) < size:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.5 for q in points):
                points.append(p)
        energy, positions = relax(np.array(points), potential, parameter)
        return positions
    
    rho = parameter
    spacing = 1.0
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. Icosahedral shell
    if _time_left() > 10:
        initial_configs.append(_build_icosahedral_shell(size, spacing))
    
    # 2. FCC-like
    if _time_left() > 10:
        initial_configs.append(_build_fcc_like(size, spacing))
    
    # 3-6. Random compact with different seeds
    for seed in [0, 1, 2, 3]:
        if _time_left() > 10:
            initial_configs.append(_build_random_compact(size, spacing, seed))
    
    # Relax each configuration
    for i, config in enumerate(initial_configs):
        if _time_left() < 5:
            break
        
        try:
            energy, positions = relax(config, "morse", rho)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            continue
    
    # If we still have time, try a few more random restarts
    rng = np.random.default_rng(999)
    extra_attempts = 0
    while _time_left() > 15 and extra_attempts < 8:
        extra_attempts += 1
        config = _build_random_compact(size, spacing, 100 + extra_attempts)
        try:
            energy, positions = relax(config, "morse", rho)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            continue
    
    if best_positions is None:
        # Fallback
        config = _build_random_compact(size, spacing, 0)
        energy, best_positions = relax(config, "morse", rho)
    
    return best_positions
# EVOLVE-BLOCK-END
