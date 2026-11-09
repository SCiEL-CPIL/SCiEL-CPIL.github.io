# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Global timing
_GLOBAL_START = time.time()
_HARD_LIMIT = 320.0  # Safety margin before 330s kill

def _time_left():
    return _HARD_LIMIT - (time.time() - _GLOBAL_START)

def _energy(positions):
    """Compute Dzugutov energy."""
    n = len(positions)
    if n < 2:
        return 0.0
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    total = 0.0
    for i in range(n):
        for j in range(i+1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            if r < 1e-12:
                return 1e10
            if r < a:
                total += A * (r**(-m) - B) * np.exp(c / (r - a))
            elif r < b:
                total += B * np.exp(d / (r - b))
    return total

def _valid(positions):
    """Check minimum distance constraint."""
    n = len(positions)
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(positions[i] - positions[j]) < 0.5 - 1e-9:
                return False
    return True

def _generate_icosahedron_shell(radius, n_atoms):
    """Generate points on an icosahedral shell."""
    # Golden ratio
    phi = (1 + np.sqrt(5)) / 2
    # 12 vertices of icosahedron
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts /= np.linalg.norm(verts[0])
    
    points = []
    if n_atoms <= 12:
        # Just take first n_atoms vertices
        for i in range(n_atoms):
            points.append(verts[i] * radius)
    else:
        # Distribute more evenly: use multiple shells or denser packing
        # For now, just use the 12 vertices and add more by scaling
        for i in range(n_atoms):
            idx = i % 12
            # Add slight perturbation for duplicates
            jitter = np.random.randn(3) * 0.1
            points.append((verts[idx] + jitter) * radius)
    
    return np.array(points[:n_atoms])

def _generate_mackay(n):
    """Generate Mackay supericosahedron-like structure."""
    if n <= 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    points = [np.array([0.0, 0.0, 0.0])]
    
    # First shell: 12 atoms
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts /= np.linalg.norm(verts[0])
    
    r1 = 1.0
    for v in verts:
        if len(points) >= n:
            break
        points.append(v * r1)
    
    # Second shell: 30 atoms (if needed)
    if len(points) < n:
        # Generate 30 points for second shell
        r2 = 1.5
        # Use icosahedral symmetry with more points
        angles = np.linspace(0, 2*np.pi, 30, endpoint=False)
        for i in range(30):
            if len(points) >= n:
                break
            theta = angles[i]
            # Vary the polar angle for better distribution
            polar = np.pi * (i + 0.5) / 30
            x = r2 * np.sin(polar) * np.cos(theta)
            y = r2 * np.sin(polar) * np.sin(theta)
            z = r2 * np.cos(polar)
            points.append(np.array([x, y, z]))
    
    # Third shell: 48 atoms (if needed)
    if len(points) < n:
        r3 = 2.0
        angles = np.linspace(0, 2*np.pi, 48, endpoint=False)
        for i in range(48):
            if len(points) >= n:
                break
            theta = angles[i]
            polar = np.pi * (i + 0.5) / 48
            x = r3 * np.sin(polar) * np.cos(theta)
            y = r3 * np.sin(polar) * np.sin(theta)
            z = r3 * np.cos(polar)
            points.append(np.array([x, y, z]))
    
    return np.array(points[:n])

def _generate_fcc(n):
    """Generate FCC-like structure."""
    points = []
    a = 1.0  # lattice parameter
    # Generate FCC lattice points
    for i in range(-2, 3):
        for j in range(-2, 3):
            for k in range(-2, 3):
                if len(points) >= n:
                    break
                # FCC: corners and face centers
                base = np.array([i*a, j*a, k*a])
                # Corner
                points.append(base)
                # Face centers
                if len(points) < n:
                    points.append(base + np.array([a/2, a/2, 0]))
                if len(points) < n:
                    points.append(base + np.array([a/2, 0, a/2]))
                if len(points) < n:
                    points.append(base + np.array([0, a/2, a/2]))
    
    points = np.array(points[:n])
    # Center it
    points -= points.mean(axis=0)
    return points

def _generate_random(n, seed=None):
    """Generate random compact structure."""
    rng = np.random.default_rng(seed)
    points = []
    radius = 1.1 * n ** (1.0/3.0) * 0.8
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            # Check minimum distance
            if all(np.linalg.norm(p - q) >= 0.55 for q in points):
                points.append(p)
    return np.array(points)

def _relax_with_timeout(positions, potential, parameter, timeout=30.0):
    """Relax with a timeout guard."""
    try:
        energy, relaxed = relax(positions, potential, parameter)
        return energy, relaxed
    except Exception:
        return 1e10, positions

def construct(instance):
    """
    Strategy: Multi-start optimization with diverse structural seeds.
    - Icosahedral/Mackay motifs (favored by Dzugutov potential)
    - FCC-like structures
    - Random compact structures
    Use parallel relaxation where possible, with deadline guard.
    """
    n = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    if _time_left() < 10.0:
        # Not enough time, return a quick random structure
        return _generate_random(n, seed=n)
    
    # Generate diverse seeds
    seeds = []
    
    # 1. Mackay/icosahedral
    seeds.append(_generate_mackay(n))
    
    # 2. FCC-like
    seeds.append(_generate_fcc(n))
    
    # 3. Random structures (multiple)
    for i in range(3):
        seeds.append(_generate_random(n, seed=n*100 + i))
    
    # 4. Perturbed icosahedral
    ico = _generate_mackay(n)
    for i in range(2):
        perturbed = ico + np.random.randn(n, 3) * 0.2
        seeds.append(perturbed)
    
    # Filter valid seeds
    valid_seeds = []
    for s in seeds:
        if len(s) == n and _valid(s):
            valid_seeds.append(s)
    
    if not valid_seeds:
        # Fallback: random
        return _generate_random(n, seed=n)
    
    # Relax each seed, keeping track of best
    best_energy = np.inf
    best_positions = valid_seeds[0]
    
    # Use parallel execution if time allows
    n_seeds = len(valid_seeds)
    time_per_seed = max(5.0, _time_left() * 0.8 / n_seeds)
    
    results = []
    if n_seeds <= 4 and _time_left() > 60.0:
        # Parallel relaxation
        with ThreadPoolExecutor(max_workers=min(4, n_seeds)) as executor:
            futures = {}
            for i, seed in enumerate(valid_seeds):
                if _time_left() < 20.0:
                    break
                future = executor.submit(_relax_with_timeout, seed, potential, parameter, time_per_seed)
                futures[future] = i
            
            for future in as_completed(futures):
                if _time_left() < 10.0:
                    break
                try:
                    energy, positions = future.result(timeout=5.0)
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = positions
                except Exception:
                    pass
    else:
        # Serial relaxation
        for seed in valid_seeds:
            if _time_left() < 15.0:
                break
            energy, positions = _relax_with_timeout(seed, potential, parameter, 20.0)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
    
    # Final check
    if best_energy == np.inf:
        return _generate_random(n, seed=n)
    
    return best_positions
# EVOLVE-BLOCK-END
