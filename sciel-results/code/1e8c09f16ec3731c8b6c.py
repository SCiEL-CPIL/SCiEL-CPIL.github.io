# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline guard
_GLOBAL_START_TIME = None
_DEADLINE = 320.0  # 320s wall clock limit (evaluator kills at 330s)

def _get_remaining_time():
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
    return _DEADLINE - (time.time() - _GLOBAL_START_TIME)

def _morse_energy(x, rho):
    """Compute Morse energy for a set of positions (vectorized)."""
    n = x.shape[0]
    if n < 2:
        return 0.0
    # Compute pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    # Mask diagonal
    np.fill_diagonal(dist, np.inf)
    
    # Morse term
    er = rho * (1.0 - dist)
    er = np.clip(er, -50, 50)
    exp_er = np.exp(er)
    energy_terms = exp_er * (exp_er - 2.0)
    
    # Sum upper triangle (i < j)
    iu = np.triu_indices(n, 1)
    return float(np.sum(energy_terms[iu]))

def _min_distance(x):
    """Compute minimum distance between any pair of atoms."""
    n = x.shape[0]
    if n < 2:
        return np.inf
    tree = cKDTree(x)
    # Query nearest neighbors, exclude self
    dists, _ = tree.query(x, k=2)
    return float(dists[:, 1].min())

def _is_valid(x):
    """Check if the configuration is valid (atoms at least 0.5 apart)."""
    return _min_distance(x) >= 0.5

def _generate_icosahedron(n, seed=42):
    """
    Generate an icosahedral cluster.
    For n <= 13: Icosahedron
    For n <= 55: Icosahedron + Shells
    """
    rng = np.random.default_rng(seed)
    
    # Icosahedron vertices (radius 1)
    phi = (1 + np.sqrt(5)) / 2
    verts = [
        (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
        (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
        (phi, 0, 1), (-phi, 0, 1), (phi, 0, -1), (-phi, 0, -1)
    ]
    verts = np.array(verts)
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis]
    
    positions = [np.array([0.0, 0.0, 0.0])]  # Center
    # Scale icosahedron to have edge length ~ 1
    if len(verts) >= 2:
        d_min = np.min(np.linalg.norm(verts[:, np.newaxis, :] - verts[np.newaxis, :, :], axis=2))
        scale = 1.0 / d_min
        verts = verts * scale
        positions.extend([v for v in verts])
    
    remaining = n - len(positions)
    
    if remaining <= 0:
        return np.array(positions[:n])
    
    # Add atoms in the next shell
    shell_radius = 1.8
    for _ in range(remaining):
        u = rng.uniform(-1, 1)
        theta = rng.uniform(0, 2 * np.pi)
        r = shell_radius
        x = r * np.sqrt(1 - u**2) * np.cos(theta)
        y = r * np.sqrt(1 - u**2) * np.sin(theta)
        z = r * u
        positions.append(np.array([x, y, z]))
        
    return np.array(positions[:n])

def _generate_fcc_cluster(n, seed=42):
    """Generate an FCC-like cluster."""
    rng = np.random.default_rng(seed)
    
    a = math.sqrt(2)
    radius = 1.5 * (n ** (1.0/3.0)) * a
    points = []
    
    basis = [(0, 0, 0), (0, a/2, a/2), (a/2, 0, a/2), (a/2, a/2, 0)]
    
    max_coord = int(radius / a) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for dx, dy, dz in basis:
                    p = np.array([i * a + dx, j * a + dy, k * a + dz])
                    if np.linalg.norm(p) <= radius:
                        points.append(p)
    
    points.sort(key=lambda p: np.linalg.norm(p))
    
    if len(points) >= n:
        jitter = rng.uniform(-0.05, 0.05, (n, 3))
        return np.array(points[:n]) + jitter
    else:
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
        return np.array(points)

def _generate_bcc_cluster(n, seed=42):
    """Generate a BCC-like cluster."""
    rng = np.random.default_rng(seed)
    
    a = 2.0 / math.sqrt(3)
    radius = 1.5 * (n ** (1.0/3.0)) * a
    points = []
    
    basis = [(0, 0, 0), (a/2, a/2, a/2)]
    
    max_coord = int(radius / a) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for dx, dy, dz in basis:
                    p = np.array([i * a + dx, j * a + dy, k * a + dz])
                    if np.linalg.norm(p) <= radius:
                        points.append(p)
    
    points.sort(key=lambda p: np.linalg.norm(p))
    
    if len(points) >= n:
        jitter = rng.uniform(-0.05, 0.05, (n, 3))
        return np.array(points[:n]) + jitter
    else:
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            points.append(p)
        return np.array(points)

def _generate_random_cluster(n, seed=42, min_dist=0.6):
    """Generate a random cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    
    radius = 2.0 * (n ** (1.0/3.0))
    points = []
    attempts = 0
    max_attempts = 50000
    
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < min_dist:
                    valid = False
                    break
            if valid:
                points.append(p)
        attempts += 1
    
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points)

def _relax_and_validate(positions, potential, rho):
    """Relax positions and check validity."""
    n = positions.shape[0]
    try:
        energy, relaxed = relax(positions, potential, rho)
        if _min_distance(relaxed) >= 0.5:
            return energy, relaxed
        else:
            return float('inf'), None
    except Exception:
        return float('inf'), None

def _polish(positions, potential, rho):
    """
    Final high-precision polish using L-BFGS-B with tight tolerances.
    """
    n = positions.shape[0]
    try:
        def obj(x_flat):
            x = x_flat.reshape(n, 3)
            try:
                e, _ = relax(x, potential, rho)
                return e
            except:
                return 1e10
        
        def grad(x_flat):
            x = x_flat.reshape(n, 3)
            try:
                e, g = morse_energy_gradient(x, rho)
                return g.flatten()
            except:
                return np.zeros_like(x_flat)
        
        x0 = positions.flatten()
        result = minimize(obj, x0, jac=grad, method='L-BFGS-B', 
                         options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12})
        
        if result.success and _min_distance(result.x.reshape(n, 3)) >= 0.5:
            return result.fun, result.x.reshape(n, 3)
        else:
            return float('inf'), None
    except Exception:
        return float('inf'), None

def _basin_hop(start_positions, potential, rho, max_iterations=60, initial_step=0.5, initial_temp=2.0):
    """
    Perform basin hopping optimization.
    Returns (best_energy, best_positions).
    """
    n = start_positions.shape[0]
    current_energy, current_positions = _relax_and_validate(start_positions, potential, rho)
    
    if current_positions is None:
        perturbed = start_positions + np.random.normal(0, 0.1, start_positions.shape)
        current_energy, current_positions = _relax_and_validate(perturbed, potential, rho)
        if current_positions is None:
            return float('inf'), None
    
    best_energy = current_energy
    best_positions = current_positions.copy()
    
    step_size = initial_step
    temp = initial_temp
    
    for it in range(max_iterations):
        if _get_remaining_time() < 5.0:
            break
        
        noise = np.random.normal(0, step_size, current_positions.shape)
        new_positions = current_positions + noise
        
        new_energy, new_relaxed = _relax_and_validate(new_positions, potential, rho)
        
        if new_relaxed is None:
            noise = np.random.normal(0, step_size * 0.5, current_positions.shape)
            new_positions = current_positions + noise
            new_energy, new_relaxed = _relax_and_validate(new_positions, potential, rho)
        
        if new_relaxed is None:
            continue
        
        delta_e = new_energy - current_energy
        if delta_e < 0 or (temp > 1e-6 and np.random.rand() < math.exp(-delta_e / temp)):
            current_energy = new_energy
            current_positions = new_relaxed
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = current_positions.copy()
        
        temp *= 0.95
        step_size *= 0.98
        
        if it > 0 and it % 20 == 0:
            step_size *= 1.2
    
    return best_energy, best_positions

def construct(instance):
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
        
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    potential = instance["potential"]
    
    if rho is None:
        rho = 6
    
    best_energy = float('inf')
    best_positions = None
    
    # Generate diverse initial guesses
    initial_guesses = []
    
    # 1. Icosahedral
    try:
        initial_guesses.append(_generate_icosahedron(n, seed=42))
    except Exception:
        pass
    
    # 2. FCC cluster
    try:
        initial_guesses.append(_generate_fcc_cluster(n, seed=42))
    except Exception:
        pass
    
    # 3. BCC cluster
    try:
        initial_guesses.append(_generate_bcc_cluster(n, seed=42))
    except Exception:
        pass
    
    # 4. Random clusters with different seeds
    for seed in [1, 2, 3, 4, 5]:
        try:
            initial_guesses.append(_generate_random_cluster(n, seed=seed))
        except Exception:
            pass
    
    # 5. Additional random clusters if n is large
    if n > 50:
        for seed in [6, 7, 8]:
            try:
                initial_guesses.append(_generate_random_cluster(n, seed=seed))
            except Exception:
                pass
    
    # Relax all initial guesses and keep the best ones
    relaxed_results = []
    for guess in initial_guesses:
        if _get_remaining_time() < 10.0:
            break
        
        energy, positions = _relax_and_validate(guess, potential, rho)
        if positions is not None and energy < float('inf'):
            relaxed_results.append((energy, positions))
    
    relaxed_results.sort(key=lambda x: x[0])
    
    if relaxed_results:
        best_energy = relaxed_results[0][0]
        best_positions = relaxed_results[0][1].copy()
    
    # Perform basin hopping from the top few starts
    num_bh_starts = min(len(relaxed_results), 3)
    
    for idx in range(num_bh_starts):
        if _get_remaining_time() < 15.0:
            break
        
        start_energy, start_pos = relaxed_results[idx]
        
        if rho <= 3:
            initial_step = 0.8
            initial_temp = 5.0
        elif rho <= 6:
            initial_step = 0.5
            initial_temp = 3.0
        else:
            initial_step = 0.3
            initial_temp = 2.0
        
        if n <= 40:
            max_iter = 80
        elif n <= 60:
            max_iter = 50
        else:
            max_iter = 30
        
        bh_energy, bh_positions = _basin_hop(
            start_pos, potential, rho,
            max_iterations=max_iter,
            initial_step=initial_step,
            initial_temp=initial_temp
        )
        
        if bh_positions is not None and bh_energy < best_energy:
            best_energy = bh_energy
            best_positions = bh_positions.copy()
        
        if _get_remaining_time() > 20.0 and idx == 0:
            bh_energy2, bh_positions2 = _basin_hop(
                start_pos, potential, rho,
                max_iterations=max_iter // 2,
                initial_step=initial_step * 1.5,
                initial_temp=initial_temp * 1.5
            )
            if bh_positions2 is not None and bh_energy2 < best_energy:
                best_energy = bh_energy2
                best_positions = bh_positions2.copy()
    
    # Final polish with high-precision optimizer
    if best_positions is not None and _get_remaining_time() > 10.0:
        polish_energy, polish_positions = _polish(best_positions, potential, rho)
        if polish_positions is not None and polish_energy < best_energy:
            best_energy = polish_energy
            best_positions = polish_positions.copy()
    
    # Final fallback
    if best_positions is None:
        best_positions = _generate_icosahedron(n, seed=0)
        if not _is_valid(best_positions):
            best_positions = best_positions * 1.5
            if not _is_valid(best_positions):
                best_positions = _generate_random_cluster(n, seed=99)
    
    return best_positions
# EVOLVE-BLOCK-END
