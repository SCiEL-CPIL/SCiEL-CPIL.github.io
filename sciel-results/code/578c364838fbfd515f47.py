"""Dzugutov cluster optimization: polytetrahedral seeding + annealed gradient descent."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient

_DEADLINE = time.time() + 290  # safety margin before 330s kill

def _dzugutov_energy(positions):
    """Vectorized Dzugutov energy computation."""
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    n = len(positions)
    if n < 2:
        return 0.0
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dist, np.inf)
    E = np.zeros_like(dist)
    mask_a = dist < a
    mask_b = (dist >= a) & (dist < b)
    # Region 1: r < a
    r1 = dist[mask_a]
    if len(r1) > 0:
        E[mask_a] = A * (r1 ** (-m) - B) * np.exp(c / (r1 - a))
    # Region 2: a <= r < b
    r2 = dist[mask_b]
    if len(r2) > 0:
        E[mask_b] = B * np.exp(d / (r2 - b))
    return 0.5 * np.sum(E)

def _generate_icosahedral_seed(n, rng):
    """Generate a polytetrahedral/icosahedral-like seed for n atoms."""
    # Start with a central atom and build icosahedral shells
    # icosahedron vertices
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [t, 0, -1], [-t, 0, 1], [-t, 0, -1],
    ], dtype=float)
    ico /= np.linalg.norm(ico[0])  # normalize to unit radius
    
    # Generate candidate positions: center + icosahedral shells at various radii
    positions = [np.zeros(3)]  # center atom
    shells = [1.0, 1.6, 2.2, 2.8, 3.4, 4.0]
    # For each shell, place atoms at icosahedral directions scaled
    shell_idx = 0
    while len(positions) < n:
        r = shells[shell_idx % len(shells)]
        # Add jitter to break symmetry slightly
        jitter = 0.05 * rng.standard_normal(3)
        for v in ico:
            if len(positions) >= n:
                break
            p = r * v + jitter * 0.3
            positions.append(p)
        shell_idx += 1
    
    positions = np.array(positions[:n])
    # Add small random perturbation to avoid exact symmetry (which can be a saddle)
    positions += 0.02 * rng.standard_normal((n, 3))
    return positions

def _generate_random_compact(n, rng):
    """Generate a random compact cluster."""
    spacing = 1.1
    radius = spacing * n ** (1.0 / 3.0) * 0.85
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 200:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.65 * spacing for q in points):
                points.append(p)
        attempts += 1
    # If we couldn't place all, fill remaining randomly
    while len(points) < n:
        p = rng.uniform(-radius * 0.5, radius * 0.5, 3)
        points.append(p)
    return np.array(points)

def _anneal_dzugutov(positions, max_iters=8000, lr0=0.002, lr_end=0.0001, temp0=0.5, temp_end=0.01):
    """Annealed gradient descent with cosine easing for the Dzugutov potential."""
    n = len(positions)
    pos = positions.copy()
    best_pos = pos.copy()
    best_E = _dzugutov_energy(pos)
    
    for it in range(max_iters):
        # Cosine-annealed learning rate and temperature
        frac = it / max(max_iters - 1, 1)
        lr = lr_end + 0.5 * (lr0 - lr_end) * (1.0 + np.cos(np.pi * frac))
        temp = temp_end + 0.5 * (temp0 - temp_end) * (1.0 + np.cos(np.pi * frac))
        
        # Get gradient
        E, grad = dzugutov_energy_gradient(pos)
        
        if not np.isfinite(E) or not np.all(np.isfinite(grad)):
            break
        
        # Gradient step
        pos_new = pos - lr * grad
        
        # Small thermal perturbation (annealing noise)
        noise = temp * 0.01 * np.random.standard_normal((n, 3))
        pos_new = pos_new + noise
        
        E_new = _dzugutov_energy(pos_new)
        
        if np.isfinite(E_new) and E_new < best_E:
            best_E = E_new
            best_pos = pos_new.copy()
        
        pos = pos_new
        
        # Early stop if converged
        if frac > 0.5 and np.linalg.norm(grad) < 1e-8:
            break
    
    return best_E, best_pos

def _local_relax(positions, potential='dzugutov'):
    """Use the built-in relax function as a final polish."""
    try:
        energy, pos = relax(positions, potential)
        if np.isfinite(energy):
            return energy, pos
    except:
        pass
    E = _dzugutov_energy(positions)
    return E, positions

def construct(instance):
    """Find low-energy Dzugutov cluster via polytetrahedral seeding + annealing."""
    size = instance["n"]
    potential = instance["potential"]
    
    start_time = time.time()
    rng = np.random.default_rng(42 + size)
    
    best_E = np.inf
    best_pos = None
    
    # Generate diverse starting points
    n_seeds = 6
    seeds = []
    
    # Icosahedral/polytetrahedral seeds (most important for Dzugutov)
    for i in range(3):
        seed = _generate_icosahedral_seed(size, rng)
        seeds.append(seed)
    
    # Random compact seeds
    for i in range(n_seeds - 3):
        seed = _generate_random_compact(size, rng)
        seeds.append(seed)
    
    # For each seed, do annealed optimization
    for si, seed in enumerate(seeds):
        if time.time() > _DEADLINE - 30:
            break
        
        # Quick initial relax to get a reasonable starting point
        E_init, pos_init = _local_relax(seed, potential)
        
        # Annealed gradient descent
        max_iters = max(2000, 8000 - si * 500)  # less work for later seeds
        E_ann, pos_ann = _anneal_dzugutov(pos_init, max_iters=max_iters)
        
        # Final polish with built-in relax
        E_final, pos_final = _local_relax(pos_ann, potential)
        
        if E_final < best_E:
            best_E = E_final
            best_pos = pos_final
    
    # If we still have time, try a few more random restarts with annealing
    remaining = _DEADLINE - time.time() - 10
    if remaining > 20 and best_pos is not None:
        extra_restarts = min(4, int(remaining // 15))
        for _ in range(extra_restarts):
            if time.time() > _DEADLINE - 15:
                break
            seed = _generate_icosahedral_seed(size, rng)
            E_init, pos_init = _local_relax(seed, potential)
            E_ann, pos_ann = _anneal_dzugutov(pos_init, max_iters=3000)
            E_final, pos_final = _local_relax(pos_ann, potential)
            if E_final < best_E:
                best_E = E_final
                best_pos = pos_final
    
    if best_pos is None:
        # Fallback: just return a random compact cluster
        best_pos = _generate_random_compact(size, rng)
    
    # Final validation: ensure minimum distance constraint
    # If any pair is too close, try to fix
    for _ in range(10):
        diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1))
        np.fill_diagonal(dist, np.inf)
        if dist.min() >= 0.5:
            break
        # Nudge close pairs apart
        i, j = np.unravel_index(np.argmin(dist), dist.shape)
        if i == j:
            break
        direction = best_pos[i] - best_pos[j]
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            direction = np.random.standard_normal(3)
            norm = np.linalg.norm(direction)
        direction /= norm
        push = (0.5 - dist[i, j] + 0.01) / 2.0
        best_pos[i] += push * direction
        best_pos[j] -= push * direction
    
    return best_pos
# EVOLVE-BLOCK-END
