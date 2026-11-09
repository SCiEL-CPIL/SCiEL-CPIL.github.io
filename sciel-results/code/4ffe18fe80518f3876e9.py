"""
Strategy: Physics-driven self-assembly with basin hopping refinement for Dzugutov clusters.

Combines the structural generation of the current program (icosahedral/Mackay seeds)
with a physics-driven self-assembly approach using Langevin dynamics. The system
naturally evolves into low-energy polytetrahedral configurations. Multiple parallel
basin-hopping chains refine the structures found by self-assembly.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient


# Global deadline tracking
_START_TIME = time.time()
_HARD_DEADLINE = 320.0  # Leave 10s safety margin before 330s limit
_MAX_THREADS = 4


def _remaining_time():
    return _HARD_DEADLINE - (time.time() - _START_TIME)


def _energy(positions, potential='dzugutov'):
    """Compute Dzugutov energy directly using vectorized operations."""
    A, B, a, b, c, d, m = 5.82, 1.28, 1.87, 1.94, 1.10, 0.27, 16
    n = len(positions)
    if n < 2:
        return 0.0
    
    # Compute all pair distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Only compute upper triangle
    i_idx, j_idx = np.triu_indices(n, k=1)
    r = dists[i_idx, j_idx]
    
    # Mask for valid distances
    valid = r > 1e-12
    if not np.any(valid):
        return 0.0
    
    r_valid = r[valid]
    E = 0.0
    
    # Apply potential
    mask_a = r_valid < a
    mask_b = r_valid < b
    
    if np.any(mask_a):
        ra = r_valid[mask_a]
        E += np.sum(A * (ra ** (-m) - B) * np.exp(c / (ra - a)))
    
    if np.any(mask_b):
        rb = r_valid[mask_b]
        E += np.sum(B * np.exp(d / (rb - b)))
    
    return E


def _is_valid(positions, min_dist=0.5):
    """Check all interatomic distances >= min_dist using vectorized operations."""
    n = len(positions)
    if n < 2:
        return True
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    r = dists[i_idx, j_idx]
    
    return np.all(r >= min_dist)


def _generate_icosahedron(n, rng):
    """
    Generate an icosahedral-like structure with n atoms.
    Uses concentric icosahedral shells and fills remaining atoms.
    """
    # Icosahedral vertices (12 vertices of a regular icosahedron)
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])  # Normalize to unit radius
    
    positions = []
    # Center atom
    if n >= 1:
        positions.append(np.array([0.0, 0.0, 0.0]))
    
    # First shell: 12 atoms at radius r1
    r1 = 1.2
    if n >= 13:
        for v in verts:
            if len(positions) >= n:
                break
            positions.append(v * r1)
    
    # Second shell: 20 or more atoms
    if len(positions) < n:
        r2 = 2.2
        count = min(20, n - len(positions))
        for k in range(count):
            t = (k + 0.5) / count
            z = 1 - 2 * t
            theta = 2 * np.pi * k / 5
            x = np.sqrt(1 - z * z) * np.cos(theta)
            y = np.sqrt(1 - z * z) * np.sin(theta)
            positions.append(np.array([x, y, z]) * r2)
    
    # Fill remaining with random close-packed positions
    while len(positions) < n:
        for attempt in range(100):
            p = rng.uniform(-3.0, 3.0, 3)
            if all(np.linalg.norm(p - q) >= 1.0 for q in positions):
                positions.append(p)
                break
        else:
            angle = rng.uniform(0, 2 * np.pi)
            elev = rng.uniform(-np.pi/2, np.pi/2)
            r = 2.5 + rng.uniform(0, 0.5)
            p = np.array([
                r * np.cos(elev) * np.cos(angle),
                r * np.cos(elev) * np.sin(angle),
                r * np.sin(elev)
            ])
            positions.append(p)
    
    return np.array(positions[:n])


def _generate_mackay(n, rng):
    """
    Generate a Mackay supericosahedron-like structure.
    """
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    verts /= np.linalg.norm(verts[0])
    
    positions = [np.array([0.0, 0.0, 0.0])]  # Center
    
    # Shell 1: 12 atoms
    r1 = 1.15
    for v in verts:
        if len(positions) < n:
            positions.append(v * r1)
    
    # Shell 2: 20 atoms
    if len(positions) < n:
        r2 = 1.95
        for k in range(20):
            if len(positions) >= n:
                break
            t = (k + 0.5) / 20
            z = 1 - 2 * t
            theta = 2 * np.pi * (k / 5.0)
            x = np.sqrt(1 - z * z) * np.cos(theta)
            y = np.sqrt(1 - z * z) * np.sin(theta)
            positions.append(np.array([x, y, z]) * r2)
    
    # Shell 3: 20 atoms
    if len(positions) < n:
        r3 = 2.7
        for k in range(20):
            if len(positions) >= n:
                break
            t = (k + 0.5) / 20
            z = 1 - 2 * t
            theta = 2 * np.pi * ((k + 2.5) / 5.0)
            x = np.sqrt(1 - z * z) * np.cos(theta)
            y = np.sqrt(1 - z * z) * np.sin(theta)
            positions.append(np.array([x, y, z]) * r3)
    
    # Shell 4: fill remaining
    while len(positions) < n:
        for attempt in range(200):
            p = rng.uniform(-3.5, 3.5, 3)
            if all(np.linalg.norm(p - q) >= 0.9 for q in positions):
                positions.append(p)
                break
        else:
            angle = rng.uniform(0, 2 * np.pi)
            elev = rng.uniform(-np.pi/2, np.pi/2)
            r = 3.2 + rng.uniform(0, 0.5)
            p = np.array([
                r * np.cos(elev) * np.cos(angle),
                r * np.cos(elev) * np.sin(angle),
                r * np.sin(elev)
            ])
            positions.append(p)
    
    return np.array(positions[:n])


def _self_assemble(n, rng, max_steps=200, dt=0.001, gamma=0.5, kT=0.1):
    """
    Physics-driven self-assembly using Langevin dynamics.
    Atoms start at random positions and evolve under the Dzugutov potential.
    """
    # Initialize atoms at random positions in a sphere
    radius = 1.5 * n ** (1.0/3.0)
    positions = []
    for i in range(n):
        while True:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                # Check minimum distance
                valid = True
                for q in positions:
                    if np.linalg.norm(p - q) < 0.6:
                        valid = False
                        break
                if valid:
                    positions.append(p)
                    break
    
    positions = np.array(positions)
    velocities = rng.normal(0, 0.1, size=positions.shape)
    
    best_pos = positions.copy()
    best_E = _energy(positions)
    
    for step in range(max_steps):
        if _remaining_time() < 5.0:
            break
        
        # Compute forces using gradient
        try:
            grad = dzugutov_energy_gradient(positions)
            forces = -grad
        except Exception:
            break
        
        # Langevin update
        noise = rng.normal(0, np.sqrt(2 * kT * gamma / dt), size=positions.shape)
        velocities = (1 - gamma * dt) * velocities + dt * forces + noise
        positions = positions + dt * velocities
        
        # Keep atoms within bounds
        max_r = 2.0 * n ** (1.0/3.0)
        r_norms = np.linalg.norm(positions, axis=1)
        mask = r_norms > max_r
        if np.any(mask):
            positions[mask] *= max_r / r_norms[mask]
        
        # Compute energy
        E = _energy(positions)
        if E < best_E:
            best_E = E
            best_pos = positions.copy()
    
    return best_E, best_pos


def _basin_hopping(seed_positions, n, max_steps=50, kick_scale=0.3, rng=None):
    """
    Basin hopping: randomly perturb, relax, accept/reject.
    Returns best energy and positions found.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    current = seed_positions.copy()
    E_current = _energy(current)
    best = current.copy()
    E_best = E_current
    
    T = 0.5  # Initial "temperature" for acceptance
    for step in range(max_steps):
        if _remaining_time() < 5.0:
            break
        
        # Random kick
        kick = rng.normal(0, kick_scale, size=current.shape)
        proposed = current + kick
        
        # Relax
        try:
            E_proposed, relaxed = relax(proposed, 'dzugutov')
        except Exception:
            E_proposed = E_current + 100
            relaxed = current
        
        # Check validity
        if not _is_valid(relaxed):
            E_proposed += 50
        
        # Metropolis acceptance
        dE = E_proposed - E_current
        if dE < 0 or rng.random() < np.exp(-dE / T):
            current = relaxed
            E_current = E_proposed
            if E_current < E_best:
                E_best = E_current
                best = current.copy()
        
        # Anneal
        T *= 0.95
        kick_scale = 0.2 + 0.3 * (step / max_steps)
    
    return E_best, best


def _worker_self_assemble(n, rng_seed, max_steps=150):
    """Worker thread: run self-assembly from random initial conditions."""
    rng = np.random.default_rng(rng_seed)
    E, pos = _self_assemble(n, rng, max_steps=max_steps)
    return E, pos


def _worker_basin_hopping(seed_positions, n, rng_seed, max_bh_steps=40):
    """Worker thread: run basin hopping from a seed."""
    rng = np.random.default_rng(rng_seed)
    E, pos = _basin_hopping(seed_positions, n, max_steps=max_bh_steps, rng=rng)
    return E, pos


def construct(instance):
    """
    Find low-energy Dzugutov cluster using self-assembly and basin hopping.
    """
    n = instance["n"]
    potential = instance["potential"]
    
    # Generate diverse seeds
    rng = np.random.default_rng(n)
    seeds = []
    
    # Seed 1: Icosahedral
    seeds.append(_generate_icosahedron(n, rng))
    # Seed 2: Mackay-like
    seeds.append(_generate_mackay(n, rng))
    # Seed 3: Perturbed icosahedral
    s3 = _generate_icosahedron(n, rng) + rng.normal(0, 0.2, (n, 3))
    seeds.append(s3)
    # Seed 4: Perturbed Mackay
    s4 = _generate_mackay(n, rng) + rng.normal(0, 0.2, (n, 3))
    seeds.append(s4)
    
    # Relax all seeds first to get better starting points
    relaxed_seeds = []
    for s in seeds:
        if _remaining_time() < 10.0:
            relaxed_seeds.append(s)
            continue
        try:
            E, relaxed = relax(s, 'dzugutov')
            if _is_valid(relaxed):
                relaxed_seeds.append(relaxed)
            else:
                relaxed_seeds.append(s)
        except Exception:
            relaxed_seeds.append(s)
    
    # Run parallel tasks: self-assembly and basin hopping
    best_E = np.inf
    best_pos = relaxed_seeds[0]
    
    # First, get energies of relaxed seeds
    for i, rs in enumerate(relaxed_seeds):
        try:
            E = _energy(rs)
            if E < best_E:
                best_E = E
                best_pos = rs.copy()
        except Exception:
            pass
    
    # Run parallel workers
    if _remaining_time() > 20.0:
        # Create tasks: 2 self-assembly + 2 basin hopping
        tasks = []
        
        # Self-assembly tasks
        for i in range(2):
            rng_seed = n * 1000 + 100 + i
            tasks.append(('sa', n, rng_seed))
        
        # Basin hopping tasks from best seeds
        for i in range(min(2, len(relaxed_seeds))):
            rng_seed = n * 1000 + i
            tasks.append(('bh', relaxed_seeds[i], n, rng_seed))
        
        with ThreadPoolExecutor(max_workers=_MAX_THREADS) as executor:
            futures = []
            for task in tasks:
                if task[0] == 'sa':
                    future = executor.submit(_worker_self_assemble, task[1], task[2], max_steps=150)
                else:
                    future = executor.submit(_worker_basin_hopping, task[1], task[2], task[3], max_bh_steps=40)
                futures.append(future)
            
            for future in as_completed(futures):
                if _remaining_time() < 5.0:
                    break
                try:
                    E, pos = future.result(timeout=10)
                    if E < best_E:
                        best_E = E
                        best_pos = pos.copy()
                except Exception:
                    pass
    
    # Final relaxation of best result
    if _remaining_time() > 5.0:
        try:
            E_final, pos_final = relax(best_pos, 'dzugutov')
            if _is_valid(pos_final):
                E_check = _energy(pos_final)
                if E_check < best_E:
                    best_E = E_check
                    best_pos = pos_final
        except Exception:
            pass
    
    return best_pos
# EVOLVE-BLOCK-END
