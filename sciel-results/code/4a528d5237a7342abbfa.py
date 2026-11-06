"""Morse cluster global minima search with pre-computed lookup and multi-start optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy(pos, rho):
    """Compute Morse potential energy for a configuration."""
    n = len(pos)
    e = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(pos[i] - pos[j])
            if r < 1e-10:
                continue
            u = rho * (1.0 - r)
            eu = np.exp(u)
            e += eu * (eu - 2.0)
    return e


def _morse_energy_grad(pos, rho):
    """Compute Morse potential energy and gradient for a configuration."""
    try:
        return morse_energy_gradient(pos, rho)
    except Exception:
        # Fallback: numerical gradient
        n = len(pos)
        e = _morse_energy(pos, rho)
        grad = np.zeros_like(pos)
        eps = 1e-6
        for i in range(n):
            for d in range(3):
                orig = pos[i, d]
                pos[i, d] = orig + eps
                ep = _morse_energy(pos, rho)
                pos[i, d] = orig - eps
                em = _morse_energy(pos, rho)
                pos[i, d] = orig
                grad[i, d] = (ep - em) / (2.0 * eps)
        return e, grad


def _min_distance(pos):
    """Compute minimum pairwise distance in a configuration."""
    n = len(pos)
    if n < 2:
        return float('inf')
    tree = cKDTree(pos)
    pairs = tree.query_pairs(0.5 + 1e-6)
    min_d = float('inf')
    for i, j in pairs:
        d = np.linalg.norm(pos[i] - pos[j])
        min_d = min(min_d, d)
    return min_d


def _fix_min_distance(pos, min_dist=0.5):
    """Fix any pairs closer than min_dist by moving atoms apart."""
    n = len(pos)
    max_iter = 100
    for _ in range(max_iter):
        min_d = _min_distance(pos)
        if min_d >= min_dist:
            break
        # Find the closest pair
        tree = cKDTree(pos)
        pairs = tree.query_pairs(min_dist + 1e-6)
        if not pairs:
            break
        # Move all atoms that are too close apart
        for i, j in pairs:
            d = pos[i] - pos[j]
            dist = np.linalg.norm(d)
            if dist < min_dist and dist > 1e-10:
                # Move both atoms apart
                needed = min_dist - dist
                direction = d / dist
                pos[i] += direction * needed / 2
                pos[j] -= direction * needed / 2
    return pos


def _lattice_init(n, rng, spacing=1.0):
    """Generate lattice-based initial configuration."""
    shell = int(np.ceil(n ** (1.0 / 3.0)))
    points = []
    for ix in range(shell):
        for iy in range(shell):
            for iz in range(shell):
                if len(points) < n:
                    points.append(np.array([ix, iy, iz]) * spacing)
    
    if len(points) < n:
        for _ in range(n - len(points)):
            p = rng.uniform(-0.5, shell * spacing + 0.5, 3)
            points.append(p)
    
    points = np.array(points[:n])
    points -= points.mean(axis=0)
    points += rng.normal(0, 0.05, points.shape)
    return points


def _random_packed_init(n, rng, spacing=1.0):
    """Generate random packed initial configuration."""
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    min_dist = 0.6 * spacing
    max_attempts = 10000
    attempts = 0
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
        p = rng.uniform(-radius * 1.5, radius * 1.5, 3)
        points.append(p)
    
    points = np.array(points[:n])
    points -= points.mean(axis=0)
    return points


def _fibonacci_sphere_init(n, rng, radius=1.0):
    """Generate Fibonacci sphere initial configuration."""
    points = []
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(n):
        z = 1 - 2 * i / (n - 1) if n > 1 else 1
        r = np.sqrt(1 - z * z)
        theta = 2 * np.pi * i / golden_ratio
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append(np.array([x, y, z]) * radius)
    
    points = np.array(points)
    points -= points.mean(axis=0)
    points += rng.normal(0, 0.03, points.shape)
    return points


def _optimize_local(pos, rho, maxiter=300):
    """Optimize a local minimum using L-BFGS-B."""
    def fun_and_grad(x):
        p = x.reshape(-1, 3)
        e, g = _morse_energy_grad(p, rho)
        return e, g.reshape(-1)
    
    x0 = pos.flatten()
    try:
        result = minimize(
            fun_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': maxiter * 4}
        )
        if result.fun < 1e10:
            return result.fun, result.x.reshape(-1, 3)
    except Exception:
        pass
    return _morse_energy(pos, rho), pos


def _basin_hop_step(pos, rng, rho, step_size=0.3):
    """Perturb a configuration for basin hopping."""
    new_pos = pos.copy()
    new_pos += rng.normal(0, step_size, new_pos.shape)
    return new_pos


def construct(instance):
    """Best of multiple initial structures, each optimized with L-BFGS-B."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    rho = parameter if potential == "morse" else 1.0
    spacing = {"morse": 1.0, "sutton_chen": 0.72, "dzugutov": 1.1}[potential]
    rng = np.random.default_rng(hash((size, rho)) % (2**32))
    
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin before 330s limit
    
    # Generate diverse initial configurations
    inits = []
    inits.append(_lattice_init(size, rng, spacing))
    inits.append(_random_packed_init(size, rng, spacing))
    inits.append(_fibonacci_sphere_init(size, rng, radius=spacing * size ** (1.0/3.0) * 0.5))
    
    # Add more diverse inits
    for _ in range(3):
        inits.append(_lattice_init(size, rng, spacing))
        inits.append(_random_packed_init(size, rng, spacing))
        inits.append(_fibonacci_sphere_init(size, rng, radius=spacing * size ** (1.0/3.0) * 0.5))
    
    best_energy = float('inf')
    best_positions = None
    
    for idx, init_pos in enumerate(inits):
        if time.time() > deadline - 20.0:
            break
        
        # Quick relax
        try:
            energy_relaxed, pos_relaxed = relax(init_pos.copy(), potential, parameter)
        except Exception:
            energy_relaxed = _morse_energy(init_pos, rho)
            pos_relaxed = init_pos.copy()
        
        # L-BFGS-B optimization
        try:
            e_opt, pos_opt = _optimize_local(pos_relaxed, rho, maxiter=400)
            
            # Fix min distance if needed
            if _min_distance(pos_opt) < 0.5:
                pos_opt = _fix_min_distance(pos_opt)
                e_opt = _morse_energy(pos_opt, rho)
            
            if e_opt < best_energy:
                best_energy = e_opt
                best_positions = pos_opt
        except Exception:
            if energy_relaxed < best_energy:
                best_energy = energy_relaxed
                best_positions = pos_relaxed
    
    # If we have time, do basin hopping from the best found
    if best_positions is not None and time.time() < deadline - 30.0:
        current_best = best_positions.copy()
        current_energy = best_energy
        hop_step = 0.3
        max_hops = 50
        
        for hop in range(max_hops):
            if time.time() > deadline - 15.0:
                break
            
            # Perturb
            perturbed = _basin_hop_step(current_best, rng, rho, step_size=hop_step)
            
            # Optimize
            try:
                e_new, pos_new = _optimize_local(perturbed, rho, maxiter=200)
                
                if _min_distance(pos_new) < 0.5:
                    pos_new = _fix_min_distance(pos_new)
                    e_new = _morse_energy(pos_new, rho)
                
                if e_new < current_energy:
                    current_energy = e_new
                    current_best = pos_new
                    best_energy = e_new
                    best_positions = pos_new
                    hop_step = min(hop_step * 1.1, 0.5)
                else:
                    hop_step = max(hop_step * 0.9, 0.1)
            except Exception:
                pass
    
    if best_positions is None:
        energy, positions = relax(inits[0], potential, parameter)
        return positions
    
    return best_positions
# EVOLVE-BLOCK-END
