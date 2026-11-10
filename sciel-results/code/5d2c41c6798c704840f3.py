"""Morse cluster global minima: basin hopping with structured crystalline seeds and deadline guard."""
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 280  # Safety margin for 300s limit

def _check_deadline():
    return time.time() > _DEADLINE

def _morse_energy_and_grad(pos, rho):
    """Compute Morse energy and gradient using vectorized operations."""
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff * diff, axis=2))
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    r_masked = r[mask]
    diff_masked = diff[mask]
    
    if len(r_masked) == 0:
        return 0.0, np.zeros_like(pos)
    
    r_safe = np.where(r_masked < 1e-12, 1e-12, r_masked)
    
    exp_term = np.exp(rho * (1.0 - r_safe))
    pair_e = exp_term * (exp_term - 2.0)
    energy = np.sum(pair_e)
    
    dE_dr = exp_term * (2.0 * exp_term - 2.0) * (-rho)
    factor = dE_dr / r_safe
    
    grad = np.zeros_like(pos)
    i_idx, j_idx = np.where(mask)
    np.add.at(grad, i_idx, factor[:, np.newaxis] * diff_masked)
    np.add.at(grad, j_idx, -factor[:, np.newaxis] * diff_masked)
    
    return energy, grad

def _is_valid(pos, min_dist=0.5):
    """Check all pairwise distances >= min_dist."""
    n = len(pos)
    if n < 2:
        return True
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(r, np.inf)
    return bool(np.min(r) >= min_dist)

def _fcc_cluster(n, spacing=1.0):
    """Generate FCC lattice points, clipped to sphere."""
    points = []
    a = spacing
    max_coord = int(np.ceil(n ** 0.5)) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                points.append(np.array([i * a, j * a, k * a]))
                points.append(np.array([(i + 0.5) * a, (j + 0.5) * a, k * a]))
                points.append(np.array([(i + 0.5) * a, j * a, (k + 0.5) * a]))
                points.append(np.array([i * a, (j + 0.5) * a, (k + 0.5) * a]))
    
    points = np.array(points)
    com = np.mean(points, axis=0)
    points -= com
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n]]
    return points

def _bcc_cluster(n, spacing=1.0):
    """Generate BCC lattice points, clipped to sphere."""
    points = []
    a = spacing
    max_coord = int(np.ceil(n ** 0.5)) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                points.append(np.array([i * a, j * a, k * a]))
                points.append(np.array([(i + 0.5) * a, (j + 0.5) * a, (k + 0.5) * a]))
    
    points = np.array(points)
    com = np.mean(points, axis=0)
    points -= com
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n]]
    return points

def _icosahedral(n, spacing=1.0):
    """Generate an icosahedral-like shell structure."""
    points = []
    if n == 1:
        return np.array([[0.0, 0.0, 0.0]])
    
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    radius = spacing * (n ** (1.0/3.0)) * 0.5
    
    for i in range(n):
        z = 1.0 - 2.0 * (i + 0.5) / n
        radius_xy = np.sqrt(1.0 - z * z)
        theta = 2.0 * np.pi * i / golden_ratio
        x = radius_xy * np.cos(theta)
        y = radius_xy * np.sin(theta)
        points.append([x * radius, y * radius, z * radius])
    
    return np.array(points, dtype=float)

def _spherical_shell(n, spacing=1.0, n_shells=2):
    """Generate atoms on concentric spherical shells."""
    points = []
    if n_shells <= 1:
        return _icosahedral(n, spacing)
    
    # Distribute atoms across shells
    per_shell = [n // n_shells] * n_shells
    remainder = n % n_shells
    for i in range(remainder):
        per_shell[i] += 1
    
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    base_radius = spacing * (n ** (1.0/3.0)) * 0.4
    
    for shell_idx in range(n_shells):
        shell_n = per_shell[shell_idx]
        if shell_n == 0:
            continue
        shell_radius = base_radius * (shell_idx + 1) / n_shells
        for i in range(shell_n):
            z = 1.0 - 2.0 * (i + 0.5) / shell_n
            radius_xy = np.sqrt(1.0 - z * z)
            theta = 2.0 * np.pi * i / golden_ratio
            x = radius_xy * np.cos(theta)
            y = radius_xy * np.sin(theta)
            points.append([x * shell_radius, y * shell_radius, z * shell_radius])
    
    return np.array(points[:n], dtype=float)

def _random_compact(n, spacing=1.0, rng=None):
    """Generate random compact cluster with minimum distance constraint."""
    if rng is None:
        rng = np.random.default_rng(42)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    min_dist = 0.7 * spacing
    attempts = 0
    max_attempts = 10000
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if len(points) == 0:
                points.append(p)
            else:
                dists = np.sqrt(np.sum((np.array(points) - p) ** 2, axis=1))
                if np.min(dists) >= min_dist:
                    points.append(p)
        attempts += 1
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points[:n])

def _local_minimize(pos, rho, max_iter=500, ftol=1e-10, gtol=1e-8):
    """Local minimization using L-BFGS-B."""
    try:
        result = minimize(
            lambda x: _morse_energy_and_grad(x.reshape(-1, 3), rho)[0],
            pos.flatten(),
            jac=lambda x: _morse_energy_and_grad(x.reshape(-1, 3), rho)[1].flatten(),
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
        )
        new_pos = result.x.reshape(-1, 3)
        new_energy = _morse_energy_and_grad(new_pos, rho)[0]
        if _is_valid(new_pos):
            return new_energy, new_pos
    except Exception:
        pass
    return _morse_energy_and_grad(pos, rho)[0], pos

def _basin_hop(pos, rho, rng, max_steps=50, perturb_scale=0.3):
    """Basin hopping: random perturbations + local relaxation."""
    current_energy, current_pos = _local_minimize(pos, rho, max_iter=200)
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    non_improving = 0
    for step in range(max_steps):
        if _check_deadline():
            break
        
        # Reset to random seed after too many non-improving steps
        if non_improving >= 20:
            perturbation = rng.normal(0, perturb_scale * 2, pos.shape)
        else:
            perturbation = rng.normal(0, perturb_scale, pos.shape)
        
        perturbed = current_pos + perturbation
        perturbed_energy, perturbed_pos = _local_minimize(perturbed, rho, max_iter=100)
        
        if _is_valid(perturbed_pos) and perturbed_energy < best_energy:
            best_energy = perturbed_energy
            best_pos = perturbed_pos.copy()
            non_improving = 0
        else:
            non_improving += 1
        
        if perturbed_energy < current_energy:
            current_energy = perturbed_energy
            current_pos = perturbed_pos
        else:
            delta_e = perturbed_energy - current_energy
            if rng.random() < np.exp(-delta_e / max(0.01, abs(current_energy) * 0.1)):
                current_energy = perturbed_energy
                current_pos = perturbed_pos
    
    return best_energy, best_pos

def _basin_hop_worker(args):
    """Worker for parallel basin hopping."""
    pos, rho, seed, max_steps, perturb_scale = args
    if _check_deadline():
        return None
    try:
        rng = np.random.default_rng(seed)
        e, p = _basin_hop(pos, rho, rng, max_steps=max_steps, perturb_scale=perturb_scale)
        if _is_valid(p):
            return e, p
    except Exception:
        pass
    return None

def construct(instance):
    """Find global minimum of Morse cluster using basin hopping with diverse seeds."""
    size = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(size)
    
    best_energy = np.inf
    best_pos = None
    
    # Generate diverse starting configurations
    start_configs = []
    
    # FCC start
    try:
        fcc_pos = _fcc_cluster(size)
        if _is_valid(fcc_pos):
            start_configs.append(fcc_pos)
    except Exception:
        pass
    
    # BCC start
    try:
        bcc_pos = _bcc_cluster(size)
        if _is_valid(bcc_pos):
            start_configs.append(bcc_pos)
    except Exception:
        pass
    
    # Icosahedral shell start
    try:
        ico_pos = _icosahedral(size)
        if _is_valid(ico_pos):
            start_configs.append(ico_pos)
    except Exception:
        pass
    
    # Spherical shell start
    try:
        shell_pos = _spherical_shell(size)
        if _is_valid(shell_pos):
            start_configs.append(shell_pos)
    except Exception:
        pass
    
    # Random compact starts
    for i in range(3):
        try:
            rand_pos = _random_compact(size, rng=rng)
            if _is_valid(rand_pos):
                start_configs.append(rand_pos)
        except Exception:
            pass
    
    # First, do local minimization on all starts to get good initial energies
    relaxed_configs = []
    for pos in start_configs:
        if _check_deadline():
            break
        try:
            e, p = _local_minimize(pos, rho, max_iter=300)
            if _is_valid(p):
                relaxed_configs.append((e, p))
                if e < best_energy:
                    best_energy = e
                    best_pos = p.copy()
        except Exception:
            pass
    
    # Strategy 1: Parallel basin hopping from best relaxed configs
    if relaxed_configs and time.time() < _DEADLINE - 40:
        relaxed_configs.sort(key=lambda x: x[0])
        top_configs = relaxed_configs[:3]
        
        tasks = []
        for i, (e, p) in enumerate(top_configs):
            tasks.append((p, rho, size + i * 100, 400, 0.25))
        
        with ThreadPoolExecutor(max_workers=min(4, len(tasks))) as executor:
            results = list(executor.map(_basin_hop_worker, tasks))
            for res in results:
                if res is not None:
                    e, p = res
                    if e < best_energy:
                        best_energy = e
                        best_pos = p.copy()
    
    # Strategy 2: Final local refinement with tighter tolerances
    if best_pos is not None and time.time() < _DEADLINE - 15:
        try:
            e, p = _local_minimize(best_pos, rho, max_iter=400, ftol=1e-13, gtol=1e-10)
            if _is_valid(p) and e < best_energy:
                best_energy = e
                best_pos = p.copy()
        except Exception:
            pass
    
    # Strategy 3: Additional basin hopping from best result if time permits
    if best_pos is not None and time.time() < _DEADLINE - 10:
        try:
            e_bh, pos_bh = _basin_hop(best_pos, rho, rng, max_steps=15, perturb_scale=0.15)
            if e_bh < best_energy and _is_valid(pos_bh):
                best_energy = e_bh
                best_pos = pos_bh.copy()
        except Exception:
            pass
    
    # Ultimate fallback
    if best_pos is None:
        best_pos = rng.uniform(-2, 2, (size, 3))
    
    return best_pos
