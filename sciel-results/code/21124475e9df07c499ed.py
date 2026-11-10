"""Morse cluster global minima: multi-chain simulated annealing basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy(pos, rho):
    """Compute Morse energy E = sum_{i<j} exp(rho(1-r_ij))(exp(rho(1-r_ij)) - 2)."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(r, np.inf)
    iu = np.triu_indices(n, k=1)
    r_ij = r[iu]
    x = rho * (1.0 - r_ij)
    e = np.exp(x) * (np.exp(x) - 2.0)
    return float(np.sum(e))


def _morse_energy_and_grad(pos, rho):
    """Compute Morse energy and gradient."""
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(r, np.inf)
    iu = np.triu_indices(n, k=1)
    r_ij = r[iu]
    x = rho * (1.0 - r_ij)
    exp_x = np.exp(x)
    e_ij = exp_x * (exp_x - 2.0)
    energy = float(np.sum(e_ij))
    
    grad = np.zeros_like(pos)
    i_idx, j_idx = iu
    dr_dr = (pos[i_idx] - pos[j_idx]) / r_ij[:, None]
    de_dr = rho * exp_x * (2.0 * exp_x - 2.0)
    np.add.at(grad, i_idx, de_dr[:, None] * dr_dr)
    np.add.at(grad, j_idx, -de_dr[:, None] * dr_dr)
    return energy, grad


def _is_valid(pos, min_dist=0.5):
    """Check all pairwise distances >= min_dist."""
    n = len(pos)
    if n < 2:
        return True
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(r, np.inf)
    return bool(np.min(r) >= min_dist)


def _generate_fcc(n, spacing=1.0):
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


def _generate_bcc(n, spacing=1.0):
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


def _generate_hcp(n, spacing=1.0):
    """Generate HCP lattice points, clipped to sphere."""
    points = []
    a = spacing
    c = a * np.sqrt(8.0 / 3.0)
    max_coord = int(np.ceil(n ** 0.5)) + 2
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                x = a * (i + 0.5 * (j % 2))
                y = a * np.sqrt(3.0) / 2.0 * j
                z = c * k
                points.append(np.array([x, y, z]))
                x_b = a * (i + 0.5 * (j % 2)) + a / 2.0
                y_b = a * np.sqrt(3.0) / 2.0 * j + a * np.sqrt(3.0) / 6.0
                z_b = c * (k + 0.5)
                points.append(np.array([x_b, y_b, z_b]))
    
    points = np.array(points)
    com = np.mean(points, axis=0)
    points -= com
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n]]
    return points


def _generate_random_compact(n, spacing=1.0, rng=None):
    """Generate random compact cluster with minimum distance constraint."""
    if rng is None:
        rng = np.random.default_rng(42)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    for _ in range(10000):
        if len(points) >= n:
            break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.7 * spacing for q in points):
                points.append(p)
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points[:n])


def _local_minimize(pos, rho, max_iter=500):
    """Local minimization using L-BFGS-B."""
    from scipy.optimize import minimize
    try:
        result = minimize(
            lambda x: _morse_energy(x.reshape(-1, 3), rho),
            pos.flatten(),
            jac=lambda x: _morse_energy_and_grad(x.reshape(-1, 3), rho)[1].flatten(),
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        new_pos = result.x.reshape(-1, 3)
        new_energy = _morse_energy(new_pos, rho)
        if _is_valid(new_pos):
            return new_energy, new_pos
    except Exception:
        pass
    return _morse_energy(pos, rho), pos


def _sa_basin_hop_chain(pos0, rho, rng, n_steps, T0, T_final, deadline):
    """Single simulated annealing basin hopping chain."""
    current_energy, current_pos = _local_minimize(pos0, rho)
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        # Temperature schedule: exponential cooling
        alpha = step / max(1, n_steps)
        T = T0 * np.exp(-alpha * np.log(T0 / max(T_final, 1e-10)))
        
        # Adaptive perturbation scale: grows with temperature
        base_scale = 0.3 + 0.7 * (T / T0)
        perturbation = rng.normal(0, base_scale, pos0.shape)
        perturbed = current_pos + perturbation
        perturbed_energy, perturbed_pos = _local_minimize(perturbed, rho)
        
        if perturbed_energy < best_energy:
            best_energy = perturbed_energy
            best_pos = perturbed_pos.copy()
        
        # Metropolis acceptance with temperature
        if perturbed_energy < current_energy:
            current_energy = perturbed_energy
            current_pos = perturbed_pos
        else:
            delta_e = perturbed_energy - current_energy
            if T > 1e-10 and rng.random() < np.exp(-delta_e / T):
                current_energy = perturbed_energy
                current_pos = perturbed_pos
    
    return best_energy, best_pos


def _run_chain(args):
    """Wrapper for parallel execution of a single chain."""
    pos0, rho, seed, n_steps, T0, T_final, deadline = args
    rng = np.random.default_rng(seed)
    return _sa_basin_hop_chain(pos0, rho, rng, n_steps, T0, T_final, deadline)


def construct(instance):
    """Find global minimum using multi-chain simulated annealing basin hopping."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind"))
    
    start_time = time.time()
    deadline = start_time + 275  # Safety margin
    
    rng = np.random.default_rng(size)
    
    # Generate diverse initial configurations
    initial_configs = []
    try:
        fcc = _generate_fcc(size)
        if _is_valid(fcc):
            initial_configs.append(fcc)
    except Exception:
        pass
    
    try:
        bcc = _generate_bcc(size)
        if _is_valid(bcc):
            initial_configs.append(bcc)
    except Exception:
        pass
    
    try:
        hcp = _generate_hcp(size)
        if _is_valid(hcp):
            initial_configs.append(hcp)
    except Exception:
        pass
    
    for _ in range(4):
        try:
            rand = _generate_random_compact(size, rng=rng)
            if _is_valid(rand):
                initial_configs.append(rand)
        except Exception:
            pass
    
    if not initial_configs:
        pos = rng.uniform(-2, 2, (size, 3))
        initial_configs.append(pos)
    
    # Local minimize each initial config to get starting energies
    minimized_configs = []
    for config in initial_configs:
        if time.time() > deadline - 30:
            break
        try:
            e, p = _local_minimize(config, rho, max_iter=200)
            if _is_valid(p):
                minimized_configs.append((e, p))
        except Exception:
            pass
    
    if not minimized_configs:
        return initial_configs[0]
    
    # Sort by energy, take best few for parallel chains
    minimized_configs.sort(key=lambda x: x[0])
    top_configs = [p for e, p in minimized_configs[:4]]
    
    # Determine chain parameters based on remaining time
    remaining_time = deadline - time.time()
    n_chains = min(4, len(top_configs))
    
    # Temperature parameters: T0 scaled by typical energy scale
    typical_energy = abs(minimized_configs[0][0]) + 1.0
    T0 = 0.5 * typical_energy
    T_final = 0.01 * typical_energy
    
    # Each chain gets equal time budget
    time_per_chain = remaining_time / max(1, n_chains)
    # Estimate steps: ~0.3s per step for n=80, scale accordingly
    est_step_time = 0.2 + 0.01 * size
    n_steps = max(10, int(time_per_chain / est_step_time))
    
    # Run chains in parallel using thread pool
    best_energy = np.inf
    best_pos = None
    
    # First, do a quick serial run on the best config to get a baseline
    if time.time() < deadline - 5:
        try:
            e_best, p_best = _sa_basin_hop_chain(
                top_configs[0], rho, np.random.default_rng(size + 100),
                n_steps // 2, T0, T_final, deadline - 5
            )
            if _is_valid(p_best):
                best_energy = e_best
                best_pos = p_best.copy()
        except Exception:
            pass
    
    # Run remaining chains in parallel
    if n_chains > 1 and time.time() < deadline - 10:
        chain_args = []
        for i in range(1, min(n_chains, len(top_configs))):
            seed = size + 200 + i * 7
            chain_args.append((top_configs[i], rho, seed, n_steps // 2, T0, T_final, deadline - 2))
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = {executor.submit(_run_chain, args): i for i, args in enumerate(chain_args)}
                for future in concurrent.futures.as_completed(futures):
                    if time.time() > deadline - 2:
                        break
                    try:
                        e_chain, p_chain = future.result(timeout=5)
                        if _is_valid(p_chain) and e_chain < best_energy:
                            best_energy = e_chain
                            best_pos = p_chain.copy()
                    except Exception:
                        pass
        except Exception:
            pass
    
    # Final refinement
    if best_pos is not None and time.time() < deadline - 2:
        try:
            e_final, pos_final = _local_minimize(best_pos, rho, max_iter=1000)
            if _is_valid(pos_final) and e_final < best_energy:
                best_energy = e_final
                best_pos = pos_final
        except Exception:
            pass
    
    if best_pos is None:
        best_pos = top_configs[0]
    
    return best_pos
# EVOLVE-BLOCK-END
