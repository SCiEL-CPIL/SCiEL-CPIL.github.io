"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _morse_energy(positions, rho):
    """Compute exact Morse energy for a cluster."""
    n = len(positions)
    if n < 2:
        return 0.0
    # Compute all pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=2))
    # Mask out self-interactions
    np.fill_diagonal(r, np.inf)
    # Only compute upper triangle to avoid double counting
    iu = np.triu_indices(n, k=1)
    r_pair = r[iu]
    x = rho * (1.0 - r_pair)
    exp_x = np.exp(x)
    e_pair = exp_x * (exp_x - 2.0)
    return float(np.sum(e_pair))


def _morse_gradient(positions, rho):
    """Compute gradient of Morse energy."""
    n = len(positions)
    grad = np.zeros_like(positions)
    if n < 2:
        return grad
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(r, 1.0)  # avoid division by zero, but these will be masked
    # Compute force for each pair
    # dE/dr = 2 * rho * exp(2*rho*(1-r)) * (exp(rho*(1-r)) - 1)
    # Actually let's derive properly:
    # E_pair = exp(x)(exp(x) - 2) where x = rho(1-r)
    # dE/dr = dE/dx * dx/dr = (exp(x)(exp(x)-2) + exp(x)*exp(x)) * (-rho)
    #       = (exp(2x) - 2exp(x) + exp(2x)) * (-rho)
    #       = (2exp(2x) - 2exp(x)) * (-rho)
    #       = 2rho * exp(x) * (1 - exp(x))
    # Wait, let me redo:
    # E = e^x (e^x - 2) = e^{2x} - 2e^x
    # dE/dx = 2e^{2x} - 2e^x = 2e^x(e^x - 1)
    # dx/dr = -rho
    # dE/dr = 2e^x(e^x - 1) * (-rho) = -2rho * e^x * (e^x - 1)
    # dE/dx_i = dE/dr * dr/dx_i = dE/dr * (x_i - x_j)/r
    # So dE/dpos_i = sum_j dE/dr_ij * (pos_i - pos_j)/r_ij
    
    iu = np.triu_indices(n, k=1)
    r_pair = r[iu]
    x = rho * (1.0 - r_pair)
    exp_x = np.exp(x)
    dE_dr = -2.0 * rho * exp_x * (exp_x - 1.0)
    
    # For each pair (i,j), contribution to grad[i] is dE_dr * (pos[i]-pos[j])/r
    # and to grad[j] is -dE_dr * (pos[i]-pos[j])/r
    i_idx, j_idx = iu
    diff_pair = positions[i_idx] - positions[j_idx]
    factor = dE_dr / r_pair
    contrib = factor[:, np.newaxis] * diff_pair
    
    np.add.at(grad, i_idx, contrib)
    np.add.at(grad, j_idx, -contrib)
    
    return grad


def _basin_hopping(positions, rho, n_iter, rng, time_limit=None, start_time=None):
    """Basin hopping with L-BFGS-B local optimization."""
    if start_time is None:
        start_time = time.time()
    
    current_pos = positions.copy()
    current_energy = _morse_energy(current_pos, rho)
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    T = 0.1  # initial temperature
    T_min = 1e-6
    T_decay = 0.995
    
    for it in range(n_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        # Generate a perturbation
        step_size = 0.3
        perturbed = current_pos + rng.normal(0, step_size, size=current_pos.shape)
        
        # Local optimization using L-BFGS-B
        try:
            result = minimize(
                fun=lambda x: _morse_energy(x, rho),
                x0=perturbed,
                jac=lambda x: _morse_gradient(x, rho),
                method='L-BFGS-B',
                options={'maxiter': 50, 'maxls': 20}
            )
            new_pos = result.x
            new_energy = _morse_energy(new_pos, rho)
        except Exception:
            new_pos = perturbed
            new_energy = _morse_energy(new_pos, rho)
        
        # Metropolis acceptance
        if new_energy < current_energy:
            current_pos = new_pos
            current_energy = new_energy
        else:
            delta_E = new_energy - current_energy
            if delta_E < 0:
                accept = True
            else:
                accept = rng.random() < np.exp(-delta_E / T)
            if accept:
                current_pos = new_pos
                current_energy = new_energy
        
        # Update best
        if current_energy < best_energy:
            best_pos = current_pos.copy()
            best_energy = current_energy
        
        # Decay temperature
        T = max(T * T_decay, T_min)
    
    return best_pos, best_energy


def _make_initial_cluster(n, rng, spacing=1.0):
    """Generate an initial cluster with reasonable packing."""
    points = []
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    max_attempts = n * 100
    
    for _ in range(max_attempts):
        if len(points) >= n:
            break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        min_dist = np.inf
        for q in points:
            d = np.linalg.norm(p - q)
            if d < min_dist:
                min_dist = d
        if min_dist >= 0.5 * spacing or len(points) == 0:
            points.append(p)
    
    # If we couldn't place enough, add random points
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points[:n])


def _try_lattice_start(n, rng, rho, time_budget):
    """Try starting from a lattice-like structure."""
    # Simple cubic grid
    side = int(np.ceil(n ** (1.0 / 3.0)))
    points = []
    for i in range(side):
        for j in range(side):
            for k in range(side):
                if len(points) < n:
                    points.append([i, j, k])
    points = np.array(points[:n], dtype=float)
    # Add some random jitter
    points += rng.normal(0, 0.1, size=points.shape)
    return points


def construct(instance):
    """Find low-energy Morse cluster configurations using basin hopping."""
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(42 + size)
    
    start_time = time.time()
    total_time_limit = 45.0  # Leave buffer for 6 instances in 300s
    
    best_pos = None
    best_energy = np.inf
    
    # Try multiple starting points with basin hopping
    n_starts = 8
    iters_per_start = 30
    
    for start_idx in range(n_starts):
        remaining_time = total_time_limit - (time.time() - start_time)
        if remaining_time < 2.0:
            break
        
        # Different initialization strategies
        if start_idx == 0:
            # Lattice-like start
            pos0 = _try_lattice_start(size, rng, parameter, remaining_time)
        elif start_idx == 1:
            # Random compact
            pos0 = _make_initial_cluster(size, rng, spacing=1.0)
        else:
            # Random with varying density
            density = 0.7 + 0.1 * (start_idx - 2)
            radius = 1.0 * size ** (1.0 / 3.0) * density
            points = []
            max_attempts = size * 200
            for _ in range(max_attempts):
                if len(points) >= size:
                    break
                p = rng.uniform(-radius, radius, 3)
                if np.linalg.norm(p) > radius:
                    continue
                min_dist = np.inf
                for q in points:
                    d = np.linalg.norm(p - q)
                    if d < min_dist:
                        min_dist = d
                if min_dist >= 0.5 or len(points) == 0:
                    points.append(p)
            while len(points) < size:
                p = rng.uniform(-radius, radius, 3)
                points.append(p)
            pos0 = np.array(points[:size])
        
        # Run basin hopping
        bh_time = remaining_time / (n_starts - start_idx) - 0.5
        bh_time = max(bh_time, 1.0)
        
        pos, energy = _basin_hopping(pos0, parameter, iters_per_start, rng, 
                                       time_limit=bh_time, start_time=start_time)
        
        if energy < best_energy:
            best_energy = energy
            best_pos = pos.copy()
    
    # Final polish with more iterations on the best solution
    remaining_time = total_time_limit - (time.time() - start_time)
    if remaining_time > 3.0 and best_pos is not None:
        final_pos, final_energy = _basin_hopping(
            best_pos, parameter, 20, rng,
            time_limit=remaining_time - 1.0, start_time=start_time
        )
        if final_energy < best_energy:
            best_energy = final_energy
            best_pos = final_pos.copy()
    
    if best_pos is None:
        # Fallback: simple random
        best_pos = _make_initial_cluster(size, rng, spacing=1.0)
    
    return best_pos
# EVOLVE-BLOCK-END
