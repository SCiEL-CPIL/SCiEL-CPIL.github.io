"""
Strategy: Basin Hopping with Polytetrahedral Seeding and L-BFGS-B Local Minimization.
The Dzugutov potential is designed to favor PT order (icosahedral shells).
Instead of relying on a single, potentially slow or trapped `relax` call from a few starts,
we implement a robust search:
1. Generate diverse initial structures (PT shells, random compact, perturbed PT).
2. Use a parallel pool (up to 4 threads) to run independent basin hopping chains.
3. Each chain uses L-BFGS-B (via SciPy) for fast, accurate local minimization,
   combined with random perturbations to escape local minima.
4. We track the global best energy and return the corresponding positions.
This active, parallel search allows escaping local minima that trap single-shot relaxations,
exploiting the parallelism budget to explore the energy landscape more broadly.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
import threading
from scipy.optimize import minimize

# Global deadline tracking
_START_TIME = time.time()
_HARD_LIMIT = 290.0  # Safety margin for 300s limit
_lock = threading.Lock()

def _get_remaining_time():
    return max(0.1, _HARD_LIMIT - (time.time() - _START_TIME))

def _dzugutov_energy(positions, A=5.82, B=1.28, a=1.87, b=1.94, c=1.10, d=0.27, m=16):
    """
    Compute the energy of the Dzugutov potential.
    Vectorized for speed.
    """
    n = len(positions)
    if n < 2:
        return 0.0
    
    # Compute pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    
    # Mask for i < j
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    dist = dist[mask]
    
    if len(dist) == 0:
        return 0.0
    
    # Avoid division by zero
    dist = np.maximum(dist, 1e-8)
    
    energy = 0.0
    
    # Term 1: A(r^-m - B)exp(c/(r - a)) [r < a]
    mask_a = dist < a
    if np.any(mask_a):
        r_a = dist[mask_a]
        v_a = A * (r_a ** (-m) - B) * np.exp(c / (r_a - a))
        energy += np.sum(v_a)
        
    # Term 2: B exp(d/(r - b)) [r < b]
    mask_b = dist < b
    if np.any(mask_b):
        r_b = dist[mask_b]
        v_b = B * np.exp(d / (r_b - b))
        energy += np.sum(v_b)
        
    return energy

def _dzugutov_energy_gradient(positions, A=5.82, B=1.28, a=1.87, b=1.94, c=1.10, d=0.27, m=16):
    """
    Compute the gradient of the Dzugutov potential.
    """
    n = len(positions)
    if n < 2:
        return np.zeros_like(positions)
    
    # Compute pairwise distances and vectors
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    
    # Mask for i < j
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    dist = dist[mask]
    diff = diff[mask]
    
    if len(dist) == 0:
        return np.zeros_like(positions)
    
    # Avoid division by zero
    dist_safe = np.maximum(dist, 1e-8)
    
    grad = np.zeros((n, 3))
    
    # Term 1: A(r^-m - B)exp(c/(r - a)) [r < a]
    # V(r) = A(r^-m - B)exp(c/(r - a))
    # dV/dr = A(-m*r^(-m-1) * exp(c/(r-a)) + (r^-m - B) * exp(c/(r-a)) * (-c/(r-a)^2))
    # dV/dr = A * exp(c/(r-a)) * (-m*r^(-m-1) - c*(r^-m - B)/(r-a)^2)
    
    mask_a = dist < a
    if np.any(mask_a):
        r_a = dist_safe[mask_a]
        diff_a = diff[mask_a]
        
        exp_term = np.exp(c / (r_a - a))
        # dV/dr
        dv_dr = A * exp_term * (-m * r_a ** (-m - 1) - c * (r_a ** (-m) - B) / (r_a - a) ** 2)
        
        # Force = -dV/dr * (r_vec / r)
        # Grad_i += -F_ij = dV/dr * (r_vec / r)
        # Grad_j -= -F_ij = -dV/dr * (r_vec / r)
        
        factor = dv_dr / r_a[:, np.newaxis]
        
        # diff_a is r_i - r_j
        # grad_i += factor * diff_a
        # grad_j -= factor * diff_a
        
        indices = np.where(mask)[0]
        i_idx = indices // n
        j_idx = indices % n
        
        np.add.at(grad, i_idx, factor * diff_a)
        np.add.at(grad, j_idx, -factor * diff_a)
        
    # Term 2: B exp(d/(r - b)) [r < b]
    # V(r) = B exp(d/(r - b))
    # dV/dr = B exp(d/(r - b)) * (-d/(r-b)^2)
    
    mask_b = dist < b
    if np.any(mask_b):
        r_b = dist_safe[mask_b]
        diff_b = diff[mask_b]
        
        exp_term = np.exp(d / (r_b - b))
        dv_dr = B * exp_term * (-d / (r_b - b) ** 2)
        
        factor = dv_dr / r_b[:, np.newaxis]
        
        indices = np.where(mask)[0]
        # Need to re-extract indices for mask_b specifically from the flattened mask
        # It's easier to just recompute or use the full mask logic carefully.
        # Let's recompute indices for mask_b
        flat_mask_b = np.triu(np.ones((n, n), dtype=bool), k=1)
        flat_mask_b = flat_mask_b & (dist < b)
        # Wait, dist was masked. Let's just use the original diff/dist.
        
        # Let's restart the gradient calculation more cleanly to avoid index confusion.
        # This function is critical for L-BFGS-B.
        
    # Let's rewrite the gradient calculation to be robust.
    grad = np.zeros((n, 3))
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    # Flatten indices
    i_idx, j_idx = np.where(mask)
    
    if len(i_idx) == 0:
        return grad
        
    r = dist[i_idx, j_idx]
    r_safe = np.maximum(r, 1e-8)
    dvec = diff[i_idx, j_idx]
    
    # Term 1
    mask_a = r < a
    if np.any(mask_a):
        r_a = r_safe[mask_a]
        dvec_a = dvec[mask_a]
        i_a = i_idx[mask_a]
        j_a = j_idx[mask_a]
        
        exp_term = np.exp(c / (r_a - a))
        dv_dr = A * exp_term * (-m * r_a ** (-m - 1) - c * (r_a ** (-m) - B) / (r_a - a) ** 2)
        factor = dv_dr / r_a
        
        np.add.at(grad, i_a, factor[:, np.newaxis] * dvec_a)
        np.add.at(grad, j_a, -factor[:, np.newaxis] * dvec_a)
        
    # Term 2
    mask_b = r < b
    if np.any(mask_b):
        r_b = r_safe[mask_b]
        dvec_b = dvec[mask_b]
        i_b = i_idx[mask_b]
        j_b = j_idx[mask_b]
        
        exp_term = np.exp(d / (r_b - b))
        dv_dr = B * exp_term * (-d / (r_b - b) ** 2)
        factor = dv_dr / r_b
        
        np.add.at(grad, i_b, factor[:, np.newaxis] * dvec_b)
        np.add.at(grad, j_b, -factor[:, np.newaxis] * dvec_b)
        
    return grad

def _penalized_energy_and_grad(positions, penalty_strength=1000.0):
    """
    Energy and gradient with a penalty for atoms closer than 0.5 Å.
    """
    base_energy = _dzugutov_energy(positions)
    base_grad = _dzugutov_energy_gradient(positions)
    
    n = len(positions)
    if n < 2:
        return base_energy, base_grad
        
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    i_idx, j_idx = np.where(mask)
    r = dist[i_idx, j_idx]
    dvec = diff[i_idx, j_idx]
    
    close_mask = r < 0.5
    if np.any(close_mask):
        r_c = r[close_mask]
        dvec_c = dvec[close_mask]
        i_c = i_idx[close_mask]
        j_c = j_idx[close_mask]
        
        # Penalty: sum (0.5 - r)^2
        # dP/dr = 2(0.5 - r)(-1) = -2(0.5 - r) = 2(r - 0.5)
        # Force = -dP/dr * (r_vec/r) = -2(r-0.5) * (r_vec/r)
        # Grad_i += Force = -2(r-0.5)/r * r_vec
        # Grad_j -= Force
        
        r_safe = np.maximum(r_c, 1e-8)
        factor = -2 * (r_c - 0.5) / r_safe
        
        penalty_energy = penalty_strength * np.sum((0.5 - r_c) ** 2)
        penalty_grad = np.zeros((n, 3))
        
        np.add.at(penalty_grad, i_c, (factor * penalty_strength)[:, np.newaxis] * dvec_c)
        np.add.at(penalty_grad, j_c, -(factor * penalty_strength)[:, np.newaxis] * dvec_c)
        
        return base_energy + penalty_energy, base_grad + penalty_grad
    else:
        return base_energy, base_grad

def _generate_icosahedron(n_atoms, rng):
    """
    Generate a structure based on icosahedral shells.
    """
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    positions = [np.zeros(3)]
    
    r1 = 1.9
    for v in verts:
        positions.append(v * r1)
        
    if n_atoms <= 13:
        return np.array(positions[:n_atoms])
        
    # For n > 13, add atoms on a second shell
    r2 = 1.9 * 1.7
    num_outer = n_atoms - 13
    if num_outer > 0:
        outer_pts = []
        for _ in range(num_outer):
            v = rng.normal(size=3)
            v = v / np.linalg.norm(v) * r2
            outer_pts.append(v)
        positions.extend(outer_pts)
        
    return np.array(positions[:n_atoms])

def _random_compact(n_atoms, rng, spacing=1.1):
    """
    Generate a random compact structure.
    """
    radius = spacing * n_atoms ** (1.0 / 3.0) * 0.8
    points = []
    max_iters = 10000
    for _ in range(max_iters):
        if len(points) >= n_atoms:
            break
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.6 * spacing for q in points):
            points.append(p)
    # If we didn't get enough, just add random points
    while len(points) < n_atoms:
        points.append(rng.uniform(-radius, radius, 3))
    return np.array(points)

def _basin_hopping_chain(n_atoms, rng, initial_pos, time_limit):
    """
    Run a single Basin Hopping chain.
    """
    start_time = time.time()
    current_pos = initial_pos.copy()
    current_energy, _ = _penalized_energy_and_grad(current_pos)
    
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    # Basin hopping parameters
    step_scale = 0.5
    max_steps = 100
    
    for step in range(max_steps):
        if time.time() - start_time > time_limit:
            break
            
        # Perturb
        new_pos = current_pos + rng.normal(0, step_scale, size=current_pos.shape)
        
        # Local Minimization using L-BFGS-B
        # We need to flatten the positions for scipy.optimize.minimize
        def objective(flat_x):
            pos = flat_x.reshape(n_atoms, 3)
            e, g = _penalized_energy_and_grad(pos)
            return e, g
            
        try:
            res = minimize(objective, new_pos.flatten(), method='L-BFGS-B', jac=True, options={'maxiter': 50})
            new_pos = res.x.reshape(n_atoms, 3)
            new_energy = res.fun
        except Exception:
            new_energy, _ = _penalized_energy_and_grad(new_pos)
            
        # Metropolis criterion
        delta_E = new_energy - current_energy
        # Use a simple temperature schedule or just accept if better
        # For basin hopping, we often use a simple accept if better, or Metropolis.
        # Let's use Metropolis with a decaying temperature.
        T = 0.1 * (1.0 - step / max_steps) + 0.01
        if delta_E < 0 or rng.uniform() < np.exp(-delta_E / T):
            current_pos = new_pos
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
                
        # Decay step scale slightly
        step_scale *= 0.99
        
    return best_pos, best_energy

def construct(instance):
    size = instance["n"]
    potential = instance["potential"]
    
    start_time = time.time()
    time_budget = _get_remaining_time()
    
    rng = np.random.default_rng(size)
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse starts
    starts = []
    
    # 1. PT-inspired
    try:
        pt_pos = _generate_icosahedron(size, rng)
        starts.append(pt_pos)
    except Exception:
        pass
        
    # 2. Random Compact
    try:
        rc_pos = _random_compact(size, rng)
        starts.append(rc_pos)
    except Exception:
        pass
        
    # 3. Perturbed PT
    try:
        pt_pos2 = _generate_icosahedron(size, rng)
        pt_pos2 = pt_pos2 + rng.normal(0, 0.3, size=pt_pos2.shape)
        starts.append(pt_pos2)
    except Exception:
        pass
        
    # 4. Another Random Compact
    try:
        rc_pos2 = _random_compact(size, rng)
        starts.append(rc_pos2)
    except Exception:
        pass
        
    # Use parallel pool
    n_workers = min(4, len(starts))
    time_per_chain = time_budget / (n_workers * 1.5)  # Leave some margin
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for i in range(n_workers):
            initial_pos = starts[i]
            # Create a new RNG for each thread to avoid contention
            chain_rng = np.random.default_rng(size + i)
            future = executor.submit(_basin_hopping_chain, size, chain_rng, initial_pos, time_per_chain)
            futures.append(future)
            
        for future in concurrent.futures.as_completed(futures):
            try:
                pos, energy = future.result()
                if energy < best_energy:
                    best_energy = energy
                    best_positions = pos
            except Exception:
                pass
                
    # If no valid result, fallback to the first start
    if best_positions is None:
        best_positions = starts[0]
        
    return best_positions
# EVOLVE-BLOCK-END
