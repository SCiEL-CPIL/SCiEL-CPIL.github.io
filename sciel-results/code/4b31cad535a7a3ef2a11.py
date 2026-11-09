"""Sutton-Chen transition-metal cluster global minima: Coarse-to-Fine Annealing Basin Hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Per-kind parameters: (n_exp, m_exp, c)
_KIND_PARAMS = {
    '12-6': (12, 6, 144.41),
    '9-6': (9, 6, 39.432),
    '10-8': (10, 8, 34.408),
}

# Spacing per kind for seed generation
_SPACING = {
    '12-6': 0.72,
    '9-6': 0.72,
    '10-8': 0.72,
}

# Soft potential parameters for basin exploration
# We reduce exponents to flatten the landscape
# (n_target, m_target, c_target) -> (n_start, m_start, c_scale_start)
# We will linearly interpolate from start to target during annealing
_SOFT_PARAMS = {
    '12-6': {'n_start': 6.0, 'm_start': 3.0, 'c_scale_start': 0.5},
    '9-6': {'n_start': 4.5, 'm_start': 3.0, 'c_scale_start': 0.5},
    '10-8': {'n_start': 5.0, 'm_start': 4.0, 'c_scale_start': 0.5},
}


def _validate_positions(pos):
    """Vectorized validity check: finite coords and min pair distance >= 0.35."""
    if not np.all(np.isfinite(pos)):
        return False
    n = pos.shape[0]
    if n < 2:
        return True
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= 0.35


def _fcc_cluster(n, spacing=0.72):
    """Generate n-atom FCC-like cluster."""
    if n <= 0:
        return np.empty((0, 3))
    basis = np.array([[0, 0, 0],
                      [0, 0.5, 0.5],
                      [0.5, 0, 0.5],
                      [0.5, 0.5, 0]], dtype=float)
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 4
    sites = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                for b in basis:
                    p = np.array([i, j, k], dtype=float) + b
                    sites.append(p)
    sites = np.array(sites)
    sites = sites * spacing
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)
    return sites[idx[:n]]


def _hcp_cluster(n, spacing=0.72):
    """Generate n-atom HCP-like cluster."""
    if n <= 0:
        return np.empty((0, 3))
    a = spacing
    c = spacing * np.sqrt(8.0 / 3.0)
    v1 = np.array([a, 0, 0])
    v2 = np.array([-a / 2, a * np.sqrt(3) / 2, 0])
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 4
    sites = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                base = i * v1 + j * v2
                z = k * c
                p1 = base + np.array([0, 0, z])
                p2 = base + np.array([a / 2, a * np.sqrt(3) / 6, z + c / 2])
                sites.append(p1)
                sites.append(p2)
    sites = np.array(sites)
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)
    return sites[idx[:n]]


def _bcc_cluster(n, spacing=0.72):
    """Generate n-atom BCC-like cluster."""
    if n <= 0:
        return np.empty((0, 3))
    a = spacing
    max_shell = int(np.ceil(n ** (1.0 / 3.0))) + 4
    sites = []
    for i in range(-max_shell, max_shell + 1):
        for j in range(-max_shell, max_shell + 1):
            for k in range(-max_shell, max_shell + 1):
                sites.append(np.array([i, j, k], dtype=float) * a)
                sites.append((np.array([i, j, k], dtype=float) + 0.5) * a)
    sites = np.array(sites)
    dists = np.linalg.norm(sites, axis=1)
    idx = np.argsort(dists)
    return sites[idx[:n]]


def _icosahedral_cluster(n, spacing=0.72):
    """Generate n-atom icosahedral-like cluster."""
    if n <= 0:
        return np.empty((0, 3))
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ], dtype=float)
    ico_verts = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)

    points = [np.zeros(3)]
    if n <= 1:
        return np.array(points[:n])

    r1 = spacing
    for v in ico_verts:
        points.append(v * r1)
    if len(points) >= n:
        return np.array(points[:n])

    r2 = spacing * np.sqrt(3.0)
    ico_norms = ico_verts / np.linalg.norm(ico_verts, axis=1, keepdims=True)
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(ico_norms[i] - ico_norms[j])
            if 1.0 < d < 2.0:
                mid = (ico_norms[i] + ico_norms[j]) / 2.0
                mid = mid / np.linalg.norm(mid)
                points.append(mid * r2)
    if len(points) >= n:
        return np.array(points[:n])

    r3 = spacing * np.sqrt(5.0)
    remaining = n - len(points)
    if remaining > 0:
        for k in range(remaining):
            t = (k + 0.5) / remaining
            inc = np.pi * (3 - np.sqrt(5))
            z = 1 - 2 * t
            r = np.sqrt(1 - z * z)
            theta = inc * k
            p = np.array([r * np.cos(theta), r * np.sin(theta), z])
            points.append(p * r3)

    return np.array(points[:n])


def _random_compact(n, spacing=0.72, seed=42):
    """Generate a random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    max_attempts = n * 500
    attempt = 0
    while len(points) < n and attempt < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.7 * spacing for q in points):
                points.append(p)
        attempt += 1
    while len(points) < n:
        base = points[len(points) % max(1, len(points))] if points else np.zeros(3)
        p = base + rng.normal(0, spacing * 0.3, 3)
        points.append(p)
    return np.array(points[:n])


def _soft_energy_gradient(x, n, n_s, m_s, c_s):
    """
    Compute energy and gradient for the softened Sutton-Chen potential.
    x: (n, 3) array
    """
    x = x.reshape(n, 3)
    # Pair indices
    i_idx, j_idx = np.triu_indices(n, k=1)
    
    # Distances
    diff = x[i_idx] - x[j_idx]
    r2 = np.sum(diff * diff, axis=1)
    r2 = np.maximum(r2, 1e-10)
    r = np.sqrt(r2)
    
    # Repulsive part: sum_{i<j} r^-n_s
    r_inv_n = r ** (-n_s)
    E_rep = np.sum(r_inv_n)
    
    # Attractive part: -c_s * sum_i sqrt(sum_{j!=i} r^-m_s)
    # We need S_i = sum_{j!=i} r_ij^-m_s
    S = np.zeros(n)
    r_inv_m = r ** (-m_s)
    
    # Add to S[i] and S[j]
    np.add.at(S, i_idx, r_inv_m)
    np.add.at(S, j_idx, r_inv_m)
    
    # S[i] is sum_{j!=i} r_ij^-m_s
    sqrt_S = np.sqrt(np.maximum(S, 1e-30))
    E_att = -c_s * np.sum(sqrt_S)
    
    E = E_rep + E_att
    
    # Gradient
    grad = np.zeros((n, 3))
    
    # Repulsive gradient contribution
    coeff_rep = -n_s * r ** (-n_s - 2)
    # diff is x_i - x_j
    forces_rep = coeff_rep[:, np.newaxis] * diff
    # Add to grad[i] and subtract from grad[j]
    np.add.at(grad, i_idx, forces_rep)
    np.add.at(grad, j_idx, -forces_rep)
    
    # Attractive gradient contribution
    # E_att = -c_s * sum_i sqrt(S_i)
    # dE_att/dx_k = -c_s * sum_i (1/(2 sqrt(S_i))) * dS_i/dx_k
    
    # Precompute inv_2sqrt_S
    inv_2sqrt_S = np.zeros(n)
    mask = S > 1e-10
    inv_2sqrt_S[mask] = 1.0 / (2.0 * np.sqrt(S[mask]))
    
    # Initialize gradient accumulator for attractive part
    grad_att = np.zeros((n, 3))
    
    # Loop over pairs (n^2/2 ~ 3000, very fast)
    for idx in range(len(i_idx)):
        i = i_idx[idx]
        j = j_idx[idx]
        r_val = r[idx]
        # coeff_att is -m_s * r^(-m_s-2)
        c_att = -m_s * r_val ** (-m_s - 2)
        d_vec = diff[idx] # x_i - x_j
        
        # dS_i/dx_i part: c_att * d_vec
        # dS_i/dx_j part: c_att * (-d_vec)
        # dS_j/dx_j part: c_att * (-d_vec)
        # dS_j/dx_i part: c_att * d_vec
        
        # Contribution to x_i:
        # from i's sum: inv_2sqrt_S[i] * (c_att * d_vec)
        # from j's sum: inv_2sqrt_S[j] * (c_att * d_vec)
        grad_att[i] += -c_s * (inv_2sqrt_S[i] + inv_2sqrt_S[j]) * c_att * d_vec
        
        # Contribution to x_j:
        # from i's sum: inv_2sqrt_S[i] * (c_att * (-d_vec))
        # from j's sum: inv_2sqrt_S[j] * (c_att * (-d_vec))
        grad_att[j] += -c_s * (inv_2sqrt_S[i] + inv_2sqrt_S[j]) * c_att * (-d_vec)
        
    grad += grad_att
    
    return E, grad


def _anneal_and_relax(pos, kind, n_target, m_target, c_target, deadline, rng):
    """
    Perform coarse-to-fine annealing followed by hard relaxation.
    Returns (energy, positions) of the best result found.
    """
    n_exp, m_exp, c_exp = _KIND_PARAMS[kind]
    soft_params = _SOFT_PARAMS[kind]
    n_start = soft_params['n_start']
    m_start = soft_params['m_start']
    c_start = c_exp * soft_params['c_scale_start']
    
    # Number of annealing steps
    # We want to be fast. 10 steps is reasonable.
    n_steps = 10
    
    current_pos = pos.copy()
    best_energy = np.inf
    best_pos = None
    
    # Temperature for basin hopping
    # Start with a larger temperature for coarse steps
    T_start = 0.5
    T_end = 0.1
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
            
        # Linear interpolation of parameters
        alpha = step / (n_steps - 1) if n_steps > 1 else 1.0
        n_s = n_start + alpha * (n_target - n_start)
        m_s = m_start + alpha * (m_target - m_start)
        c_s = c_start + alpha * (c_target - c_start)
        
        T = T_start + alpha * (T_end - T_start)
        
        # Define the objective function for this step
        def func(x):
            E, G = _soft_energy_gradient(x, len(current_pos), n_s, m_s, c_s)
            if not np.isfinite(E):
                return 1e10, np.zeros_like(x)
            return E, G
        
        # Relax using L-BFGS-B
        try:
            x0 = current_pos.flatten()
            res = minimize(func, x0, method='L-BFGS-B', jac=True, options={'maxiter': 100, 'ftol': 1e-8})
            if res.success or np.isfinite(res.fun):
                new_pos = res.x.reshape(len(current_pos), 3)
                if _validate_positions(new_pos):
                    current_pos = new_pos
                    
                    # Evaluate on hard potential to track best
                    # We only do this if we are getting close to target or periodically
                    # To save time, we can check hard energy every few steps or at the end
                    if step % 2 == 0 or step == n_steps - 1:
                        try:
                            hard_energy, hard_pos = relax(current_pos, 'sutton_chen', kind)
                            if np.isfinite(hard_energy) and _validate_positions(hard_pos):
                                if hard_energy < best_energy:
                                    best_energy = hard_energy
                                    best_pos = hard_pos
                        except Exception:
                            pass
        except Exception:
            pass
            
        # Basin Hopping Step: Perturb and accept/reject
        # We perturb the current soft-relaxed position
        perturbation_scale = T * 0.5
        perturbed_pos = current_pos + rng.normal(0, perturbation_scale, current_pos.shape)
        
        if _validate_positions(perturbed_pos):
            # Try to relax the perturbed position
            try:
                x0 = perturbed_pos.flatten()
                res = minimize(func, x0, method='L-BFGS-B', jac=True, options={'maxiter': 50, 'ftol': 1e-8})
                if res.success or np.isfinite(res.fun):
                    new_pos = res.x.reshape(len(current_pos), 3)
                    if _validate_positions(new_pos):
                        # Metropolis acceptance
                        # We need energies. Let's use the soft energy for acceptance during annealing
                        # E_old is approx res.fun from previous step? No, we need E_old of current_pos under current params
                        # Let's just always accept if valid in coarse phase, or use simple hill climbing in fine phase
                        # For simplicity and speed, let's just take the lower soft energy
                        # Actually, let's evaluate soft energy of new_pos
                        E_new, _ = _soft_energy_gradient(new_pos, len(current_pos), n_s, m_s, c_s)
                        E_old, _ = _soft_energy_gradient(current_pos, len(current_pos), n_s, m_s, c_s)
                        
                        if E_new < E_old:
                            current_pos = new_pos
                        else:
                            # Metropolis
                            dE = E_new - E_old
                            if dE < 0 or rng.random() < np.exp(-dE / T):
                                current_pos = new_pos
                                
            except Exception:
                pass

    # Final Hard Relaxation on the best candidate found or the final annealed position
    if best_pos is not None:
        # Try to polish the best hard position
        try:
            final_energy, final_pos = relax(best_pos, 'sutton_chen', kind)
            if np.isfinite(final_energy) and _validate_positions(final_pos):
                if final_energy < best_energy:
                    best_energy = final_energy
                    best_pos = final_pos
        except Exception:
            pass
    else:
        # If no intermediate hard check succeeded, do one at the end
        try:
            final_energy, final_pos = relax(current_pos, 'sutton_chen', kind)
            if np.isfinite(final_energy) and _validate_positions(final_pos):
                best_energy = final_energy
                best_pos = final_pos
        except Exception:
            pass

    return best_energy, best_pos


def construct(instance):
    """
    Coarse-to-Fine Annealing Basin Hopping.
    1. Generate diverse seeds.
    2. For each seed, perform annealing from soft to hard potential.
    3. Keep the best result.
    """
    start_time = time.time()
    deadline = start_time + 290.0
    
    size = instance["n"]
    potential = instance["potential"]
    kind = instance.get("kind", instance.get("rho"))
    spacing = _SPACING.get(kind, 0.72)
    
    n_exp, m_exp, c_exp = _KIND_PARAMS[kind]
    
    # Generate seeds
    seeds = []
    for scale in [0.85, 1.0, 1.15]:
        s = spacing * scale
        seeds.append(_fcc_cluster(size, s))
        seeds.append(_hcp_cluster(size, s))
        seeds.append(_bcc_cluster(size, s))
        seeds.append(_icosahedral_cluster(size, s))
        
    for rs in [size, size + 1, size + 7, size + 13, size + 42]:
        seeds.append(_random_compact(size, spacing, seed=rs))
        
    # Remove duplicates and validate
    unique_seeds = []
    seen = set()
    for s in seeds:
        if s.shape[0] != size:
            continue
        key = tuple(np.round(s, 3).flatten())
        if key not in seen:
            seen.add(key)
            if _validate_positions(s):
                unique_seeds.append(s)
                
    best_energy = np.inf
    best_positions = None
    rng = np.random.default_rng(42)
    
    # Process all seeds
    # We have ~18 seeds.
    # Each anneal takes maybe 0.5-2s depending on n.
    # 18 * 2s = 36s. We have 290s. So we can do multiple rounds or more seeds.
    # Let's do 2 rounds of seeds, or just process all seeds and if time permits, perturb best.
    
    for seed in unique_seeds:
        if time.time() > deadline - 10.0:
            break
            
        energy, positions = _anneal_and_relax(seed, kind, n_exp, m_exp, c_exp, deadline, rng)
        
        if np.isfinite(energy) and positions is not None and _validate_positions(positions):
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
                
    # If no valid result, fallback
    if best_positions is None:
        positions = np.zeros((size, 3))
        positions[:, 0] = np.arange(size) * spacing
        best_positions = positions
        best_energy = np.inf
        
    # Polish phase: Perturb best result and re-process
    # Use remaining time for local search around the best solution
    if best_positions is not None and time.time() < deadline - 5.0:
        # Use the soft potential with parameters close to target for polishing
        # Or just use the provided relax which is robust
        for perturbation_scale in [0.1, 0.2, 0.3, 0.5, 0.8]:
            if time.time() > deadline - 5.0:
                break
            perturbed = best_positions + rng.normal(0, perturbation_scale * spacing, best_positions.shape)
            if _validate_positions(perturbed):
                try:
                    energy, positions = relax(perturbed, potential, kind)
                    if np.isfinite(energy) and _validate_positions(positions):
                        if energy < best_energy:
                            best_energy = energy
                            best_positions = positions
                except Exception:
                    pass

    # Final validation
    if not _validate_positions(best_positions):
        best_positions = np.zeros((size, 3))
        best_positions[:, 0] = np.arange(size) * spacing
        
    return best_positions
# EVOLVE-BLOCK-END
