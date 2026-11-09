"""Coarse-to-fine annealing for Sutton-Chen transition-metal clusters."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
import time

def _sc_energy_gradient(positions, n_exp, m_exp, c):
    """Compute Sutton-Chen energy and gradient with given parameters."""
    n = positions.shape[0]
    if n < 2:
        return 0.0, np.zeros_like(positions)
    
    # Pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff**2, axis=-1))
    # Avoid division by zero
    mask = r > 1e-10
    r_safe = np.where(mask, r, 1.0)
    
    # r^-n and r^-m
    rn = r_safe ** (-n_exp)
    rm = r_safe ** (-m_exp)
    
    # Sum over j != i for each i
    # Set diagonal to 0
    rn[~mask] = 0
    rm[~mask] = 0
    
    phi_i = np.sum(rn, axis=1)  # sum_j r_ij^-n
    psi_i = np.sum(rm, axis=1)  # sum_j r_ij^-m
    
    # Energy: E = sum_i [1/2 sum_{j!=i} r_ij^-n_exp - c sqrt(sum_{j!=i} r_ij^-m_exp)]
    E_rep = 0.5 * np.sum(phi_i)
    E_att = -c * np.sum(np.sqrt(psi_i))
    energy = E_rep + E_att
    
    # Gradient
    rn2 = r_safe ** (-(n_exp + 2))
    rm2 = r_safe ** (-(m_exp + 2))
    rn2[~mask] = 0
    rm2[~mask] = 0
    
    sum_rn2 = np.sum(rn2, axis=1)
    dot_rn2_x = rn2 @ positions  # shape (n, 3)
    
    grad_rep = -0.5 * n_exp * (positions * sum_rn2[:, np.newaxis] - dot_rn2_x)
    
    sum_rm2 = np.sum(rm2, axis=1)
    dot_rm2_x = rm2 @ positions
    
    # Avoid division by zero when psi_i is small
    sqrt_psi = np.sqrt(np.maximum(psi_i, 1e-10))
    grad_att = c * m_exp / (2.0 * sqrt_psi[:, np.newaxis]) * (positions * sum_rm2[:, np.newaxis] - dot_rm2_x)
    
    gradient = grad_rep + grad_att
    
    return energy, gradient


def _sc_energy(positions, n_exp, m_exp, c):
    """Compute Sutton-Chen energy only (faster, no gradient)."""
    n = positions.shape[0]
    if n < 2:
        return 0.0
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff**2, axis=-1))
    mask = r > 1e-10
    r_safe = np.where(mask, r, 1.0)
    
    rn = r_safe ** (-n_exp)
    rm = r_safe ** (-m_exp)
    rn[~mask] = 0
    rm[~mask] = 0
    
    phi_i = np.sum(rn, axis=1)
    psi_i = np.sum(rm, axis=1)
    
    E_rep = 0.5 * np.sum(phi_i)
    E_att = -c * np.sum(np.sqrt(psi_i))
    return E_rep + E_att


def _local_relax(positions, n_exp, m_exp, c, maxiter=300):
    """Local relaxation using L-BFGS-B with the given parameters."""
    def fun(x):
        pos = x.reshape(-1, 3)
        e, g = _sc_energy_gradient(pos, n_exp, m_exp, c)
        return e, g.ravel()
    
    try:
        result = minimize(fun, positions.ravel(), method='L-BFGS-B', jac=True,
                          options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
        return result.fun, result.x.reshape(-1, 3)
    except:
        return _sc_energy(positions, n_exp, m_exp, c), positions


def _generate_initial(n, rng, spacing=0.75):
    """Generate a compact random initial cluster."""
    radius = spacing * n ** (1.0 / 3.0) * 0.9
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 200:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.55 * spacing for q in points):
                points.append(p)
        attempts += 1
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)


def _basin_hop_single(positions, n_exp, m_exp, c, temperature, n_hops, maxiter_per_relax, rng):
    """
    Single basin hopping run with soft potential.
    Returns best energy and positions found.
    """
    best_e, best_pos = _local_relax(positions, n_exp, m_exp, c, maxiter=maxiter_per_relax)
    
    current_pos = best_pos
    current_e = best_e
    
    for hop in range(n_hops):
        # Perturb with Gaussian noise scaled by temperature
        noise_scale = temperature * 0.4
        perturbed = current_pos + rng.normal(0, noise_scale, size=current_pos.shape)
        
        # Relax the perturbed structure
        e_new, pos_new = _local_relax(perturbed, n_exp, m_exp, c, maxiter=maxiter_per_relax)
        
        # Metropolis accept/reject
        delta_e = e_new - current_e
        if delta_e < 0 or rng.random() < np.exp(-delta_e / max(temperature, 1e-10)):
            current_pos = pos_new
            current_e = e_new
        
        # Track best
        if e_new < best_e:
            best_e = e_new
            best_pos = pos_new
    
    return best_e, best_pos


def construct(instance):
    """
    Coarse-to-fine annealing for Sutton-Chen clusters.
    """
    n = instance["n"]
    kind = instance["kind"]
    n_exp_target = instance["n_exp"]
    m_exp_target = instance["m_exp"]
    c_target = instance["c"]
    
    start_time = time.time()
    time_budget = 280.0  # Leave 20s margin
    
    # Multi-stage cooling schedule
    # Stage 0: Very soft, high T - large rearrangements
    # Stage 1: Soft, medium-high T
    # Stage 2: Medium soft, medium T
    # Stage 3: Slightly soft, low T
    # Stage 4: Target potential, very low T - final refinement
    
    # Define schedule: (t_fraction, n_exp_frac, m_exp_frac, c_frac, temperature, n_hops, maxiter)
    schedule = [
        (0.0, 0.35, 0.35, 0.15, 1.0, 8, 60),
        (0.2, 0.5, 0.5, 0.35, 0.6, 10, 80),
        (0.4, 0.65, 0.65, 0.6, 0.35, 12, 100),
        (0.6, 0.8, 0.8, 0.8, 0.15, 12, 120),
        (0.8, 0.95, 0.95, 0.95, 0.05, 10, 150),
        (1.0, 1.0, 1.0, 1.0, 0.02, 8, 200),
    ]
    
    # Use multiple restarts to explore different basins
    n_restarts = 3
    rng = np.random.default_rng(42 + n)
    
    best_overall_e = np.inf
    best_overall_pos = None
    
    for restart in range(n_restarts):
        if time.time() - start_time > time_budget:
            break
        
        # Generate initial structure
        pos = _generate_initial(n, rng)
        
        # First quick relax to get a reasonable starting point
        e_init, pos = _local_relax(pos, n_exp_target, m_exp_target, c_target, maxiter=100)
        
        # Run coarse-to-fine schedule
        for stage_idx, (t_frac, ne_frac, me_frac, c_frac, temp, n_hops, maxiter) in enumerate(schedule):
            if time.time() - start_time > time_budget:
                break
            
            n_exp_s = n_exp_target * ne_frac
            m_exp_s = m_exp_target * me_frac
            c_s = c_target * c_frac
            
            e_stage, pos = _basin_hop_single(pos, n_exp_s, m_exp_s, c_s, temp, n_hops, maxiter, rng)
        
        # Final precise relaxation with target parameters
        if time.time() - start_time < time_budget:
            e_final, pos_final = _local_relax(pos, n_exp_target, m_exp_target, c_target, maxiter=500)
            
            if e_final < best_overall_e:
                best_overall_e = e_final
                best_overall_pos = pos_final.copy()
    
    pos_final = best_overall_pos
    
    # If we didn't get a valid result, try a simple approach
    if pos_final is None or not np.all(np.isfinite(pos_final)):
        rng2 = np.random.default_rng(123 + n)
        pos_final = _generate_initial(n, rng2)
        e_final, pos_final = _local_relax(pos_final, n_exp_target, m_exp_target, c_target, maxiter=300)
    
    # Validate: check no atoms too close
    if n > 1:
        min_dist = np.min(pdist(pos_final))
        if min_dist < 0.35:
            # Try to fix by pushing apart
            for _ in range(100):
                diff = pos_final[:, np.newaxis, :] - pos_final[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff**2, axis=-1))
                mask = (dists < 0.35) & (dists > 1e-10)
                if not np.any(mask):
                    break
                # Push apart
                for i in range(n):
                    for j in range(i + 1, n):
                        if dists[i, j] < 0.35 and dists[i, j] > 1e-10:
                            direction = diff[i, j] / dists[i, j]
                            push = (0.35 - dists[i, j]) * 0.5
                            pos_final[i] += direction * push
                            pos_final[j] -= direction * push
                # Re-relax after pushing
                _, pos_final = _local_relax(pos_final, n_exp_target, m_exp_target, c_target, maxiter=100)
    
    # Final check
    if n > 1:
        min_dist = np.min(pdist(pos_final))
        if min_dist < 0.35:
            # Fallback: return a valid configuration
            rng3 = np.random.default_rng(999 + n)
            pos_final = _generate_initial(n, rng3)
            _, pos_final = _local_relax(pos_final, n_exp_target, m_exp_target, c_target, maxiter=300)
    
    # Ensure finite
    if not np.all(np.isfinite(pos_final)):
        rng4 = np.random.default_rng(777 + n)
        pos_final = _generate_initial(n, rng4)
        _, pos_final = _local_relax(pos_final, n_exp_target, m_exp_target, c_target, maxiter=200)
    
    return pos_final
# EVOLVE-BLOCK-END
