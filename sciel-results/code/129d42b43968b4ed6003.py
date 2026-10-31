"""Rewritten program using robust multi-start FCC lattice construction with local relaxation."""
N = 38

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys
from scipy.optimize import minimize

def _lj_energy(pos):
    """Compute total Lennard-Jones energy."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=-1)
    mask = ~np.eye(len(pos), dtype=bool)
    r2 = r2[mask]
    r6 = r2 ** 3
    r12 = r6 ** 2
    energy = 4.0 * np.sum(1.0 / r12 - 1.0 / r6)
    return energy

def _lj_grad(pos):
    """Compute gradient of total Lennard-Jones energy."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=-1)
    r2 = np.where(r2 < 1e-12, 1e-12, r2)
    r6 = r2 ** 3
    r12 = r6 ** 2
    coeff = 4.0 * (-12.0 / (r2 ** 7) + 6.0 / (r2 ** 4))
    np.fill_diagonal(coeff, 0.0)
    grad = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    return grad

def _minimize_lbfgs(pos, max_iter=2000, tol=1e-10):
    """L-BFGS minimization with tight tolerances and gradient-based convergence check."""
    try:
        res = minimize(
            _lj_energy,
            pos,
            jac=_lj_grad,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': tol, 'gtol': tol, 'maxls': 50}
        )
        result = res.x
        # Check gradient norm - if it's very small, we're likely at a minimum
        grad = _lj_grad(result)
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-8:
            # Try another pass with tighter tolerance
            try:
                res2 = minimize(
                    _lj_energy,
                    result,
                    jac=_lj_grad,
                    method='L-BFGS-B',
                    options={'maxiter': max_iter // 2, 'ftol': tol * 0.1, 'gtol': tol * 0.1, 'maxls': 50}
                )
                if _lj_energy(res2.x) < _lj_energy(result):
                    result = res2.x
            except Exception:
                pass
        return result
    except Exception:
        return pos

def _optimize_scale(pos, max_iter=1000):
    """Optimize the 1D scale parameter to find optimal interatomic distance."""
    def scale_energy(log_s):
        s = np.exp(log_s[0])
        scaled = pos * s
        return _lj_energy(scaled)
    
    def scale_grad(log_s):
        s = np.exp(log_s[0])
        scaled = pos * s
        # dE/ds = sum of dE/dpos_i * pos_i (chain rule)
        grad_pos = _lj_grad(scaled)
        dEds = np.sum(grad_pos * scaled)
        dE_dlog_s = dEds * s
        return np.array([dE_dlog_s])
    
    # Start from scale=1 (log_s=0)
    try:
        res = minimize(
            scale_energy,
            np.array([0.0]),
            jac=scale_grad,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-12}
        )
        optimal_scale = np.exp(res.x[0])
        return pos * optimal_scale
    except Exception:
        return pos

def _generate_fcc_cluster(n_atoms, r_nn=2**(1/6)):
    """Generate an FCC cluster of n_atoms closest to the origin."""
    scale = r_nn / np.sqrt(2.0)
    max_coord = 10  # Ensure we have enough points
    
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                if (i + j + k) % 2 == 0:
                    points.append((i, j, k))
    
    pts = np.array(points, dtype=float)
    pts *= scale
    
    # Sort by distance to origin
    dists = np.sum(pts**2, axis=1)
    sorted_indices = np.argsort(dists)
    
    # Take the first n_atoms points
    cluster = pts[sorted_indices[:n_atoms]].copy()
    
    # Center the cluster at the origin
    center = np.mean(cluster, axis=0)
    cluster -= center
    
    return cluster

def _basin_hop(pos, rng, n_hops=10, max_iter=1000):
    """Basin hopping to escape local minima."""
    current_pos = pos.copy()
    current_energy = _lj_energy(current_pos)
    
    for hop in range(n_hops):
        if time.time() > deadline_global - 20.0:
            break
        
        # Perturb with random displacement
        scale = 0.3 + 0.2 * rng.random()
        perturbed = current_pos + rng.normal(0, scale, size=current_pos.shape)
        
        # Relax
        relaxed = _minimize_lbfgs(perturbed, max_iter=max_iter, tol=1e-10)
        new_energy = _lj_energy(relaxed)
        
        # Metropolis accept/reject
        if new_energy < current_energy or rng.random() < np.exp(-(new_energy - current_energy) / 0.1):
            current_pos = relaxed
            current_energy = new_energy
    
    return current_pos, current_energy

def construct():
    global deadline_global
    start_time = time.time()
    deadline_global = start_time + 115.0  # Safety margin (150s limit, 35s buffer)
    
    # 1. Generate the ideal FCC truncated octahedron
    base_cluster = _generate_fcc_cluster(N)
    
    # Initial relaxation of the perfect lattice
    best_pos = _minimize_lbfgs(base_cluster, max_iter=1000, tol=1e-10)
    best_energy = _lj_energy(best_pos)
    
    # Check if we're already at the theoretical minimum
    # The known global minimum is approximately -173.928427
    if best_energy < -173.92:
        # Already at or near global minimum, just polish
        if time.time() < deadline_global - 10.0:
            final_pos = _minimize_lbfgs(best_pos, max_iter=5000, tol=1e-12)
            final_scaled = _optimize_scale(final_pos, max_iter=1000)
            final_energy = _lj_energy(final_scaled)
            if final_energy < best_energy:
                best_pos = final_scaled
                best_energy = final_energy
        return best_pos
    
    # 2. Basin hopping from the FCC start to escape any local minima
    rng = np.random.RandomState(42)
    bh_pos, bh_energy = _basin_hop(best_pos, rng, n_hops=15, max_iter=800)
    
    # Optimize scale for basin-hopped result
    bh_scaled = _optimize_scale(bh_pos, max_iter=500)
    bh_energy_final = _lj_energy(bh_scaled)
    
    if bh_energy_final < best_energy:
        best_energy = bh_energy_final
        best_pos = bh_scaled
    
    # 3. Multi-start with small perturbations for fine-tuning
    if time.time() < deadline_global - 25.0:
        scales = [0.01, 0.05, 0.1, 0.2]
        for scale in scales:
            for _ in range(3):
                if time.time() > deadline_global - 15.0:
                    break
                
                perturbed = best_pos + rng.normal(0, scale, size=best_pos.shape)
                relaxed = _minimize_lbfgs(perturbed, max_iter=1500, tol=1e-10)
                scaled_relaxed = _optimize_scale(relaxed, max_iter=500)
                energy = _lj_energy(scaled_relaxed)
                
                if energy < best_energy:
                    best_energy = energy
                    best_pos = scaled_relaxed
    
    # 4. Final high-precision polish with gradient validation
    if time.time() < deadline_global - 10.0:
        final_pos = _minimize_lbfgs(best_pos, max_iter=5000, tol=1e-12)
        final_scaled = _optimize_scale(final_pos, max_iter=1000)
        final_energy = _lj_energy(final_scaled)
        
        # Validate: check gradient norm to ensure convergence
        final_grad = _lj_grad(final_scaled)
        final_grad_norm = np.linalg.norm(final_grad)
        
        if final_energy < best_energy:
            best_pos = final_scaled
            best_energy = final_energy

    return best_pos
# EVOLVE-BLOCK-END
