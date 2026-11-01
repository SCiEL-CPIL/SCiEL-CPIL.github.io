"""Lennard-Jones 55-atom global minimum: Mackay icosahedron seed + L-BFGS relaxation."""
import numpy as np
import time
import sys

N = 55

# EVOLVE-BLOCK-START
import numpy as np
import time

def _lj_energy(pos):
    """Compute total LJ energy for a set of positions (vectorized)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, np.inf)
    r6 = r2 ** 3
    # Avoid division by zero or negative if r2 is 0
    e = 4.0 * (1.0 / (r6 * r6) - 1.0 / r6)
    return np.sum(e) / 2.0

def _lj_energy_grad(pos):
    """Compute total LJ energy and gradient (vectorized, numerically stable)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0) # Avoid div by 0 in calculation, handled later
    
    r6 = r2 * r2 * r2
    r12 = r6 * r6
    
    e_pair = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(e_pair, 0.0)
    energy = np.sum(e_pair) / 2.0
    
    # Gradient calculation
    # dE/dr2 = 4 * (-6/r2^4 + 12/r2^7)
    # dE/dpos_i = sum_j dE/dr2_ij * 2*(pos_i - pos_j)
    # Let's use the explicit form:
    # grad_i = sum_j 4 * (-12/r2^7 + 6/r2^4) * (pos_i - pos_j)
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = 4.0 * (-12.0 / (r2 ** 7) + 6.0 / (r2 ** 4))
    np.fill_diagonal(factor, 0.0)
    
    sum_factor = np.sum(factor, axis=1, keepdims=True)
    sum_factor_pos = np.sum(factor[:, :, np.newaxis] * pos[np.newaxis, :, :], axis=1)
    grad = pos * sum_factor - sum_factor_pos
    
    return energy, grad

def _get_mackay_icosahedron_coords():
    """
    Returns unit vectors for the 55 positions in a Mackay icosahedron.
    1 center (0,0,0)
    12 first shell (icosahedron vertices)
    30 second shell (icosahedron edge midpoints)
    12 third shell (icosahedron vertices, same direction as first shell)
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms
    
    # Find edges of the icosahedron
    dists = np.zeros((12, 12))
    for i in range(12):
        for j in range(12):
            dists[i, j] = np.linalg.norm(verts_unit[i] - verts_unit[j])
    
    min_dist = np.min(dists[dists > 1e-6])
    edges = []
    for i in range(12):
        for j in range(i + 1, 12):
            if abs(dists[i, j] - min_dist) < 1e-6:
                edges.append((i, j))
    
    # Edge midpoints (30 atoms)
    edge_mids = []
    for i, j in edges:
        mid = (verts_unit[i] + verts_unit[j]) / 2.0
        norm = np.linalg.norm(mid)
        if norm > 1e-6:
            edge_mids.append(mid / norm)
    edge_mids = np.array(edge_mids)
    
    # Construct full unit vectors
    # Order: Center, Shell1 (12), Shell2 (30), Shell3 (12)
    center = np.array([[0.0, 0.0, 0.0]])
    shell1_dir = verts_unit
    shell2_dir = edge_mids
    shell3_dir = verts_unit # Same directions as shell 1
    
    return center, shell1_dir, shell2_dir, shell3_dir

def _make_mackay_icosahedron(r1=1.05, r2=1.75, r3=1.95):
    """Construct a two-shell Mackay icosahedron for 55 atoms."""
    center, s1, s2, s3 = _get_mackay_icosahedron_coords()
    
    first_shell = s1 * r1
    second_inner = s2 * r2
    second_outer = s3 * r3
    
    all_pos = np.vstack([center, first_shell, second_inner, second_outer])
    return all_pos

def _optimize_full_structure(pos0, deadline, maxiter=5000, gtol=1e-12):
    """
    Optimize the full 3D structure using L-BFGS-B with tight tolerance.
    This allows the structure to deviate from perfect icosahedral symmetry
    while still converging to a high-precision local minimum.
    """
    from scipy.optimize import minimize
    
    best_energy = _lj_energy(pos0)
    best_pos = pos0.copy()
    
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()
    
    x0 = pos0.flatten()
    
    try:
        result = minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 200, 'ftol': 1e-15}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x.reshape(-1, 3)
    except Exception:
        pass
    
    return best_pos, best_energy

def _relax_lbfgs(pos, deadline, maxiter=1000, gtol=1e-10):
    """Relax using L-BFGS-B with deadline guard."""
    from scipy.optimize import minimize
    
    best_energy = _lj_energy(pos)
    best_pos = pos.copy()
    
    def obj_and_grad(x):
        pos_flat = x.reshape(-1, 3)
        e, g = _lj_energy_grad(pos_flat)
        return e, g.flatten()
    
    x0 = pos.flatten()
    
    try:
        result = minimize(
            obj_and_grad, x0, jac=True, method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 200, 'ftol': 1e-15}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x.reshape(-1, 3)
    except Exception:
        pass
    
    return best_pos, best_energy

def construct():
    """Construct 55-atom LJ cluster near global minimum using multi-start stochastic search."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin (evaluator kills at 150s)
    
    # 1. Create base Mackay icosahedron with reasonable initial radii
    base_pos = _make_mackay_icosahedron(1.05, 1.75, 1.95)
    
    best_pos = base_pos.copy()
    best_energy = _lj_energy(base_pos)
    
    # 2. First, do a high-precision full 3D relaxation of the symmetric structure
    # This allows the structure to deviate from perfect icosahedral symmetry
    if time.time() < deadline - 50.0:
        pos_optimized, energy_optimized = _optimize_full_structure(base_pos, deadline, maxiter=5000, gtol=1e-12)
        if energy_optimized < best_energy:
            best_energy = energy_optimized
            best_pos = pos_optimized
        base_pos = pos_optimized  # Use this as the new base for perturbations
    
    # 3. Multi-start stochastic search with perturbed candidates
    rng = np.random.default_rng(42)
    
    # Adaptive perturbation strategy: spend more time on small perturbations
    # which are more likely to stay near the global minimum
    perturbation_scales = [0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.2]
    
    # Reserve time for final polish
    polish_time = 15.0
    search_deadline = deadline - polish_time
    
    # Try perturbed starts with adaptive time allocation
    total_search_time = search_deadline - time.time()
    if total_search_time < 10.0:
        search_deadline = time.time() + max(0.0, total_search_time)
    
    for i, scale in enumerate(perturbation_scales):
        if time.time() >= search_deadline - 5.0:
            break
        
        # Allocate more time to smaller scales (more likely to improve)
        # Small scales: 3-4 perturbations, Medium: 2-3, Large: 1
        if scale < 0.05:
            n_perturbations = 4
        elif scale < 0.15:
            n_perturbations = 3
        elif scale < 0.5:
            n_perturbations = 2
        else:
            n_perturbations = 1
        
        for _ in range(n_perturbations):
            if time.time() >= search_deadline - 3.0:
                break
            
            # Generate a perturbed version
            noise = rng.normal(0, scale, size=base_pos.shape)
            perturbed_pos = base_pos + noise
            
            # Check minimum distance constraint
            diff = perturbed_pos[:, np.newaxis, :] - perturbed_pos[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_dist = np.min(dists)
            
            if min_dist < 0.5:
                # Scale up to satisfy constraint
                scale_factor = 0.5 / min_dist * 1.01
                perturbed_pos = perturbed_pos * scale_factor
            
            # Relax this perturbed candidate
            # Use adaptive iteration count based on perturbation scale
            if scale < 0.1:
                maxiter = 4000
                gtol = 1e-11
            elif scale < 0.5:
                maxiter = 2500
                gtol = 1e-10
            else:
                maxiter = 1500
                gtol = 1e-9
            
            pos_relaxed, energy_relaxed = _relax_lbfgs(perturbed_pos, search_deadline, maxiter=maxiter, gtol=gtol)
            
            if energy_relaxed < best_energy:
                best_energy = energy_relaxed
                best_pos = pos_relaxed
    
    # 4. Final high-precision polish with two-stage approach
    remaining_time = deadline - time.time()
    if remaining_time > 10.0:
        # Stage 1: Moderate precision polish
        pos_polished1, energy_polished1 = _optimize_full_structure(best_pos, deadline, maxiter=5000, gtol=1e-13)
        if energy_polished1 < best_energy:
            best_energy = energy_polished1
            best_pos = pos_polished1
        
        # Stage 2: Very tight polish if time remains
        remaining_time = deadline - time.time()
        if remaining_time > 5.0:
            pos_polished2, energy_polished2 = _optimize_full_structure(best_pos, deadline, maxiter=10000, gtol=1e-15)
            if energy_polished2 < best_energy:
                best_energy = energy_polished2
                best_pos = pos_polished2
    
    # Final Check: Minimum distance constraint
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist_found = np.min(dists)
    
    if min_dist_found < 0.5:
        # If constraint is violated, scale up
        scale = 0.5 / min_dist_found * 1.01
        best_pos = best_pos * scale
    
    return best_pos
# EVOLVE-BLOCK-END
