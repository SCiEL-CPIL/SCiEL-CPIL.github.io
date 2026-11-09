"""Sutton-Chen cluster global minima search using LAMMPS-style sequential atom addition."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Deadline: 330s wall-clock limit, use 285s as our deadline
_DEADLINE = time.time() + 285.0


def _check_deadline():
    return time.time() > _DEADLINE


def _sc_energy(pos, n_exp, m_exp, c):
    """Compute Sutton-Chen energy for a cluster."""
    n = len(pos)
    if n < 2:
        return 0.0
    # Compute all pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    dist = np.maximum(dist, 1e-10)
    
    r_inv_n = np.sum(dist ** (-n_exp), axis=1)
    r_inv_m = np.sum(dist ** (-m_exp), axis=1)
    
    e_pair = 0.5 * np.sum(r_inv_n)
    e_coh = -c * np.sum(np.sqrt(r_inv_m))
    return e_pair + e_coh


def _sc_gradient(pos, n_exp, m_exp, c):
    """Compute gradient of Sutton-Chen energy."""
    n = len(pos)
    if n < 2:
        return np.zeros_like(pos)
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dist, np.inf)
    dist = np.maximum(dist, 1e-10)
    
    # Pairwise forces
    r_inv_n = dist ** (-n_exp)
    r_inv_m = dist ** (-m_exp)
    
    # Gradient contribution from pair term: d/dr (0.5 * r^-n) = -0.5*n*r^(-n-1)
    # For each pair (i,j), force on i is -0.5*n*r^(-n-1) * (pos_i - pos_j)/r
    # = -0.5*n*r^(-n-2) * (pos_i - pos_j)
    pair_force = 0.5 * n_exp * dist ** (-n_exp - 2) * diff
    
    # Cohesive term gradient
    # E_coh_i = -c * sqrt(sum_j r_ij^-m)
    # dE/dpos_i = -c * (1/(2*sqrt(sum_j r_ij^-m))) * sum_j (-m*r_ij^(-m-2))*(pos_i - pos_j)
    r_inv_m_sum = np.sum(r_inv_m, axis=1)
    r_inv_m_sum = np.maximum(r_inv_m_sum, 1e-20)
    
    # For each i, the gradient of sqrt(sum_j r_ij^-m) with respect to pos_i
    # = (1/(2*sqrt(sum_j r_ij^-m))) * sum_j (-m * r_ij^(-m-2)) * (pos_i - pos_j)
    # = -(m/(2*sqrt(sum_j r_ij^-m))) * sum_j r_ij^(-m-2) * (pos_i - pos_j)
    
    # Compute sum_j r_ij^(-m-2) * (pos_i - pos_j) for each i
    weight = r_inv_m ** (1.0 + 2.0/m_exp)  # r^(-m-2)
    coh_term = np.sum(weight[:, :, np.newaxis] * diff, axis=1)
    
    coh_grad = -(m_exp / (2.0 * np.sqrt(r_inv_m_sum))) * coh_grad_factor(r_inv_m_sum) * coh_term
    # Wait, let me redo this more carefully
    
    # E_coh = -c * sum_i sqrt(sum_{j!=i} r_ij^{-m})
    # dE_coh/dpos_i = -c * (1/(2*sqrt(S_i))) * dS_i/dpos_i
    # where S_i = sum_{j!=i} r_ij^{-m}
    # dS_i/dpos_i = sum_{j!=i} (-m) * r_ij^{-m-2} * (pos_i - pos_j)
    # So dE_coh/dpos_i = -c * (1/(2*sqrt(S_i))) * (-m) * sum_{j!=i} r_ij^{-m-2} * (pos_i - pos_j)
    # = c * m * (1/(2*sqrt(S_i))) * sum_{j!=i} r_ij^{-m-2} * (pos_i - pos_j)
    
    S_i = r_inv_m_sum
    weight_m = dist ** (-m_exp - 2)
    coh_sum = np.sum(weight_m[:, :, np.newaxis] * diff, axis=1)
    coh_grad = c * m_exp * coh_sum / (2.0 * np.sqrt(S_i)[:, np.newaxis])
    
    total_grad = -pair_force.sum(axis=1) + coh_grad
    # Wait, pair_force[i,j] is the force contribution to atom i from atom j
    # The gradient is the negative of the force
    # Let me reconsider:
    # E_pair = 0.5 * sum_{i!=j} r_ij^{-n}
    # dE_pair/dpos_i = sum_{j!=i} (-n/2) * r_ij^{-n-2} * (pos_i - pos_j)
    # = -0.5*n * sum_{j!=i} r_ij^{-n-2} * (pos_i - pos_j)
    # = -0.5*n * (pair_force_sum_i)
    # where pair_force_sum_i = sum_j r_ij^{-n-2} * (pos_i - pos_j)
    
    pair_grad = -0.5 * n_exp * np.sum(dist ** (-n_exp - 2)[:, :, np.newaxis] * diff, axis=1)
    total_grad = pair_grad + coh_grad
    
    return total_grad


def _generate_fcc_cluster(n):
    """Generate an FCC-like cluster with n atoms."""
    # FCC lattice constant ~1.0 for SC potential
    a = 1.0
    positions = []
    
    # Generate FCC points in a cube
    # FCC basis: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    
    # Determine cube size
    side = int(np.ceil((n / 4) ** (1/3)))
    
    for i in range(side):
        for j in range(side):
            for k in range(side):
                for b in basis:
                    p = a * (np.array([i, j, k]) + b)
                    positions.append(p)
                    if len(positions) >= n:
                        break
                if len(positions) >= n:
                    break
            if len(positions) >= n:
                break
        if len(positions) >= n:
            break
    
    if len(positions) < n:
        # Add more layers
        for i in range(side + 1):
            for j in range(side + 1):
                for k in range(side + 1):
                    for b in basis:
                        p = a * (np.array([i, j, k]) + b)
                        positions.append(p)
                        if len(positions) >= n:
                            break
                    if len(positions) >= n:
                        break
                if len(positions) >= n:
                    break
            if len(positions) >= n:
                break
    
    return np.array(positions[:n])


def _generate_hcp_cluster(n):
    """Generate an HCP-like cluster with n atoms."""
    a = 1.0
    c = 1.633 * a  # ideal c/a ratio
    positions = []
    
    # HCP layers: ABAB...
    # Layer A: hexagonal close-packed
    # Layer B: offset by (1/3, 1/3, 0) in fractional coords
    
    layer_spacing = c / 2.0
    num_layers = int(np.ceil(n / 12))  # ~12 atoms per layer
    
    for layer in range(num_layers):
        z = layer * layer_spacing
        is_a = (layer % 2 == 0)
        
        # Hexagonal lattice in the layer
        # Basis for hexagonal: (0,0), (1,0), (0.5, sqrt(3)/2), etc.
        radius = 2
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                x = a * (i + 0.5 * (j if j % 2 != 0 else 0))
                y = a * (np.sqrt(3) / 2 * j)
                if is_a:
                    p = np.array([x, y, z])
                else:
                    p = np.array([x + a/2, y + a/(2*np.sqrt(3)), z])
                positions.append(p)
                if len(positions) >= n:
                    break
            if len(positions) >= n:
                break
        if len(positions) >= n:
            break
    
    return np.array(positions[:n])


def _generate_icosahedron(n):
    """Generate an icosahedral cluster."""
    if n <= 1:
        return np.zeros((n, 3))
    
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    positions = [np.zeros(3)]  # center atom
    
    # First shell: 12 vertices of icosahedron
    shell1_radius = 1.0
    for v in verts:
        positions.append(shell1_radius * v)
        if len(positions) >= n:
            break
    
    if len(positions) < n:
        # Second shell: 30 atoms (edges of icosahedron)
        # Generate midpoints of edges
        edges = []
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(verts[i] - verts[j])
                if d < 1.1:  # close to edge length
                    edges.append((i, j))
        
        shell2_radius = 1.0 * np.sqrt(1 + 1/phi**2)  # approximate
        for i, j in edges:
            mid = (verts[i] + verts[j]) / 2
            mid /= np.linalg.norm(mid)
            positions.append(shell2_radius * mid)
            if len(positions) >= n:
                break
    
    if len(positions) < n:
        # Third shell: 20 atoms (face centers)
        faces = []
        # Find triangular faces
        for i in range(12):
            for j in range(i+1, 12):
                for k in range(j+1, 12):
                    d1 = np.linalg.norm(verts[i] - verts[j])
                    d2 = np.linalg.norm(verts[j] - verts[k])
                    d3 = np.linalg.norm(verts[i] - verts[k])
                    if d1 < 1.1 and d2 < 1.1 and d3 < 1.1:
                        faces.append((i, j, k))
        
        shell3_radius = 1.0 * (1 + 1/phi**2) ** 0.5  # approximate
        for i, j, k in faces:
            center = (verts[i] + verts[j] + verts[k]) / 3
            center /= np.linalg.norm(center)
            positions.append(shell3_radius * center)
            if len(positions) >= n:
                break
    
    return np.array(positions[:n])


def _add_atom_lammps(current_pos, n_exp, m_exp, c, rng):
    """Add one atom to the cluster using LAMMPS-style algorithm."""
    n = len(current_pos)
    
    # Try multiple random starting positions for the new atom
    best_energy = np.inf
    best_pos = None
    
    # Estimate cluster radius
    if n > 0:
        centroid = np.mean(current_pos, axis=0)
        radius = np.max(np.linalg.norm(current_pos - centroid, axis=1)) + 0.5
    else:
        centroid = np.zeros(3)
        radius = 1.0
    
    # Try several candidate positions
    n_candidates = 20
    for _ in range(n_candidates):
        # Random position near the surface
        theta = rng.uniform(0, 2 * np.pi)
        phi = rng.uniform(0, np.pi)
        r = radius * rng.uniform(0.8, 1.2)
        
        new_atom = np.array([
            r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi)
        ])
        
        trial_pos = np.vstack([current_pos, new_atom])
        
        # Check minimum distance
        if n > 0:
            dists = np.linalg.norm(trial_pos[-1] - current_pos, axis=1)
            if np.min(dists) < 0.35:
                continue
        
        # Optimize the new atom's position with the core fixed
        def objective(x):
            pos = np.vstack([current_pos, x])
            return _sc_energy(pos, n_exp, m_exp, c)
        
        def grad(x):
            pos = np.vstack([current_pos, x])
            g = _sc_gradient(pos, n_exp, m_exp, c)
            return g[-1]
        
        result = minimize(objective, new_atom, jac=grad, method='L-BFGS-B',
                          options={'maxiter': 200, 'ftol': 1e-8})
        
        if result.success or result.fun < best_energy:
            # Verify minimum distance after optimization
            final_pos = np.vstack([current_pos, result.x])
            if n > 0:
                dists = np.linalg.norm(result.x - current_pos, axis=1)
                if np.min(dists) >= 0.35:
                    best_energy = result.fun
                    best_pos = result.x
    
    if best_pos is None:
        # Fallback: place at a reasonable distance
        theta = rng.uniform(0, 2 * np.pi)
        phi = rng.uniform(0, np.pi)
        r = radius + 0.5
        best_pos = np.array([
            r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi)
        ])
    
    return np.vstack([current_pos, best_pos])


def _lammps_build(n, n_exp, m_exp, c, seed=42):
    """Build cluster using LAMMPS-style sequential atom addition."""
    rng = np.random.default_rng(seed)
    
    if n == 0:
        return np.zeros((0, 3))
    if n == 1:
        return np.zeros((1, 3))
    
    # Start with 2 atoms
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    
    # Relax the initial pair
    energy, pos = relax(pos, 'sutton_chen', {'n_exp': n_exp, 'm_exp': m_exp, 'c': c})
    
    # Add atoms one by one
    for i in range(2, n):
        if _check_deadline():
            break
        
        pos = _add_atom_lammps(pos, n_exp, m_exp, c, rng)
        
        # Periodically relax the whole cluster
        if i % 5 == 0 or i == n - 1:
            energy, pos = relax(pos, 'sutton_chen', {'n_exp': n_exp, 'm_exp': m_exp, 'c': c})
    
    return pos


def _basin_hopping(pos, n_exp, m_exp, c, kind, n_restarts=3, max_steps=50):
    """Perform basin hopping to improve the structure."""
    best_energy, best_pos = _sc_energy(pos, n_exp, m_exp, c), pos.copy()
    
    for restart in range(n_restarts):
        if _check_deadline():
            break
        
        current_energy, current_pos = _sc_energy(pos, n_exp, m_exp, c), pos.copy()
        
        for step in range(max_steps):
            if _check_deadline():
                break
            
            # Perturb the structure
            perturbation = 0.1 * (restart + 1)
            perturbed = current_pos + np.random.default_rng(step + restart * 100).normal(0, perturbation, current_pos.shape)
            
            # Relax the perturbed structure
            relaxed_energy, relaxed_pos = relax(perturbed, 'sutton_chen', kind)
            
            # Metropolis acceptance (or just take if better)
            if relaxed_energy < current_energy:
                current_energy, current_pos = relaxed_energy, relaxed_pos
            
            # Update best
            if current_energy < best_energy:
                best_energy, best_pos = current_energy, current_pos.copy()
    
    return best_pos, best_energy


def construct(instance):
    """Construct a low-energy Sutton-Chen cluster using LAMMPS-style building and basin hopping."""
    size = instance["n"]
    kind = instance["kind"]
    n_exp = instance["n_exp"]
    m_exp = instance["m_exp"]
    c = instance["c"]
    
    # Create the kind dictionary for relax
    kind_dict = {'n_exp': n_exp, 'm_exp': m_exp, 'c': c}
    
    best_energy = np.inf
    best_pos = None
    
    # Strategy 1: LAMMPS-style sequential building
    try:
        pos_lammps = _lammps_build(size, n_exp, m_exp, c, seed=42)
        energy, pos_lammps = relax(pos_lammps, 'sutton_chen', kind_dict)
        if energy < best_energy:
            best_energy, best_pos = energy, pos_lammps.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_pos if best_pos is not None else _generate_fcc_cluster(size)
    
    # Strategy 2: Basin hopping from LAMMPS result
    try:
        pos_bh, energy_bh = _basin_hopping(pos_lammps, n_exp, m_exp, c, kind_dict, n_restarts=2, max_steps=30)
        if energy_bh < best_energy:
            best_energy, best_pos = energy_bh, pos_bh.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_pos
    
    # Strategy 3: Basin hopping from FCC seed
    try:
        pos_fcc = _generate_fcc_cluster(size)
        energy_fcc, pos_fcc = relax(pos_fcc, 'sutton_chen', kind_dict)
        if energy_fcc < best_energy:
            best_energy, best_pos = energy_fcc, pos_fcc.copy()
        
        pos_bh2, energy_bh2 = _basin_hopping(pos_fcc, n_exp, m_exp, c, kind_dict, n_restarts=2, max_steps=30)
        if energy_bh2 < best_energy:
            best_energy, best_pos = energy_bh2, pos_bh2.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_pos
    
    # Strategy 4: Basin hopping from HCP seed
    try:
        pos_hcp = _generate_hcp_cluster(size)
        energy_hcp, pos_hcp = relax(pos_hcp, 'sutton_chen', kind_dict)
        if energy_hcp < best_energy:
            best_energy, best_pos = energy_hcp, pos_hcp.copy()
        
        pos_bh3, energy_bh3 = _basin_hopping(pos_hcp, n_exp, m_exp, c, kind_dict, n_restarts=2, max_steps=30)
        if energy_bh3 < best_energy:
            best_energy, best_pos = energy_bh3, pos_bh3.copy()
    except Exception:
        pass
    
    if _check_deadline():
        return best_pos
    
    # Strategy 5: Basin hopping from icosahedral seed
    try:
        pos_ico = _generate_icosahedron(size)
        energy_ico, pos_ico = relax(pos_ico, 'sutton_chen', kind_dict)
        if energy_ico < best_energy:
            best_energy, best_pos = energy_ico, pos_ico.copy()
        
        pos_bh4, energy_bh4 = _basin_hopping(pos_ico, n_exp, m_exp, c, kind_dict, n_restarts=2, max_steps=30)
        if energy_bh4 < best_energy:
            best_energy, best_pos = energy_bh4, pos_bh4.copy()
    except Exception:
        pass
    
    # Final relaxation of the best structure
    if best_pos is not None:
        final_energy, final_pos = relax(best_pos, 'sutton_chen', kind_dict)
        return final_pos
    
    # Fallback
    return _generate_fcc_cluster(size)
# EVOLVE-BLOCK-END
