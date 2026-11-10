"""Dzugutov cluster global minima search using basin hopping with polytetrahedral seeding."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient
import time

# Dzugutov potential parameters
A_POT = 5.82
B_POT = 1.28
a_POT = 1.87
b_POT = 1.94
c_POT = 1.10
d_POT = 0.27
m_POT = 16


def dzugutov_energy(positions):
    """Compute Dzugutov potential energy for a cluster."""
    n = len(positions)
    if n < 2:
        return 0.0
    
    # Compute all pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(r, np.inf)
    
    # Only consider pairs i < j
    triu_idx = np.triu_indices(n, k=1)
    r_ij = r[triu_idx]
    
    energy = 0.0
    for ri in r_ij:
        if ri < a_POT:
            energy += A_POT * (ri ** (-m_POT) - B_POT) * np.exp(c_POT / (ri - a_POT))
        elif ri < b_POT:
            energy += B_POT * np.exp(d_POT / (ri - b_POT))
    return energy


def dzugutov_energy_gradient(positions):
    """Compute Dzugutov potential energy and gradient."""
    n = len(positions)
    if n < 2:
        return 0.0, np.zeros_like(positions)
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(r, np.inf)
    
    energy = 0.0
    grad = np.zeros_like(positions)
    
    for i in range(n):
        for j in range(i + 1, n):
            rij = r[i, j]
            if rij < 1e-12:
                continue
            
            if rij < a_POT:
                rinv_m = rij ** (-m_POT)
                exp_term = np.exp(c_POT / (rij - a_POT))
                v = A_POT * (rinv_m - B_POT) * exp_term
                dv_dr = A_POT * (-m_POT * rij ** (-m_POT - 1) * exp_term +
                                 (rinv_m - B_POT) * exp_term * (-c_POT / (rij - a_POT) ** 2))
            elif rij < b_POT:
                v = B_POT * np.exp(d_POT / (rij - b_POT))
                dv_dr = B_POT * np.exp(d_POT / (rij - b_POT)) * (-d_POT / (rij - b_POT) ** 2)
            else:
                continue
            
            energy += v
            rij_vec = diff[i, j]
            grad[i] -= dv_dr * rij_vec / rij
            grad[j] += dv_dr * rij_vec / rij
    
    return energy, grad


def generate_icosahedron_seeds(n, rng):
    """Generate diverse icosahedral-based starting structures."""
    seeds = []
    
    # Icosahedron vertices (12 vertices)
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    ico_verts /= np.linalg.norm(ico_verts[0])  # Normalize to unit sphere
    
    # Try different core sizes and shell configurations
    core_sizes = [1, 12, 13, 23, 55]
    core_sizes = [cs for cs in core_sizes if cs <= n]
    if not core_sizes:
        core_sizes = [1]
    
    for core_size in core_sizes:
        for attempt in range(3):  # Multiple random perturbations per core size
            positions = []
            
            # Place core atoms
            if core_size == 1:
                positions.append(np.zeros(3))
            elif core_size == 12:
                # Icosahedron of 12 atoms
                scale = 1.0
                positions.extend([v * scale for v in ico_verts])
            elif core_size == 13:
                # Icosahedron + center
                scale = 1.0
                positions.append(np.zeros(3))
                positions.extend([v * scale for v in ico_verts])
            elif core_size == 23:
                # Two concentric icosahedra
                scale1 = 0.7
                scale2 = 1.2
                positions.extend([v * scale1 for v in ico_verts])
                positions.extend([v * scale2 for v in ico_verts])
            else:
                # For larger cores, use multiple shells
                remaining = core_size
                shells = []
                radii = [0.5, 1.0, 1.5, 2.0]
                for radius in radii:
                    if remaining <= 0:
                        break
                    shell_size = min(remaining, 12)
                    if shell_size == 12:
                        shell = [v * radius for v in ico_verts]
                    else:
                        # Place atoms on sphere at this radius
                        shell = []
                        for _ in range(shell_size):
                            # Random point on sphere
                            theta = rng.uniform(0, 2 * np.pi)
                            z = rng.uniform(-1, 1)
                            r_xy = np.sqrt(1 - z ** 2)
                            x = r_xy * np.cos(theta)
                            y = r_xy * np.sin(theta)
                            shell.append(np.array([x, y, z]) * radius)
                    positions.extend(shell[:remaining])
                    remaining -= shell_size
            
            # Add remaining atoms in outer shells
            remaining = n - len(positions)
            if remaining > 0:
                # Place additional atoms in shells around the core
                shell_radius = 1.8 + attempt * 0.1
                for _ in range(remaining):
                    theta = rng.uniform(0, 2 * np.pi)
                    z = rng.uniform(-1, 1)
                    r_xy = np.sqrt(1 - z ** 2)
                    x = r_xy * np.cos(theta)
                    y = r_xy * np.sin(theta)
                    pos = np.array([x, y, z]) * shell_radius
                    # Add small random perturbation
                    pos += rng.normal(0, 0.05, 3)
                    positions.append(pos)
            
            if len(positions) == n:
                positions = np.array(positions)
                # Center of mass to origin
                positions -= np.mean(positions, axis=0)
                seeds.append(positions)
    
    # Also generate some random compact structures
    for _ in range(3):
        radius = 1.1 * n ** (1.0 / 3.0) * 0.8
        positions = []
        while len(positions) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.6 for q in positions):
                positions.append(p)
        if len(positions) == n:
            positions = np.array(positions)
            positions -= np.mean(positions, axis=0)
            seeds.append(positions)
    
    return seeds


def basin_hopping(positions, max_iter=50, step_size=0.5, rng=None):
    """Perform basin hopping optimization."""
    if rng is None:
        rng = np.random.default_rng()
    
    current_energy, _ = dzugutov_energy_gradient(positions)
    best_positions = positions.copy()
    best_energy = current_energy
    
    for _ in range(max_iter):
        # Random perturbation
        perturbed = positions + rng.normal(0, step_size, positions.shape)
        
        # Local relaxation
        try:
            res = minimize(
                lambda x: dzugutov_energy_gradient(x)[0],
                perturbed,
                jac=lambda x: dzugutov_energy_gradient(x)[1],
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            new_positions = res.x
            new_energy = dzugutov_energy(new_positions)
        except Exception:
            continue
        
        # Check minimum distance constraint
        if np.min(np.linalg.norm(new_positions[:, np.newaxis] - new_positions[np.newaxis], axis=-1)[np.triu_indices(n := len(new_positions), k=1)]) < 0.5:
            continue
        
        # Metropolis accept/reject
        if new_energy < current_energy:
            current_energy = new_energy
            positions = new_positions
            if new_energy < best_energy:
                best_energy = new_energy
                best_positions = new_positions.copy()
        else:
            delta_e = new_energy - current_energy
            if delta_e < 0 or rng.random() < np.exp(-delta_e / 0.1):
                current_energy = new_energy
                positions = new_positions
    
    return best_positions, best_energy


def construct(instance):
    """Find low-energy Dzugutov cluster structures using basin hopping with diverse seeds."""
    n = instance["n"]
    start_time = time.time()
    deadline = start_time + 320  # Safety margin before 330s limit
    
    rng = np.random.default_rng(n)
    
    # Generate diverse starting structures
    seeds = generate_icosahedron_seeds(n, rng)
    
    best_positions = None
    best_energy = np.inf
    
    # Try each seed with basin hopping
    for seed in seeds:
        if time.time() > deadline:
            break
        
        try:
            positions, energy = basin_hopping(seed, max_iter=30, step_size=0.4, rng=rng)
            if energy < best_energy:
                best_energy = energy
                best_positions = positions
        except Exception:
            continue
    
    # If we have time, do additional random restarts
    if best_positions is None or time.time() < deadline - 60:
        for _ in range(5):
            if time.time() > deadline:
                break
            radius = 1.1 * n ** (1.0 / 3.0) * 0.9
            positions = []
            while len(positions) < n:
                p = rng.uniform(-radius, radius, 3)
                if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.6 for q in positions):
                    positions.append(p)
            if len(positions) == n:
                positions = np.array(positions)
                positions -= np.mean(positions, axis=0)
                try:
                    pos, energy = basin_hopping(positions, max_iter=20, step_size=0.3, rng=rng)
                    if energy < best_energy:
                        best_energy = energy
                        best_positions = pos
                except Exception:
                    continue
    
    if best_positions is None:
        # Fallback: return a simple compact structure
        radius = 1.1 * n ** (1.0 / 3.0) * 0.8
        positions = []
        while len(positions) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.6 for q in positions):
                positions.append(p)
        best_positions = np.array(positions)
    
    return best_positions
# EVOLVE-BLOCK-END
