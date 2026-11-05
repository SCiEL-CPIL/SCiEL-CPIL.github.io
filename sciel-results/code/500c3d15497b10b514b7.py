import numpy as np
import time
from autoresearch_suite.chem_clusters import relax
from scipy.optimize import minimize, minimize_scalar

# Dzugutov potential parameters
A = 5.82
B = 1.28
a_pot = 1.87
b_pot = 1.94
c_pot = 1.10
d_pot = 0.27
m_pot = 16.0

# Precompute the minimum of the Dzugutov potential
def _find_r_min():
    """Find the minimum of the Dzugutov potential in (0, a)."""
    def V(r):
        if r < a_pot:
            return A * (r ** (-m_pot) - B) * np.exp(c_pot / (r - a_pot))
        elif r < b_pot:
            return B * np.exp(d_pot / (r - b_pot))
        else:
            return 0.0
    # The minimum should be in the first branch (r < a)
    res = minimize_scalar(V, bounds=(0.5, a_pot - 0.01), method='bounded', options={'xatol': 1e-8})
    return res.x

R_MIN = _find_r_min()

def _energy_and_gradient(positions):
    """
    Vectorized calculation of energy and gradient for Dzugutov potential.
    Returns (energy, gradient).
    """
    n = len(positions)
    if n < 2:
        return 0.0, np.zeros_like(positions)
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    r_safe = np.where(mask, r, 1.0)
    
    m1 = r_safe < a_pot
    m2 = (r_safe >= a_pot) & (r_safe < b_pot)
    
    r1 = np.where(m1, r_safe, 1.0)
    r2 = np.where(m2, r_safe, 1.0)
    
    e1 = A * (r1 ** (-m_pot) - B) * np.exp(c_pot / (r1 - a_pot))
    e2 = B * np.exp(d_pot / (r2 - b_pot))
    
    pair_energy = np.where(m1, e1, 0.0) + np.where(m2, e2, 0.0)
    energy = 0.5 * np.sum(pair_energy)
    
    exp1 = np.exp(c_pot / (r1 - a_pot))
    de1_dr = A * (
        -m_pot * r1 ** (-m_pot - 1) * exp1 + 
        (r1 ** (-m_pot) - B) * exp1 * (-c_pot / (r1 - a_pot) ** 2)
    )
    
    exp2 = np.exp(d_pot / (r2 - b_pot))
    de2_dr = B * exp2 * (-d_pot / (r2 - b_pot) ** 2)
    
    de_dr = np.where(m1, de1_dr, 0.0) + np.where(m2, de2_dr, 0.0)
    
    factor = np.where(r_safe > 1e-12, de_dr / r_safe, 0.0)
    factor = np.where(mask, factor, 0.0)
    
    grad = np.einsum('ij,ijk->ik', factor, diff)
    
    return energy, grad

def _barrier_energy_and_gradient(positions, bond_pairs, barrier_strength=100.0):
    """
    Energy with barrier term that penalizes bond-length deviations from R_MIN.
    This constrains the optimization to maintain polytetrahedral connectivity.
    """
    n = len(positions)
    if n < 2:
        return 0.0, np.zeros_like(positions)
    
    energy, grad = _energy_and_gradient(positions)
    
    # Add barrier term for bond pairs
    if len(bond_pairs) > 0:
        bond_pairs = np.array(bond_pairs)
        if len(bond_pairs) > 0:
            i_idx = bond_pairs[:, 0]
            j_idx = bond_pairs[:, 1]
            
            rij = positions[i_idx] - positions[j_idx]
            r_bonds = np.linalg.norm(rij, axis=1)
            
            # Barrier: penalize deviations from R_MIN
            dr = r_bonds - R_MIN
            barrier_e = 0.5 * barrier_strength * np.sum(dr ** 2)
            energy += barrier_e
            
            # Gradient of barrier
            # d(barrier)/d(positions[i]) = barrier_strength * dr * (rij / r)
            # d(barrier)/d(positions[j]) = -barrier_strength * dr * (rij / r)
            factor_b = barrier_strength * dr / np.maximum(r_bonds, 1e-12)
            
            for k in range(len(bond_pairs)):
                i, j = i_idx[k], j_idx[k]
                f = factor_b[k]
                grad[i] += f * rij[k]
                grad[j] -= f * rij[k]
    
    return energy, grad

def _icosahedron_vertices(radius=1.0):
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1],
        [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts, axis=1)
    verts = verts / norms[:, np.newaxis] * radius
    return verts

def _get_mackay_55_positions():
    """
    Get the 55-atom Mackay icosahedron positions.
    Shell 0: 1 atom at center
    Shell 1: 12 atoms at icosahedron vertices
    Shell 2: 42 atoms (face centers and edge midpoints)
    """
    ico = _icosahedron_vertices(1.0)
    
    # Center
    center = np.zeros(3)
    
    # Shell 1: 12 icosahedron vertices
    shell1 = ico.copy()
    
    # Shell 2: 42 positions
    # For a proper Mackay icosahedron, shell 2 consists of:
    # - 30 edge midpoints (between adjacent shell-1 atoms)
    # - 12 face centers (of the 20 triangular faces)
    # Total: 30 + 12 = 42
    
    # Find adjacent pairs in the icosahedron
    # An icosahedron has 30 edges
    # We'll compute this from the geometry
    
    # For simplicity, generate shell 2 positions on a sphere
    # and let optimization refine them
    
    # Actually, let's use a known construction:
    # The 42 shell-2 atoms are at positions corresponding to:
    # - 12 face centers of the icosahedron
    # - 30 edge midpoints
    
    # Compute face centers
    # An icosahedron has 20 faces
    # We need to identify the 20 triangular faces
    
    # For a regular icosahedron with vertices v_i, the faces are:
    # This is complex to compute. Let's use a simpler approach:
    # Place 42 atoms on a sphere of radius R2 and let optimization work
    
    # For now, generate 42 points on a sphere using Fibonacci sphere
    n_shell2 = 42
    R2 = 1.7  # Approximate radius for shell 2
    
    # Fibonacci sphere for even distribution
    shell2 = []
    golden_angle = np.pi * (3 - np.sqrt(5))
    for i in range(n_shell2):
        y = 1 - (i / (n_shell2 - 1)) * 2
        r = np.sqrt(1 - y ** 2)
        theta = golden_angle * i
        x = r * np.cos(theta)
        z = r * np.sin(theta)
        shell2.append(np.array([x, y, z]) * R2)
    shell2 = np.array(shell2)
    
    all_points = np.vstack([center[None, :], shell1, shell2])
    return all_points

def _get_mackay_positions(n, rng):
    """
    Generate Mackay-like positions for n atoms.
    For n <= 55, use a subset of the 55-atom Mackay icosahedron.
    For n > 55, add atoms to the outer shell.
    """
    if n <= 55:
        mackay55 = _get_mackay_55_positions()
        # For n < 55, remove atoms from the outer shell
        # Keep center (1), shell 1 (12), and as many shell 2 as needed
        if n <= 1:
            return mackay55[:n]
        elif n <= 13:
            return mackay55[:n]
        else:
            # Keep center + shell 1 + (n-13) from shell 2
            return mackay55[:n]
    else:
        # For n > 55, start with Mackay 55 and add more
        mackay55 = _get_mackay_55_positions()
        extra = n - 55
        R3 = 2.5  # Approximate radius for shell 3
        
        extra_points = []
        for i in range(extra):
            # Add points on a sphere of radius R3
            theta = rng.uniform(0, 2 * np.pi)
            phi = np.arccos(rng.uniform(-1, 1))
            p = np.array([
                R3 * np.sin(phi) * np.cos(theta),
                R3 * np.sin(phi) * np.sin(theta),
                R3 * np.cos(phi)
            ])
            extra_points.append(p)
        
        all_points = np.vstack([mackay55, np.array(extra_points)])
        return all_points[:n]

def _build_bond_pairs(positions, max_bond_length=2.0):
    """
    Build bond pairs based on proximity.
    Atoms within max_bond_length are considered bonded.
    """
    n = len(positions)
    bond_pairs = []
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < max_bond_length:
                bond_pairs.append((i, j))
    
    return bond_pairs

def _generate_mackay_like(n, rng):
    """
    Generate a Mackay-like icosahedral structure.
    """
    return _get_mackay_positions(n, rng)

def _generate_dense_cluster(n, rng):
    """
    Generate a dense cluster by random packing.
    """
    radius = 1.0 * (n / 13) ** (1/3) * 1.3
    points = []
    attempts = 0
    min_dist = 0.5
    
    while len(points) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < min_dist:
                    valid = False
                    break
            if valid:
                points.append(p)
        attempts += 1
    
    # If we didn't get enough, add random points
    while len(points) < n:
        points.append(rng.uniform(-radius, radius, 3))
    
    return np.array(points)

def _polish_structure(positions, rng, max_iter=100):
    """
    Polish a structure using L-BFGS-B with barrier-augmented energy.
    """
    n = len(positions)
    
    # Build initial bond pairs
    bond_pairs = _build_bond_pairs(positions, max_bond_length=2.0)
    
    def objective(x):
        pos = x.reshape(n, 3)
        # Check minimum distance constraint
        dists = np.linalg.norm(pos[:, np.newaxis, :] - pos[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        if np.min(dists) < 0.49:
            return 1e10
        e, g = _barrier_energy_and_gradient(pos, bond_pairs, barrier_strength=50.0)
        return e, g
    
    x0 = positions.flatten()
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', jac=True, 
                      options={'maxiter': max_iter, 'gtol': 1e-6, 'ftol': 1e-8})
        if res.success or res.fun < 1e9:
            polished = res.x.reshape(n, 3)
            # Verify minimum distance
            dists = np.linalg.norm(polished[:, np.newaxis, :] - polished[np.newaxis, :, :], axis=2)
            np.fill_diagonal(dists, np.inf)
            if np.min(dists) >= 0.49:
                return polished, res.fun
    except Exception:
        pass
    
    return positions, None

def _topological_optimize(n, rng, time_limit=300.0, start_time=None):
    """
    Topological optimization: start from Mackay-like structure, 
    build bond connectivity, then optimize with barrier-augmented L-BFGS.
    """
    if start_time is None:
        start_time = time.time()
    
    # Start from Mackay-like structure
    pos = _generate_mackay_like(n, rng)
    
    # Add small noise to break symmetry
    pos = pos + 0.01 * rng.standard_normal((n, 3))
    
    # Build bond pairs
    bond_pairs = _build_bond_pairs(pos, max_bond_length=2.0)
    
    def objective(x):
        pos_local = x.reshape(n, 3)
        # Check minimum distance constraint
        dists = np.linalg.norm(pos_local[:, np.newaxis, :] - pos_local[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        if np.min(dists) < 0.49:
            return 1e10, np.zeros_like(x)
        e, g = _barrier_energy_and_gradient(pos_local, bond_pairs, barrier_strength=50.0)
        return e, g.flatten()
    
    x0 = pos.flatten()
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', jac=True,
                      options={'maxiter': 200, 'gtol': 1e-5, 'ftol': 1e-8})
        if res.success or res.fun < 1e9:
            optimized = res.x.reshape(n, 3)
            # Verify minimum distance
            dists = np.linalg.norm(optimized[:, np.newaxis, :] - optimized[np.newaxis, :, :], axis=2)
            np.fill_diagonal(dists, np.inf)
            if np.min(dists) >= 0.49:
                return optimized, res.fun
    except Exception:
        pass
    
    # Fallback
    energy, _ = _energy_and_gradient(pos)
    return pos, energy

def construct(instance):
    start_time = time.time()
    time_limit = 300.0
    safety_margin = 15.0
    n = instance["n"]
    potential = instance["potential"]
    param = None
    
    rng = np.random.default_rng(n + 42)
    
    best_energy = float('inf')
    best_positions = None
    
    def check_time():
        return (time.time() - start_time) < (time_limit - safety_margin)
    
    def try_relax(pos, label=""):
        nonlocal best_energy, best_positions
        try:
            pos_noisy = pos + 0.01 * rng.standard_normal((n, 3))
            energy, relaxed_pos = relax(pos_noisy, potential, param)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pos.copy()
                return True
        except Exception:
            pass
        return False
    
    def try_topological(pos, label=""):
        nonlocal best_energy, best_positions
        try:
            optimized, energy = _topological_optimize(n, rng, time_limit, start_time)
            if energy is not None and energy < best_energy:
                best_energy = energy
                best_positions = optimized.copy()
                return True
        except Exception:
            pass
        return False
    
    # Phase 1: Topological optimization from Mackay-like starting points
    for i in range(3):
        if not check_time():
            break
        pos = _generate_mackay_like(n, rng)
        try_topological(pos, f"topo_{i}")
    
    # Phase 2: Relax the best found
    if best_positions is not None:
        try_relax(best_positions, "relax_best")
    
    # Phase 3: Generate additional starting points and relax
    for i in range(5):
        if not check_time():
            break
        
        # Alternate between Mackay-like and dense random
        if i % 2 == 0:
            pos = _generate_mackay_like(n, rng)
        else:
            pos = _generate_dense_cluster(n, rng)
        
        try_relax(pos, f"init_{i}")
    
    # Phase 4: Basin hopping from the best
    if best_positions is not None:
        kicks = 15
        for i in range(kicks):
            if not check_time():
                break
            kick_size = 0.1 + 0.03 * i
            pos_kicked = best_positions + kick_size * rng.standard_normal((n, 3))
            try_relax(pos_kicked, f"kick_{i}")
    
    # Phase 5: Final topological polish if time remains
    if best_positions is not None and check_time():
        try_topological(best_positions, "final_topo")
    
    # Fallback
    if best_positions is None:
        best_positions = _generate_mackay_like(n, rng)
    
    return best_positions

# EVOLVE-BLOCK-END

# EVOLVE-BLOCK-END
