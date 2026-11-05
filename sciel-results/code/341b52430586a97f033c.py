import numpy as np
import time
from autoresearch_suite.chem_clusters import relax
from scipy.optimize import minimize

# Dzugutov potential parameters
A = 5.82
B = 1.28
a_pot = 1.87
b_pot = 1.94
c_pot = 1.10
d_pot = 0.27
m_pot = 16.0

# Precompute Mackay icosahedron positions for various n
def _mackay_positions(n, rng):
    """Generate Mackay icosahedral positions: 1 + 12 + 42 = 55 atoms."""
    phi = (1 + np.sqrt(5)) / 2
    
    # 12 vertices of icosahedron
    ico = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1],
        [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    # 42 positions in second shell (Mackay 55-atom icosahedron)
    # These are at the centers of faces and edge midpoints of the icosahedron
    # For simplicity, we'll use a dense spherical distribution
    
    points = [np.zeros(3)]
    points.extend(list(ico))
    
    if n <= 13:
        return np.array(points[:n])
    
    # For n > 13, add atoms on second shell
    n_rest = n - 13
    R2 = 1.75  # Approximate second shell radius
    
    # Generate well-distributed points on sphere
    shell2 = []
    min_dist = 0.5
    
    # Use Fibonacci sphere for good distribution
    golden_angle = np.pi * (3 - np.sqrt(5))
    for i in range(n_rest):
        z = 1 - 2 * (i + 0.5) / n_rest
        r = np.sqrt(1 - z**2)
        theta = golden_angle * i
        p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * R2
        shell2.append(p)
    
    shell2 = np.array(shell2)
    points.extend(list(shell2))
    
    return np.array(points[:n])

def _energy_and_gradient(positions):
    """
    Vectorized calculation of energy and gradient for Dzugutov potential.
    Returns (energy, gradient).
    """
    n = len(positions)
    if n < 2:
        return 0.0, np.zeros_like(positions)
    
    # Compute pairwise differences
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    
    # Mask to avoid division by zero and self-interaction
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    r = np.where(mask, r, 1.0)
    
    energy = 0.0
    grad = np.zeros_like(positions)
    
    # Region 1: r < a
    m1 = r < a_pot
    # Region 2: a <= r < b
    m2 = (r >= a_pot) & (r < b_pot)
    
    r_safe1 = np.where(m1, r, 1.0)
    r_safe2 = np.where(m2, r, 1.0)
    
    e1_vals = A * (r_safe1 ** (-m_pot) - B) * np.exp(c_pot / (r_safe1 - a_pot))
    e2_vals = B * np.exp(d_pot / (r_safe2 - b_pot))
    
    e1 = np.where(m1, e1_vals, 0.0)
    e2 = np.where(m2, e2_vals, 0.0)
    
    pair_energy = e1 + e2
    energy = 0.5 * np.sum(pair_energy)
    
    # Gradient
    exp_term1 = np.exp(c_pot / (r_safe1 - a_pot))
    de1_dr_vals = A * (
        -m_pot * r_safe1 ** (-m_pot - 1) * exp_term1 + 
        (r_safe1 ** (-m_pot) - B) * exp_term1 * (-c_pot / (r_safe1 - a_pot) ** 2)
    )
    de1_dr = np.where(m1, de1_dr_vals, 0.0)
    
    exp_term2 = np.exp(d_pot / (r_safe2 - b_pot))
    de2_dr_vals = B * exp_term2 * (-d_pot / (r_safe2 - b_pot) ** 2)
    de2_dr = np.where(m2, de2_dr_vals, 0.0)
    
    de_dr = de1_dr + de2_dr
    
    factor = np.where(r > 1e-12, de_dr / r, 0.0)
    factor = np.where(mask, factor, 0.0)
    
    grad = np.einsum('ij,ijk->ik', factor, diff)
    
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

def _generate_mackay_like(n, rng):
    """
    Generate a Mackay-like icosahedral structure.
    """
    return _mackay_positions(n, rng)

def _generate_dense_cluster(n, rng, radius=None):
    """Generate a dense cluster with well-spaced atoms."""
    if radius is None:
        radius = 1.2 * (n / 13) ** (1.0/3.0)
    
    points = []
    attempts = 0
    min_dist = 0.5
    
    while len(points) < n and attempts < 10000:
        # Sample from ball
        r = radius * rng.random() ** (1.0/3.0)
        theta = 2 * np.pi * rng.random()
        phi = np.arccos(2 * rng.random() - 1)
        p = np.array([
            r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi)
        ])
        
        valid = True
        for q in points:
            if np.linalg.norm(p - q) < min_dist:
                valid = False
                break
        if valid:
            points.append(p)
        attempts += 1
    
    # Fill remaining with random points
    while len(points) < n:
        r = radius * rng.random() ** (1.0/3.0)
        theta = 2 * np.pi * rng.random()
        phi = np.arccos(2 * rng.random() - 1)
        p = np.array([
            r * np.sin(phi) * np.cos(theta),
            r * np.sin(phi) * np.sin(theta),
            r * np.cos(phi)
        ])
        points.append(p)
    
    return np.array(points)

def _parametric_optimize(n, rng):
    """
    Optimize the radii of icosahedral shells.
    """
    ico_dir = _icosahedron_vertices(1.0)
    
    n_rest = max(0, n - 13)
    if n_rest > 0:
        # Use Fibonacci sphere for well-distributed directions
        golden_angle = np.pi * (3 - np.sqrt(5))
        rest_dirs = []
        for i in range(n_rest):
            z = 1 - 2 * (i + 0.5) / n_rest
            r = np.sqrt(1 - z**2)
            theta = golden_angle * i
            d = np.array([r * np.cos(theta), r * np.sin(theta), z])
            rest_dirs.append(d)
        rest_dirs = np.array(rest_dirs)
    else:
        rest_dirs = np.array([])
    
    def objective(params):
        r1, r2 = params
        if r1 < 0.5 or r2 < 0.5:
            return 1e10
        pos = np.zeros((n, 3))
        pos[0] = 0
        for i in range(12):
            if 1 + i < n:
                pos[1 + i] = ico_dir[i] * r1
        for i in range(n_rest):
            if 13 + i < n:
                pos[13 + i] = rest_dirs[i] * r2
        
        dists = np.linalg.norm(pos[:, np.newaxis, :] - pos[np.newaxis, :, :], axis=2)
        np.fill_diagonal(dists, np.inf)
        if np.min(dists) < 0.4:
            return 1e10
            
        energy, _ = _energy_and_gradient(pos)
        return energy
        
    r1_init = 1.0
    r2_init = 1.7
    x0 = [r1_init, r2_init]
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
        if res.success:
            r1_opt, r2_opt = res.x
            pos = np.zeros((n, 3))
            pos[0] = 0
            for i in range(12):
                if 1 + i < n:
                    pos[1 + i] = ico_dir[i] * r1_opt
            for i in range(n_rest):
                if 13 + i < n:
                    pos[13 + i] = rest_dirs[i] * r2_opt
            return pos, res.fun
    except Exception:
        pass
        
    pos = _generate_mackay_like(n, rng)
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
        
    def try_relax(pos):
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

    # Phase 1: Generate diverse initial guesses
    initial_guesses = []
    
    # 1. Mackay icosahedral (primary structure for Dzugutov)
    initial_guesses.append(_generate_mackay_like(n, rng))
    
    # 2. Parametric icosahedral optimization
    pos, e = _parametric_optimize(n, rng)
    initial_guesses.append(pos)
    
    # 3. Dense random clusters with different radii
    for radius_factor in [0.9, 1.0, 1.1, 1.2]:
        radius = 1.0 * (n / 13) ** (1.0/3.0) * radius_factor
        initial_guesses.append(_generate_dense_cluster(n, rng, radius))
    
    # 4. Additional Mackay variants with noise
    for i in range(2):
        pos = _generate_mackay_like(n, rng)
        pos += 0.1 * rng.standard_normal((n, 3))
        initial_guesses.append(pos)

    # Phase 2: Relax all initial guesses
    for pos in initial_guesses:
        if not check_time(): break
        try_relax(pos)
        
    # Phase 3: Basin Hopping from multiple good structures
    if best_positions is not None:
        # Keep a list of good structures for hopping
        good_structures = [best_positions.copy()]
        
        # Add a few more diverse starting points for hopping
        for pos in initial_guesses[:3]:
            if not check_time(): break
            try:
                pos_noisy = pos + 0.05 * rng.standard_normal((n, 3))
                energy, relaxed_pos = relax(pos_noisy, potential, param)
                if energy < best_energy * 1.1:  # Within 10% of best
                    good_structures.append(relaxed_pos.copy())
            except Exception:
                pass
        
        # Basin hopping from each good structure
        for struct_idx, struct in enumerate(good_structures):
            if not check_time(): break
            
            # Adaptive kick size
            base_kick = 0.15
            for i in range(15):
                if not check_time(): break
                
                kick_size = base_kick * (1.0 + 0.1 * i)
                pos_kicked = struct + kick_size * rng.standard_normal((n, 3))
                
                try_relax(pos_kicked)
                
                # If we found a new best, update our reference
                if best_positions is not None and np.linalg.norm(best_positions - struct) > 0.3:
                    struct = best_positions.copy()
                    break

    # Phase 4: Final polish - try to improve with smaller kicks
    if best_positions is not None and check_time():
        for i in range(10):
            if not check_time(): break
            kick_size = 0.05 + 0.01 * i
            pos_kicked = best_positions + kick_size * rng.standard_normal((n, 3))
            try_relax(pos_kicked)

    # Fallback
    if best_positions is None:
        best_positions = _generate_mackay_like(n, rng)
        
    return best_positions

# EVOLVE-BLOCK-END
