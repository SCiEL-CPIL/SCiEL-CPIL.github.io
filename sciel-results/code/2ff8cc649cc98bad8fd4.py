"""Dzugutov cluster global minima search using topological optimization with polytetrahedral constraints."""
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax
from scipy.optimize import minimize

# Dzugutov potential parameters
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c_true = 1.10
d_true = 0.27
m = 16.0

def _find_r_min():
    """Find the minimum energy distance r_min for the Dzugutov potential."""
    r = np.linspace(0.5, 2.0, 1000)
    energies = []
    for ri in r:
        if ri < a:
            e = A * (ri ** (-m) - B) * np.exp(c_true / (ri - a))
        elif ri < b:
            e = B * np.exp(d_true / (ri - b))
        else:
            e = 0.0
        energies.append(e)
    energies = np.array(energies)
    idx = np.argmin(energies)
    return r[idx]

R_MIN = _find_r_min()

def _icosahedron_vertices(radius=1.0):
    """Generate 12 vertices of a regular icosahedron centered at origin."""
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

def _build_mackay_icosahedron(n, rng, r_min=R_MIN):
    """
    Build a Mackay icosahedral structure.
    For n=13: center + 12 vertices
    For n=55: 1 + 12 + 42 (Mackay icosahedron)
    For other n: start with appropriate Mackay shell and add/remove atoms.
    """
    points = [np.zeros(3)]
    
    # Shell 1: 12 icosahedron vertices
    ico = _icosahedron_vertices(r_min)
    points.extend(list(ico))
    
    if n <= 13:
        if n < 13:
            idx = rng.choice(13, n, replace=False)
            return np.array([points[i] for i in idx])
        return np.array(points)
    
    # Shell 2: Mackay icosahedron has 42 atoms on second shell
    # The 42 positions come from:
    # - 12 face centers (distance sqrt(1+1/phi^2) * r_min from center)
    # - 30 edge midpoints (distance sqrt(1+1/(4*phi^2)) * r_min from center)
    # Actually, let me use a different approach: generate points on a sphere
    
    # For Mackay icosahedron, the second shell has radius R2 = R1 * sqrt(1 + 1/phi^2)
    # where phi = golden ratio
    phi = (1 + np.sqrt(5)) / 2
    r2 = r_min * np.sqrt(1 + 1/phi**2)
    
    needed = min(n - 13, 42)
    added = 0
    attempts = 0
    min_dist = 0.5
    
    # Generate candidate points on the second shell using icosahedral symmetry
    # The 42 points of Mackay icosahedron second shell:
    # 12 face centers + 30 edge midpoints
    
    # Get icosahedron faces (triangles)
    ico_verts = _icosahedron_vertices(1.0)  # unit icosahedron
    n_ico = len(ico_verts)
    
    # Find all triangles (faces) of the icosahedron
    faces = []
    for i in range(n_ico):
        for j in range(i+1, n_ico):
            for k in range(j+1, n_ico):
                # Check if i,j,k form a face (equilateral triangle)
                d_ij = np.linalg.norm(ico_verts[i] - ico_verts[j])
                d_jk = np.linalg.norm(ico_verts[j] - ico_verts[k])
                d_ik = np.linalg.norm(ico_verts[i] - ico_verts[k])
                if abs(d_ij - d_jk) < 1e-6 and abs(d_jk - d_ik) < 1e-6:
                    # Check it's a face (not just any equilateral triangle)
                    # The edge length of icosahedron is 2 (for unit circumradius)
                    if d_ij < 2.5:  # threshold to distinguish faces from other triangles
                        faces.append((i, j, k))
    
    # Add face centers
    for face in faces:
        if added >= needed:
            break
        center = (ico_verts[face[0]] + ico_verts[face[1]] + ico_verts[face[2]]) / 3
        p = center / np.linalg.norm(center) * r2
        valid = True
        for q in points:
            if np.linalg.norm(p - q) < min_dist:
                valid = False
                break
        if valid:
            points.append(p)
            added += 1
    
    # Add edge midpoints if still needed
    if added < needed:
        edges = set()
        for face in faces:
            for i in range(3):
                for j in range(i+1, 3):
                    edge = tuple(sorted((face[i], face[j])))
                    edges.add(edge)
        
        for edge in edges:
            if added >= needed:
                break
            mid = (ico_verts[edge[0]] + ico_verts[edge[1]]) / 2
            p = mid / np.linalg.norm(mid) * r2
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < min_dist:
                    valid = False
                    break
            if valid:
                points.append(p)
                added += 1
    
    # If we need more than 55, add random points on outer shells
    while len(points) < n:
        r_outer = r2 + 0.3 * rng.random()
        u = rng.random()
        v = rng.random()
        theta = 2 * np.pi * u
        phi_ang = np.arccos(2 * v - 1)
        p = np.array([
            r_outer * np.sin(phi_ang) * np.cos(theta),
            r_outer * np.sin(phi_ang) * np.sin(theta),
            r_outer * np.cos(phi_ang)
        ])
        points.append(p)
    
    # If we have too many, remove random ones from the outermost
    if len(points) > n:
        # Keep center and shell 1, remove from shell 2
        keep = list(range(13))
        for i in range(13, len(points)):
            if len(keep) < n:
                keep.append(i)
        points = [points[i] for i in keep]
    
    return np.array(points)

def _build_random_compact(n, rng, r_min=R_MIN):
    """Build a random compact cluster."""
    # Estimate radius
    radius = r_min * (n / 13) ** (1/3) * 1.2
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
        
    while len(points) < n:
        base = points[len(points) % max(1, len(points))] if points else np.zeros(3)
        p = base + 0.2 * rng.standard_normal(3)
        points.append(p)
        
    return np.array(points)

def _dzugutov_energy(positions):
    """Compute exact Dzugutov potential energy."""
    n = len(positions)
    if n < 2:
        return 0.0
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    
    iu = np.triu_indices(n, k=1)
    r2_pair = r2[iu]
    
    energy = 0.0
    valid = r2_pair > 1e-12
    if not np.any(valid):
        return 0.0
    
    r = np.sqrt(r2_pair[valid])
    
    mask1 = r < a
    if np.any(mask1):
        r1 = r[mask1]
        energy += np.sum(A * (r1 ** (-m) - B) * np.exp(c_true / (r1 - a)))
    
    mask2 = (r >= a) & (r < b)
    if np.any(mask2):
        r2_ = r[mask2]
        energy += np.sum(B * np.exp(d_true / (r2_ - b)))
    
    return float(energy)

def _dzugutov_energy_flat(x, n):
    """Flattened Dzugutov energy for scipy.optimize."""
    positions = x.reshape(n, 3)
    return _dzugutov_energy(positions)

def _dzugutov_gradient_flat(x, n):
    """Compute gradient of Dzugutov potential energy."""
    positions = x.reshape(n, 3)
    grad = np.zeros_like(positions)
    if n < 2:
        return grad.flatten()
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    
    iu = np.triu_indices(n, k=1)
    r2_pair = r2[iu]
    valid = r2_pair > 1e-12
    if not np.any(valid):
        return grad.flatten()
    
    i_idx = iu[0][valid]
    j_idx = iu[1][valid]
    r = np.sqrt(r2_pair[valid])
    rij = diff[i_idx, j_idx]
    
    dV_dr = np.zeros_like(r)
    
    mask1 = r < a
    if np.any(mask1):
        r1 = r[mask1]
        exp_term = np.exp(c_true / (r1 - a))
        dV_dr[mask1] = A * (-m * r1 ** (-m - 1)) * exp_term + A * (r1 ** (-m) - B) * exp_term * (-c_true / (r1 - a) ** 2)
    
    mask2 = (r >= a) & (r < b)
    if np.any(mask2):
        r2_ = r[mask2]
        dV_dr[mask2] = B * np.exp(d_true / (r2_ - b)) * (-d_true / (r2_ - b) ** 2)
    
    factor = -dV_dr / r
    forces = factor[:, np.newaxis] * rij
    
    np.add.at(grad, i_idx, forces)
    np.add.at(grad, j_idx, -forces)
    
    return grad.flatten()

def _topological_energy(positions, bonds, r_target, barrier_strength=100.0):
    """
    Compute energy with topological constraints.
    bonds: list of (i, j) pairs that should be bonded
    r_target: target bond length (r_min)
    barrier_strength: strength of the barrier penalty
    """
    base_energy = _dzugutov_energy(positions)
    
    # Add barrier penalty for bonds deviating from r_target
    barrier = 0.0
    for i, j in bonds:
        d = np.linalg.norm(positions[i] - positions[j])
        if d > 1e-6:
            barrier += barrier_strength * (d - r_target) ** 2
    
    return base_energy + barrier

def _topological_energy_flat(x, n, bonds, r_target, barrier_strength=100.0):
    """Flattened topological energy."""
    positions = x.reshape(n, 3)
    return _topological_energy(positions, bonds, r_target, barrier_strength)

def _topological_gradient_flat(x, n, bonds, r_target, barrier_strength=100.0):
    """Compute gradient of topological energy."""
    positions = x.reshape(n, 3)
    grad = _dzugutov_gradient_flat(x, n).reshape(n, 3)
    
    # Add barrier gradient
    for i, j in bonds:
        rij = positions[i] - positions[j]
        d = np.linalg.norm(rij)
        if d > 1e-6:
            # d/dx of barrier_strength * (d - r_target)^2
            # = 2 * barrier_strength * (d - r_target) * d(d)/dx
            # d(d)/dx_i = rij / d (for atom i)
            # d(d)/x_j = -rij / d (for atom j)
            factor = 2 * barrier_strength * (d - r_target) / d
            grad[i] += factor * rij
            grad[j] -= factor * rij
    
    return grad.flatten()

def _find_bonds(positions, r_min, max_bonds_per_atom=12):
    """
    Identify bonds based on proximity to r_min.
    Returns list of (i, j) pairs where distance is within tolerance of r_min.
    """
    n = len(positions)
    bonds = []
    bond_count = np.zeros(n, dtype=int)
    
    # Compute all pairwise distances
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    r = np.sqrt(r2)
    
    # Sort pairs by distance
    iu = np.triu_indices(n, k=1)
    r_pair = r[iu]
    i_idx = iu[0]
    j_idx = iu[1]
    
    # Sort by distance
    order = np.argsort(r_pair)
    
    for idx in order:
        i, j = i_idx[idx], j_idx[idx]
        d = r_pair[idx]
        
        # Only consider distances close to r_min
        if d < 0.5 * r_min or d > 1.5 * r_min:
            continue
        
        # Check if either atom has too many bonds
        if bond_count[i] >= max_bonds_per_atom or bond_count[j] >= max_bonds_per_atom:
            continue
        
        bonds.append((i, j))
        bond_count[i] += 1
        bond_count[j] += 1
    
    return bonds

def _topological_optimize(positions, bonds, r_min, max_iter=200):
    """
    Optimize positions with topological constraints.
    Uses L-BFGS-B to minimize energy + barrier penalty.
    """
    n = len(positions)
    x0 = positions.flatten()
    
    def fun(x):
        return _topological_energy_flat(x, n, bonds, r_min)
    
    def jac(x):
        return _topological_gradient_flat(x, n, bonds, r_min)
    
    try:
        res = minimize(fun, x0, jac=jac, method='L-BFGS-B',
                      options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
        return res.x.reshape(n, 3)
    except Exception:
        return positions

def construct(instance):
    """Find low-energy Dzugutov clusters using topological optimization."""
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
            energy, relaxed_pos = relax(pos, potential, param)
            if energy < best_energy:
                best_energy = energy
                best_positions = relaxed_pos.copy()
                return True
        except Exception:
            pass
        return False
    
    # Phase 1: Generate diverse initial guesses
    initial_guesses = []
    
    # Mackay icosahedral guesses (primary)
    for i in range(3):
        if not check_time(): break
        pos = _build_mackay_icosahedron(n, rng)
        # Add small random perturbation to break symmetry
        pos_noisy = pos + 0.05 * rng.standard_normal((n, 3))
        initial_guesses.append(pos_noisy)
    
    # Random compact guesses
    for i in range(3):
        if not check_time(): break
        pos = _build_random_compact(n, rng)
        initial_guesses.append(pos)
    
    # Phase 2: Topological optimization for each initial guess
    for i, pos_guess in enumerate(initial_guesses):
        if not check_time(): break
        
        # Identify bonds from initial guess
        bonds = _find_bonds(pos_guess, R_MIN, max_bonds_per_atom=12)
        
        if len(bonds) > 0:
            # Optimize with topological constraints
            pos_opt = _topological_optimize(pos_guess, bonds, R_MIN, max_iter=100)
        else:
            # Fall back to standard optimization
            x0 = pos_guess.flatten()
            try:
                res = minimize(_dzugutov_energy_flat, x0, jac=lambda x: _dzugutov_gradient_flat(x, n),
                              method='L-BFGS-B', options={'maxiter': 100})
                pos_opt = res.x.reshape(n, 3)
            except Exception:
                pos_opt = pos_guess
        
        # Final relaxation
        try_relax(pos_opt, f"topo_{i}")
    
    # Phase 3: Basin Hopping on best solution
    kicks = 20
    if best_positions is not None:
        for i in range(kicks):
            if not check_time(): break
            kick_size = 0.15 + 0.03 * i
            pos_kicked = best_positions + kick_size * rng.standard_normal((n, 3))
            
            # Identify bonds from kicked structure
            bonds = _find_bonds(pos_kicked, R_MIN, max_bonds_per_atom=12)
            
            if len(bonds) > 0:
                pos_opt = _topological_optimize(pos_kicked, bonds, R_MIN, max_iter=50)
            else:
                x0 = pos_kicked.flatten()
                try:
                    res = minimize(_dzugutov_energy_flat, x0, jac=lambda x: _dzugutov_gradient_flat(x, n),
                                  method='L-BFGS-B', options={'maxiter': 50})
                    pos_opt = res.x.reshape(n, 3)
                except Exception:
                    pos_opt = pos_kicked
            
            try_relax(pos_opt, f"kick_{i}")
    
    # Phase 4: Additional random restarts if time permits
    extra_attempts = 8
    for i in range(extra_attempts):
        if not check_time(): break
        
        if i % 2 == 0:
            pos = _build_mackay_icosahedron(n, rng)
        else:
            pos = _build_random_compact(n, rng)
        
        pos_noisy = pos + 0.03 * rng.standard_normal((n, 3))
        
        bonds = _find_bonds(pos_noisy, R_MIN, max_bonds_per_atom=12)
        
        if len(bonds) > 0:
            pos_opt = _topological_optimize(pos_noisy, bonds, R_MIN, max_iter=50)
        else:
            x0 = pos_noisy.flatten()
            try:
                res = minimize(_dzugutov_energy_flat, x0, jac=lambda x: _dzugutov_gradient_flat(x, n),
                              method='L-BFGS-B', options={'maxiter': 50})
                pos_opt = res.x.reshape(n, 3)
            except Exception:
                pos_opt = pos_noisy
        
        try_relax(pos_opt, f"extra_{i}")
    
    # Fallback
    if best_positions is None:
        best_positions = _build_random_compact(n, rng)
        
    return best_positions
