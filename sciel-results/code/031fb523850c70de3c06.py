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
    r = np.linspace(0.5, 2.0, 2000)
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

def _get_mackay_topology(n):
    """
    Get the Mackay icosahedral topology for a given cluster size.
    Returns a list of atom positions (relative to center) and bond pairs.
    """
    # Mackay icosahedra: 1, 13, 55, 147...
    # For our range (38-60), we use modified Mackay structures
    
    # Base icosahedron (13 atoms: center + 12 vertices)
    ico = _icosahedron_vertices(1.0)
    base_positions = [np.zeros(3)] + list(ico)
    base_bonds = []
    
    # Add bonds between center and vertices
    for i in range(12):
        base_bonds.append((0, i+1))
    
    # Add bonds between adjacent vertices (icosahedron edges)
    ico_dist = np.linalg.norm(ico[0] - ico[1])
    for i in range(12):
        for j in range(i+1, 12):
            d = np.linalg.norm(ico[i] - ico[j])
            if abs(d - ico_dist) < 0.01:
                base_bonds.append((i+1, j+1))
    
    if n <= 13:
        return base_positions[:n], base_bonds
    
    # For n > 13, add second shell atoms
    # Second shell positions for Mackay icosahedron
    phi = (1 + np.sqrt(5)) / 2
    
    # Face centers (20 faces)
    faces = []
    for i in range(12):
        for j in range(i+1, 12):
            for k in range(j+1, 12):
                d_ij = np.linalg.norm(ico[i] - ico[j])
                d_jk = np.linalg.norm(ico[j] - ico[k])
                d_ik = np.linalg.norm(ico[i] - ico[k])
                if abs(d_ij - d_jk) < 0.01 and abs(d_jk - d_ik) < 0.01:
                    if d_ij < 2.5:
                        faces.append((i, j, k))
    
    # Edge midpoints (30 edges)
    edges = set()
    for face in faces:
        for i in range(3):
            for j in range(i+1, 3):
                edge = tuple(sorted((face[i], face[j])))
                edges.add(edge)
    
    # Generate second shell positions
    second_shell = []
    second_shell_bonds = []
    
    # Add face centers first (20 positions)
    for face in faces:
        center = (ico[face[0]] + ico[face[1]] + ico[face[2]]) / 3
        p = center / np.linalg.norm(center)
        second_shell.append(p)
    
    # Add edge midpoints (30 positions)
    for edge in edges:
        mid = (ico[edge[0]] + ico[edge[1]]) / 2
        p = mid / np.linalg.norm(mid)
        second_shell.append(p)
    
    # Scale second shell
    r2_scale = np.sqrt(1 + 1/phi**2)
    second_shell = [p * r2_scale for p in second_shell]
    
    # Select how many second shell atoms we need
    needed_second = min(n - 13, len(second_shell))
    selected_second = second_shell[:needed_second]
    
    all_positions = base_positions + selected_second
    
    # Add bonds for second shell atoms
    # Each second shell atom bonds to nearby first shell atoms
    for idx, pos in enumerate(selected_second):
        second_idx = 13 + idx
        for first_idx in range(12):
            d = np.linalg.norm(pos - ico[first_idx])
            if d < 1.5:  # Bonding distance threshold
                base_bonds.append((first_idx + 1, second_idx))
    
    # Add bonds between second shell atoms if they're close
    for i in range(len(selected_second)):
        for j in range(i+1, len(selected_second)):
            d = np.linalg.norm(selected_second[i] - selected_second[j])
            if d < 1.2:
                base_bonds.append((13 + i, 13 + j))
    
    return all_positions, base_bonds

def _build_mackay_icosahedron(n, rng, r_min=R_MIN):
    """
    Build a Mackay icosahedral structure with proper topology.
    """
    positions, bonds = _get_mackay_topology(n)
    positions = np.array(positions) * r_min
    
    # Add small noise to break symmetry
    positions = positions + 0.01 * rng.standard_normal((len(positions), 3))
    
    return positions

def _build_random_compact(n, rng, r_min=R_MIN):
    """Build a random compact cluster."""
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

def _topological_energy_flat(x_flat, n, bonds, r_target=R_MIN, k=50.0):
    """
    Topological energy: Dzugutov potential + bond length penalty.
    x_flat: flattened positions (3n,)
    bonds: list of (i, j) pairs
    r_target: target bond length
    k: penalty strength
    """
    positions = x_flat.reshape(n, 3)
    
    # Compute Dzugutov energy
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    
    iu = np.triu_indices(n, k=1)
    r2_pair = r2[iu]
    
    energy = 0.0
    valid = r2_pair > 1e-12
    if np.any(valid):
        r = np.sqrt(r2_pair[valid])
        
        mask1 = r < a
        if np.any(mask1):
            r1 = r[mask1]
            energy += np.sum(A * (r1 ** (-m) - B) * np.exp(c_true / (r1 - a)))
        
        mask2 = (r >= a) & (r < b)
        if np.any(mask2):
            r2_ = r[mask2]
            energy += np.sum(B * np.exp(d_true / (r2_ - b)))
    
    # Add bond length penalty
    for i, j in bonds:
        d = np.linalg.norm(positions[i] - positions[j])
        penalty = k * (d - r_target) ** 2
        energy += penalty
    
    return energy

def _topological_gradient_flat(x_flat, n, bonds, r_target=R_MIN, k=50.0):
    """Gradient of topological energy."""
    positions = x_flat.reshape(n, 3)
    grad = np.zeros_like(positions)
    
    # Dzugutov gradient
    for i in range(n):
        for j in range(i+1, n):
            diff = positions[i] - positions[j]
            r2 = np.sum(diff ** 2)
            if r2 < 1e-12:
                continue
            r = np.sqrt(r2)
            
            # Compute force
            if r < a:
                # dV/dr for r < a
                dr_term = A * (-m * r ** (-m-1)) * np.exp(c_true / (r - a)) + \
                          A * (r ** (-m) - B) * np.exp(c_true / (r - a)) * (-c_true / (r - a) ** 2)
            elif r < b:
                dr_term = B * np.exp(d_true / (r - b)) * (-d_true / (r - b) ** 2)
            else:
                dr_term = 0.0
            
            force = dr_term * diff / r
            grad[i] += force
            grad[j] -= force
    
    # Bond length penalty gradient
    for i, j in bonds:
        diff = positions[i] - positions[j]
        d = np.linalg.norm(diff)
        if d < 1e-12:
            continue
        penalty_grad = 2 * k * (d - r_target) * diff / d
        grad[i] += penalty_grad
        grad[j] -= penalty_grad
    
    return grad.flatten()

def _optimize_topological(positions, n, bonds, r_target=R_MIN, k=50.0, maxiter=100):
    """
    Optimize positions using topological energy with L-BFGS.
    """
    x0 = positions.flatten()
    
    result = minimize(
        _topological_energy_flat,
        x0,
        args=(n, bonds, r_target, k),
        jac=_topological_gradient_flat,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    return result.x.reshape(n, 3), result.fun

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

def construct(instance):
    """Find low-energy Dzugutov clusters using topological optimization with polytetrahedral constraints."""
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
    
    # Get Mackay topology
    positions_template, bonds = _get_mackay_topology(n)
    positions_template = np.array(positions_template) * R_MIN
    
    # Phase 1: Topological optimization from Mackay start
    if check_time():
        # Try different penalty strengths
        for k in [20.0, 50.0, 100.0]:
            if not check_time():
                break
            
            pos_init = positions_template + 0.02 * rng.standard_normal((n, 3))
            try:
                opt_pos, opt_energy = _optimize_topological(
                    pos_init, n, bonds, R_MIN, k, maxiter=200
                )
                try_relax(opt_pos, f"topo_k{k}")
            except Exception:
                pass
    
    # Phase 2: Basin hopping from best topological result
    if best_positions is not None and check_time():
        kick_sizes = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
        for kick_size in kick_sizes:
            if not check_time():
                break
            
            n_kicks = max(1, int(15 * (time_limit - safety_margin - (time.time() - start_time)) / time_limit))
            for i in range(n_kicks):
                if not check_time():
                    break
                
                pos_kicked = best_positions + kick_size * rng.standard_normal((n, 3))
                try_relax(pos_kicked, f"kick_{kick_size}_{i}")
    
    # Phase 3: Additional diverse starts if time permits
    if check_time():
        for i in range(5):
            if not check_time():
                break
            
            # Deformed Mackay with different noise
            noise_scale = 0.05 + 0.1 * i
            pos = positions_template + noise_scale * rng.standard_normal((n, 3))
            
            # Topological optimization
            try:
                opt_pos, opt_energy = _optimize_topological(
                    pos, n, bonds, R_MIN, k=50.0, maxiter=100
                )
                try_relax(opt_pos, f"deformed_topo_{i}")
            except Exception:
                pass
            
            # Direct relaxation
            try_relax(pos, f"deformed_{i}")
    
    # Phase 4: Random compact starts as fallback
    if check_time():
        for i in range(3):
            if not check_time():
                break
            pos = _build_random_compact(n, rng)
            try_relax(pos, f"random_{i}")
    
    # Fallback
    if best_positions is None:
        best_positions = _build_mackay_icosahedron(n, rng)
        
    return best_positions
