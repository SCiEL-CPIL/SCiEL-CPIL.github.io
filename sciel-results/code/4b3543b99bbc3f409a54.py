"""Dzugutov cluster global minima search using basin hopping with diverse polytetrahedral starts."""
import numpy as np
import time
import itertools
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

def _build_mackay_icosahedron(n, rng, r_min=R_MIN):
    """
    Build a Mackay icosahedral structure with proper topology.
    For n=13: center + 12 vertices
    For n=55: 1 + 12 + 42 (Mackay icosahedron)
    For other n: start with appropriate Mackay shell and add/remove atoms.
    """
    phi = (1 + np.sqrt(5)) / 2
    ico = _icosahedron_vertices(1.0)
    
    # Base icosahedron (13 atoms: center + 12 vertices)
    base_positions = [np.zeros(3)] + list(ico)
    
    if n <= 13:
        positions = base_positions[:n]
        return np.array(positions) * r_min
    
    # For n > 13, add second shell atoms
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
    
    # If we still need more atoms, add random positions near the surface
    if len(all_positions) < n:
        remaining = n - len(all_positions)
        for _ in range(remaining):
            # Pick a random direction and place at second shell radius
            dir_vec = rng.standard_normal(3)
            dir_vec = dir_vec / np.linalg.norm(dir_vec)
            pos = dir_vec * r2_scale
            all_positions.append(pos)
    
    positions = np.array(all_positions[:n]) * r_min
    
    # Add small noise to break symmetry
    positions = positions + 0.01 * rng.standard_normal((n, 3))
    
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

def _dzugutov_energy_and_gradient(positions):
    """Compute Dzugutov potential energy and gradient."""
    n = len(positions)
    if n < 2:
        return 0.0, np.zeros_like(positions)
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    
    iu = np.triu_indices(n, k=1)
    r2_pair = r2[iu]
    valid = r2_pair > 1e-12
    r = np.sqrt(r2_pair[valid])
    
    energy = 0.0
    grad = np.zeros_like(positions)
    
    mask1 = r < a
    mask2 = (r >= a) & (r < b)
    
    if np.any(mask1):
        r1 = r[mask1]
        # V(r) = A*(r^-m - B)*exp(c/(r-a))
        # dV/dr = A*[(-m*r^(-m-1))*exp(c/(r-a)) + (r^-m - B)*exp(c/(r-a))*(-c/(r-a)^2)]
        exp_term = np.exp(c_true / (r1 - a))
        r_m = r1 ** (-m)
        dV_dr = A * (-m * r1 ** (-m - 1) * exp_term + (r_m - B) * exp_term * (-c_true / (r1 - a) ** 2))
        energy += np.sum(A * (r_m - B) * exp_term)
        
        # For each pair (i,j), force on i is -dV/dr * (r_i - r_j)/r
        idx1 = iu[0][valid][mask1]
        idx2 = iu[1][valid][mask1]
        for k in range(len(r1)):
            i, j = idx1[k], idx2[k]
            rij = positions[i] - positions[j]
            f = -dV_dr[k] / r1[k] * rij
            grad[i] += f
            grad[j] -= f
    
    if np.any(mask2):
        r2_ = r[mask2]
        # V(r) = B*exp(d/(r-b))
        # dV/dr = B*exp(d/(r-b))*(-d/(r-b)^2)
        exp_term = np.exp(d_true / (r2_ - b))
        dV_dr = B * exp_term * (-d_true / (r2_ - b) ** 2)
        energy += np.sum(B * exp_term)
        
        idx1 = iu[0][valid][mask2]
        idx2 = iu[1][valid][mask2]
        for k in range(len(r2_)):
            i, j = idx1[k], idx2[k]
            rij = positions[i] - positions[j]
            f = -dV_dr[k] / r2_[k] * rij
            grad[i] += f
            grad[j] -= f
    
    return float(energy), grad

def _polish_structure(positions, maxiter=500):
    """Polish a structure using L-BFGS-B with the Dzugutov energy and gradient."""
    flat = positions.flatten()
    
    def objective(x):
        pos = x.reshape(-1, 3)
        e, g = _dzugutov_energy_and_gradient(pos)
        return e, g.flatten()
    
    try:
        result = minimize(objective, flat, jac=True, method='L-BFGS-B',
                         options={'maxiter': maxiter, 'gtol': 1e-8})
        if result.success:
            return result.x.reshape(-1, 3), result.fun
    except Exception:
        pass
    return positions, _dzugutov_energy(positions)

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
    """Find low-energy Dzugutov clusters using basin hopping with diverse polytetrahedral starts."""
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
    
    # Phase 1: Generate diverse initial structures
    initial_structures = []
    
    # Mackay icosahedron
    mackay_pos = _build_mackay_icosahedron(n, rng)
    initial_structures.append(mackay_pos)
    
    # Random compact structures
    for i in range(8):
        if not check_time():
            break
        random_pos = _build_random_compact(n, rng)
        initial_structures.append(random_pos)
    
    # Deformed Mackay structures
    for i in range(5):
        if not check_time():
            break
        noise_scale = 0.05 + 0.1 * i
        deformed = mackay_pos + noise_scale * rng.standard_normal((n, 3))
        initial_structures.append(deformed)
    
    # Phase 2: Relax all initial structures
    for i, pos in enumerate(initial_structures):
        if not check_time():
            break
        try_relax(pos, f"init_{i}")
    
    # Phase 3: Basin hopping from best result with adaptive kick sizes
    if best_positions is not None and check_time():
        # Adaptive kick schedule: start small, grow, then shrink
        kick_schedule = [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 0.5, 0.3, 0.2, 0.1]
        
        for kick_size in kick_schedule:
            if not check_time():
                break
            
            # Number of kicks depends on remaining time
            remaining = time_limit - safety_margin - (time.time() - start_time)
            n_kicks = max(2, int(15 * remaining / time_limit))
            
            for i in range(n_kicks):
                if not check_time():
                    break
                
                pos_kicked = best_positions + kick_size * rng.standard_normal((n, 3))
                try_relax(pos_kicked, f"kick_{kick_size}_{i}")
                
                # If we found a better structure, polish it
                if best_positions is not None and check_time():
                    polished, pol_energy = _polish_structure(best_positions)
                    if pol_energy < best_energy:
                        best_energy = pol_energy
                        best_positions = polished.copy()
    
    # Phase 4: Additional diverse starts if time permits
    if check_time():
        for i in range(8):
            if not check_time():
                break
            
            # More random starts
            pos = _build_random_compact(n, rng)
            try_relax(pos, f"extra_random_{i}")
            
            # More deformed Mackay
            noise_scale = 0.05 + 0.03 * i
            pos = mackay_pos + noise_scale * rng.standard_normal((n, 3))
            try_relax(pos, f"extra_deformed_{i}")
            
            # Polish the best found so far
            if best_positions is not None and check_time():
                polished, pol_energy = _polish_structure(best_positions)
                if pol_energy < best_energy:
                    best_energy = pol_energy
                    best_positions = polished.copy()
    
    # Final polish
    if best_positions is not None and check_time():
        polished, pol_energy = _polish_structure(best_positions, maxiter=1000)
        if pol_energy < best_energy:
            best_energy = pol_energy
            best_positions = polished.copy()
    
    # Fallback
    if best_positions is None:
        best_positions = _build_mackay_icosahedron(n, rng)
        
    return best_positions
