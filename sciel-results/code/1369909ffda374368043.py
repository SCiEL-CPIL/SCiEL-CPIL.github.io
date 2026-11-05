"""Dzugutov cluster global minima search using polytetrahedral/icosahedral initial guesses with basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, dzugutov_energy_gradient


# Dzugutov potential parameters
A = 5.82
B = 1.28
a = 1.87
b = 1.94
c = 1.10
d = 0.27
m = 16.0


def _dzugutov_energy(positions):
    """Compute Dzugutov potential energy (vectorized)."""
    n = len(positions)
    energy = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r = np.linalg.norm(positions[i] - positions[j])
            if r < 1e-12:
                continue
            if r < a:
                energy += A * (r ** (-m) - B) * np.exp(c / (r - a))
            elif r < b:
                energy += B * np.exp(d / (r - b))
    return energy


def _energy_and_gradient_flat(x):
    """Compute energy and gradient for flat position array."""
    n = x.shape[0] // 3
    pos = x.reshape(n, 3)
    energy = 0.0
    grad = np.zeros_like(x)
    for i in range(n):
        for j in range(i + 1, n):
            r_vec = pos[i] - pos[j]
            r = np.linalg.norm(r_vec)
            if r < 1e-12:
                continue
            r_hat = r_vec / r
            if r < a:
                # V(r) = A(r^-m - B)exp(c/(r-a))
                # dV/dr = A[-m*r^(-m-1) * exp(c/(r-a)) + (r^-m - B) * exp(c/(r-a)) * (-c/(r-a)^2)]
                exp_term = np.exp(c / (r - a))
                dV_dr = A * (-m * r ** (-m - 1) * exp_term + (r ** (-m) - B) * exp_term * (-c / ((r - a) ** 2)))
                energy += A * (r ** (-m) - B) * exp_term
            elif r < b:
                # V(r) = B * exp(d/(r-b))
                # dV/dr = B * exp(d/(r-b)) * (-d/(r-b)^2)
                exp_term = np.exp(d / (r - b))
                dV_dr = B * exp_term * (-d / ((r - b) ** 2))
                energy += B * exp_term
            else:
                continue
            
            # grad_i += dV/dr * r_hat, grad_j -= dV/dr * r_hat
            grad[3*i:3*i+3] += dV_dr * r_hat
            grad[3*j:3*j+3] -= dV_dr * r_hat
    
    return energy, grad


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


def _get_icosahedron_faces():
    """Get the 20 faces of the icosahedron."""
    phi = (1 + np.sqrt(5)) / 2
    verts_unit = np.array([
        [0, 1, phi], [0, -1, phi],
        [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0],
        [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1],
        [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    norms = np.linalg.norm(verts_unit, axis=1)
    verts_unit = verts_unit / norms[:, np.newaxis]
    
    faces = []
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                d1 = np.linalg.norm(verts_unit[i] - verts_unit[j])
                d2 = np.linalg.norm(verts_unit[j] - verts_unit[k])
                d3 = np.linalg.norm(verts_unit[k] - verts_unit[i])
                edge_len = 2.0 / np.sqrt(phi * phi + 1)
                if abs(d1 - edge_len) < 1e-6 and abs(d2 - edge_len) < 1e-6 and abs(d3 - edge_len) < 1e-6:
                    faces.append((i, j, k))
    return faces, verts_unit


def _build_mackay_icosahedron(n, rng):
    """Build a Mackay icosahedral cluster for n atoms."""
    # Mackay icosahedron: 1 + 12 + 42 + 92 + ...
    # For our sizes (38-60), we use first shell (13) + part of second shell
    
    r1 = 1.0  # first shell radius
    r2 = 1.7  # second shell radius
    
    points = []
    
    # Center atom
    points.append(np.array([0.0, 0.0, 0.0]))
    
    # First shell: icosahedron vertices
    ico = _icosahedron_vertices(r1)
    points.extend(list(ico))
    
    if n <= 13:
        idx = rng.choice(len(points), n, replace=False)
        return np.array([points[i] for i in idx])
    
    # Second shell: face centers and edge midpoints
    faces, verts_unit = _get_icosahedron_faces()
    
    # Face centers (20 faces)
    face_centers = []
    for face in faces:
        fc = np.mean(verts_unit[list(face)], axis=0)
        fc = fc / np.linalg.norm(fc)
        face_centers.append(fc)
    face_centers = np.array(face_centers)
    
    # Edge midpoints (30 edges)
    edge_mids = []
    for i in range(12):
        for j in range(i + 1, 12):
            d = np.linalg.norm(verts_unit[i] - verts_unit[j])
            edge_len = 2.0 / np.sqrt(phi * phi + 1) if 'phi' in dir() else 2.0 / np.sqrt(((1+np.sqrt(5))/2)**2 + 1)
            phi = (1 + np.sqrt(5)) / 2
            edge_len = 2.0 / np.sqrt(phi * phi + 1)
            if abs(d - edge_len) < 1e-6:
                em = (verts_unit[i] + verts_unit[j]) / 2
                em = em / np.linalg.norm(em)
                edge_mids.append(em)
    edge_mids = np.array(edge_mids)
    
    # Generate second shell candidates
    second_shell = []
    for fc in face_centers:
        second_shell.append(fc * r2)
    for em in edge_mids:
        second_shell.append(em * r2)
    for v in verts_unit:
        second_shell.append(v * r2)
    
    second_shell = np.array(second_shell)
    
    needed = n - len(points)
    if needed <= 0:
        return np.array(points[:n])
    
    # Shuffle and select
    idx = rng.permutation(len(second_shell))
    for i in range(min(needed, len(second_shell))):
        points.append(second_shell[idx[i]])
    
    # If we still need more, add random points
    while len(points) < n:
        p = rng.uniform(-r2, r2, 3)
        if np.linalg.norm(p) <= r2:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.5:
                    valid = False
                    break
            if valid:
                points.append(p)
            else:
                points.append(p + 0.1 * rng.standard_normal(3))
    
    return np.array(points[:n])


def _build_random_compact(n, rng):
    """Build a random compact cluster."""
    spacing = 1.1
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < 10000:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in points:
                if np.linalg.norm(p - q) < 0.6 * spacing:
                    valid = False
                    break
            if valid:
                points.append(p)
        attempts += 1
    
    while len(points) < n:
        base = points[len(points) % max(1, len(points))] if points else np.zeros(3)
        p = base + 0.3 * rng.standard_normal(3)
        points.append(p)
    
    return np.array(points[:n])


def _polish_structure(positions, potential, parameter):
    """Polish a structure using scipy's L-BFGS-B with energy/gradient."""
    x0 = positions.flatten()
    
    def fun(x):
        e, g = _energy_and_gradient_flat(x)
        return e, g
    
    result = minimize(fun, x0, jac=True, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
    return result.x.reshape(-1, 3), result.fun


def _basin_hopping(positions, potential, parameter, rng, time_limit, start_time, best_energy, best_positions):
    """Perform basin hopping around the given structure."""
    current_energy = best_energy
    current_positions = best_positions.copy()
    
    kick_scale = 0.3
    n_iterations = 0
    max_iterations = 50
    
    while n_iterations < max_iterations:
        elapsed = time.time() - start_time
        if elapsed > time_limit - 10.0:
            break
        
        # Kick
        kick = kick_scale * rng.standard_normal((len(positions), 3))
        kicked_positions = current_positions + kick
        
        # Polish
        try:
            polished_positions, polished_energy = _polish_structure(kicked_positions, potential, parameter)
        except Exception:
            polished_positions = kicked_positions
            polished_energy = _dzugutov_energy(kicked_positions)
        
        # Metropolis acceptance
        if polished_energy < current_energy or rng.random() < np.exp(-(polished_energy - current_energy) / 0.5):
            current_energy = polished_energy
            current_positions = polished_positions
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_positions = current_positions.copy()
        
        # Reduce kick scale
        kick_scale *= 0.95
        n_iterations += 1
    
    return best_energy, best_positions


def construct(instance):
    """Find low-energy Dzugutov clusters using polytetrahedral initial guesses with basin hopping."""
    start_time = time.time()
    time_limit = 300.0
    safety_margin = 15.0
    
    size = instance["n"]
    potential = instance["potential"]
    parameter = instance.get("rho", instance.get("kind"))
    
    rng = np.random.default_rng(size)
    
    best_energy = float('inf')
    best_positions = None
    
    # Generate diverse initial candidates
    n_candidates = 8
    
    candidates = []
    
    # Icosahedral candidates
    for i in range(3):
        try:
            pos = _build_mackay_icosahedron(size, rng)
            pos = pos + 0.05 * rng.standard_normal((size, 3))
            candidates.append(pos)
        except Exception:
            pass
    
    # Random compact candidates
    for i in range(3):
        try:
            pos = _build_random_compact(size, rng)
            pos = pos + 0.03 * rng.standard_normal((size, 3))
            candidates.append(pos)
        except Exception:
            pass
    
    # Mixed candidates
    for i in range(2):
        try:
            ico = _build_mackay_icosahedron(size, rng)
            rand = _build_random_compact(size, rng)
            pos = 0.5 * ico + 0.5 * rand
            pos = pos + 0.05 * rng.standard_normal((size, 3))
            candidates.append(pos)
        except Exception:
            pass
    
    # Relax each candidate and keep the best
    for pos in candidates:
        elapsed = time.time() - start_time
        if elapsed > time_limit - safety_margin:
            break
        
        try:
            # First, polish with scipy
            polished_pos, polished_energy = _polish_structure(pos, potential, parameter)
            
            # Then use the provided relax function
            energy, positions = relax(polished_pos, potential, parameter)
            
            if energy < best_energy:
                best_energy = energy
                best_positions = positions.copy()
        except Exception:
            try:
                energy, positions = relax(pos, potential, parameter)
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()
            except Exception:
                continue
    
    # Basin hopping from the best solution
    if best_positions is not None:
        elapsed = time.time() - start_time
        if elapsed < time_limit - safety_margin:
            best_energy, best_positions = _basin_hopping(
                best_positions, potential, parameter, rng,
                time_limit, start_time, best_energy, best_positions
            )
    
    # If we still have time, try more basin hopping from different starts
    elapsed = time.time() - start_time
    if elapsed < time_limit - safety_margin - 30:
        for i in range(3):
            elapsed = time.time() - start_time
            if elapsed > time_limit - safety_margin:
                break
            
            try:
                if i % 2 == 0:
                    pos = _build_mackay_icosahedron(size, rng)
                else:
                    pos = _build_random_compact(size, rng)
                pos = pos + 0.1 * rng.standard_normal((size, 3))
                
                polished_pos, polished_energy = _polish_structure(pos, potential, parameter)
                energy, positions = relax(polished_pos, potential, parameter)
                
                if energy < best_energy:
                    best_energy = energy
                    best_positions = positions.copy()
                
                # Basin hopping from this new best
                best_energy, best_positions = _basin_hopping(
                    best_positions, potential, parameter, rng,
                    time_limit, start_time, best_energy, best_positions
                )
            except Exception:
                continue
    
    if best_positions is None:
        # Fallback: return a simple structure
        best_positions = _build_random_compact(size, rng)
    
    return best_positions
# EVOLVE-BLOCK-END
