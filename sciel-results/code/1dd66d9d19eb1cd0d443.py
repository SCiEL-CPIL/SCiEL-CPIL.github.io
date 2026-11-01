"""Lennard-Jones 55-atom global minimum: Mackay icosahedron seed + L-BFGS relaxation."""
import numpy as np
import time

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
    e = 4.0 * (1.0 / (r6 * r6) - 1.0 / r6)
    return np.sum(e) / 2.0

def _lj_energy_grad(pos):
    """Compute total LJ energy and gradient (vectorized)."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r2 = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(r2, 1.0)
    r6 = r2 ** 3
    r12 = r6 * r6
    
    e_pair = 4.0 * (1.0 / r12 - 1.0 / r6)
    np.fill_diagonal(e_pair, 0.0)
    energy = np.sum(e_pair) / 2.0
    
    with np.errstate(divide='ignore', invalid='ignore'):
        factor = 4.0 * (-12.0 / (r2 ** 7) + 6.0 / (r2 ** 4))
    np.fill_diagonal(factor, 0.0)
    
    factor_pos = factor[:, :, np.newaxis] * pos[np.newaxis, :, :]
    grad = -np.sum(factor_pos, axis=1) + pos * np.sum(factor, axis=1, keepdims=True)
    
    return energy, grad

def _make_mackay_icosahedron(r1=1.05, r2=1.75, r3=1.95):
    """Construct a two-shell Mackay icosahedron for 55 atoms.
    1 center + 12 first shell + 30 edge-midpoints + 12 vertex-like = 55
    """
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ], dtype=float)
    
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts_unit = verts / norms
    
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
    
    edge_mids = []
    for i, j in edges:
        mid = (verts_unit[i] + verts_unit[j]) / 2.0
        norm = np.linalg.norm(mid)
        if norm > 1e-6:
            edge_mids.append(mid / norm)
    edge_mids = np.array(edge_mids)
    
    center = np.array([[0.0, 0.0, 0.0]])
    first_shell = verts_unit * r1
    second_inner = edge_mids * r2
    second_outer = verts_unit * r3
    
    all_pos = np.vstack([center, first_shell, second_inner, second_outer])
    return all_pos

def _relax_lbfgs(pos, deadline=None, maxiter=500, gtol=1e-10):
    """Relax using L-BFGS-B with deadline guard."""
    from scipy.optimize import minimize
    
    start_time = time.time()
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
            options={'maxiter': maxiter, 'gtol': gtol, 'maxls': 50}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x.reshape(-1, 3)
    except Exception:
        pass
    
    if deadline is not None and time.time() > deadline - 5.0:
        return best_pos, best_energy
    
    return best_pos, best_energy

def _polish(pos, deadline=None):
    """High-precision polish pass with tighter tolerance and more iterations."""
    from scipy.optimize import minimize
    
    start_time = time.time()
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
            options={'maxiter': 1000, 'gtol': 1e-12, 'maxls': 100}
        )
        if result.fun < best_energy:
            best_energy = result.fun
            best_pos = result.x.reshape(-1, 3)
    except Exception:
        pass
    
    if deadline is not None and time.time() > deadline - 3.0:
        return best_pos, best_energy
    
    return best_pos, best_energy

def construct():
    """Construct 55-atom LJ cluster near global minimum."""
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit
    
    # Generate multiple Mackay variants with slightly different radii
    candidates = []
    
    # Base Mackay
    base = _make_mackay_icosahedron()
    candidates.append(base)
    
    # Perturbed radii variants - broader search
    radius_combos = [
        (1.0, 1.7, 1.9), (1.1, 1.8, 2.0), (1.05, 1.72, 1.92),
        (1.08, 1.78, 1.96), (1.02, 1.68, 1.88),
        (0.95, 1.65, 1.85), (1.15, 1.85, 2.05),
        (1.0, 1.75, 1.95), (1.05, 1.8, 2.0),
        (0.98, 1.72, 1.92), (1.07, 1.76, 1.94)
    ]
    
    for r1, r2, r3 in radius_combos:
        cand = _make_mackay_icosahedron(r1, r2, r3)
        candidates.append(cand)
    
    # Add random perturbations of the base Mackay
    np.random.seed(42)
    for scale in [0.03, 0.05, 0.08, 0.1]:
        perturbed = base + np.random.randn(55, 3) * scale
        candidates.append(perturbed)
    
    # Relax each candidate, keep the best
    best_energy = float('inf')
    best_pos = base.copy()
    
    # First pass: quick relaxation of all candidates
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 30.0:
            break
        relaxed, energy = _relax_lbfgs(cand, deadline=deadline, maxiter=300, gtol=1e-8)
        if energy < best_energy:
            best_energy = energy
            best_pos = relaxed
    
    # Second pass: polish the best result
    if time.time() < deadline - 10.0:
        polished, polished_energy = _polish(best_pos, deadline=deadline)
        if polished_energy < best_energy:
            best_energy = polished_energy
            best_pos = polished
    
    # Final check: ensure minimum distance >= 0.5
    min_dist_found = float('inf')
    for i in range(55):
        for j in range(i+1, 55):
            d = np.linalg.norm(best_pos[i] - best_pos[j])
            if d < min_dist_found:
                min_dist_found = d
    
    if min_dist_found < 0.5:
        scale = 0.5 / min_dist_found * 1.01
        best_pos = best_pos * scale
    
    return best_pos
# EVOLVE-BLOCK-END
