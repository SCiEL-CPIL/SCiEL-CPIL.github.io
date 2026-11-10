"""Morse cluster global minima: structured initial guesses + L-BFGS-B local relaxation."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def _morse_energy_and_grad(pos, rho):
    """Compute Morse energy and gradient. pos shape (n,3)."""
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, 1.0)  # avoid div by zero
    dist = np.maximum(dist, 1e-12)
    x = rho * (1.0 - dist)
    ex = np.exp(x)
    pair_e = ex * (ex - 2.0)
    # Gradient: dE/dr_ij * dr_ij/dpos
    # dE/dr = rho * exp(x) * (2*exp(x) - 2) = 2*rho*ex*(ex-1)
    dE_dr = 2.0 * rho * ex * (ex - 1.0)
    # dr_ij/dpos_i = -(pos_i - pos_j)/r_ij
    # Sum over j for each i
    # grad_i = -sum_j dE_dr_ij * (pos_i - pos_j)/r_ij
    # = sum_j dE_dr_ij * (pos_j - pos_i)/r_ij
    unit = diff / dist[:, :, np.newaxis]
    # dE_dr shape (n,n), unit shape (n,n,3)
    grad = -np.sum(dE_dr[:, :, np.newaxis] * unit, axis=1)
    energy = np.sum(pair_e) - np.trace(pair_e) / 2.0  # each pair counted twice
    energy = 0.5 * np.sum(pair_e)  # sum over i<j
    return energy, grad


def _enforce_min_dist(pos, min_d=0.5):
    """Nudge atoms that are too close apart."""
    n = len(pos)
    if n < 2:
        return pos
    for _ in range(20):
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dist, 1.0)
        too_close = dist < min_d
        if not np.any(too_close):
            break
        for i in range(n):
            for j in range(i + 1, n):
                if too_close[i, j]:
                    d = pos[i] - pos[j]
                    dist_ij = np.linalg.norm(d)
                    if dist_ij < 1e-10:
                        d = np.array([0.01, 0.01, 0.01])
                        dist_ij = np.linalg.norm(d)
                    d_unit = d / dist_ij
                    push = (min_d - dist_ij) * 0.5
                    pos[i] += push * d_unit
                    pos[j] -= push * d_unit
    return pos


def _fcc_cluster(n, spacing=1.0):
    """Generate n atoms in FCC-like arrangement."""
    a = spacing
    # FCC unit cell positions
    fcc_positions = [
        [0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0],
        [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [0.5, 0.5, 0.5],
    ]
    # Generate FCC lattice
    atoms = []
    n_per_cell = 4
    n_cells = (n + n_per_cell - 1) // n_per_cell
    side = int(np.ceil(n_cells ** (1.0/3.0)))
    for ix in range(side):
        for iy in range(side):
            for iz in range(side):
                for fp in fcc_positions[:4]:
                    if len(atoms) >= n:
                        break
                    pos = np.array([ix + fp[0], iy + fp[1], iz + fp[2]]) * a
                    atoms.append(pos)
                if len(atoms) >= n:
                    break
            if len(atoms) >= n:
                break
        if len(atoms) >= n:
            break
    atoms = atoms[:n]
    if len(atoms) < n:
        # Fill remaining with perturbed positions
        rng = np.random.default_rng(42)
        while len(atoms) < n:
            p = rng.uniform(-0.5, side * a + 0.5, 3)
            atoms.append(p)
    return np.array(atoms)


def _bcc_cluster(n, spacing=1.0):
    """Generate n atoms in BCC-like arrangement."""
    a = spacing
    bcc_positions = [
        [0, 0, 0], [0.5, 0.5, 0.5],
    ]
    atoms = []
    n_per_cell = 2
    n_cells = (n + n_per_cell - 1) // n_per_cell
    side = int(np.ceil(n_cells ** (1.0/3.0)))
    for ix in range(side):
        for iy in range(side):
            for iz in range(side):
                for bp in bcc_positions:
                    if len(atoms) >= n:
                        break
                    pos = np.array([ix + bp[0], iy + bp[1], iz + bp[2]]) * a
                    atoms.append(pos)
                if len(atoms) >= n:
                    break
            if len(atoms) >= n:
                break
        if len(atoms) >= n:
            break
    atoms = atoms[:n]
    if len(atoms) < n:
        rng = np.random.default_rng(42)
        while len(atoms) < n:
            p = rng.uniform(-0.5, side * a + 0.5, 3)
            atoms.append(p)
    return np.array(atoms)


def _icosahedral_cluster(n, spacing=1.0):
    """Generate n atoms in icosahedral-like arrangement."""
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
        [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
        [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
    ], dtype=float)
    ico_verts = ico_verts / np.linalg.norm(ico_verts[0]) * spacing
    atoms = [np.zeros(3)]  # center
    atoms.extend(ico_verts.tolist())  # 12 vertices
    # Add second shell if needed
    if n > 13:
        # Add atoms at edge midpoints of icosahedron
        edges = []
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(ico_verts[i] - ico_verts[j])
                if d < 1.5 * spacing:  # adjacent vertices
                    mid = (ico_verts[i] + ico_verts[j]) / 2
                    atoms.append(mid)
    # Add more shells if needed
    rng = np.random.default_rng(42)
    while len(atoms) < n:
        # Random position on sphere of radius 2*spacing
        theta = rng.uniform(0, 2*np.pi)
        z = rng.uniform(-1, 1)
        r = 2.0 * spacing
        x = r * np.sqrt(1-z*z) * np.cos(theta)
        y = r * np.sqrt(1-z*z) * np.sin(theta)
        atoms.append(np.array([x, y, z]))
    atoms = atoms[:n]
    return np.array(atoms)


def _random_compact(n, spacing=1.0, seed=None):
    """Generate random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 0.8
    points = []
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.7 * spacing for q in points):
                points.append(p)
    return np.array(points)


def construct(instance):
    """Find low-energy Morse cluster configuration."""
    start_time = time.time()
    deadline = start_time + 270.0  # 300s total, leave 30s margin
    
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 3))
    potential = instance["potential"]
    
    # Generate diverse initial guesses
    initial_guesses = []
    
    # Structured guesses
    initial_guesses.append(_fcc_cluster(n, 1.0))
    initial_guesses.append(_bcc_cluster(n, 1.0))
    initial_guesses.append(_icosahedral_cluster(n, 1.0))
    
    # Random compact guesses with different seeds
    for seed in [0, 1, 2, 3, 4, 5]:
        initial_guesses.append(_random_compact(n, 1.0, seed))
    
    best_energy = np.inf
    best_positions = None
    
    for idx, init in enumerate(initial_guesses):
        if time.time() > deadline:
            break
        
        # Ensure minimum distance
        init = _enforce_min_dist(init.copy(), 0.5)
        
        # Local minimization with L-BFGS-B
        def objective(x):
            pos = x.reshape(-1, 3)
            e, g = _morse_energy_and_grad(pos, rho)
            return e, g.flatten()
        
        x0 = init.flatten()
        
        try:
            result = minimize(
                objective, x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            final_pos = result.x.reshape(-1, 3)
            final_energy, _ = _morse_energy_and_grad(final_pos, rho)
            
            if final_energy < best_energy:
                best_energy = final_energy
                best_positions = final_pos.copy()
        except Exception:
            pass
    
    # If no valid result, return the best initial guess
    if best_positions is None:
        # Try one more time with the first guess
        init = initial_guesses[0].copy()
        init = _enforce_min_dist(init, 0.5)
        x0 = init.flatten()
        try:
            result = minimize(
                objective, x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            best_positions = result.x.reshape(-1, 3)
        except Exception:
            best_positions = init
    
    # Final check: ensure minimum distance
    best_positions = _enforce_min_dist(best_positions, 0.5)
    
    return best_positions
# EVOLVE-BLOCK-END
