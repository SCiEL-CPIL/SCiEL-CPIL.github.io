"""Morse cluster global minima: crystalline seeds + basin hopping with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _morse_energy(pos, rho):
    """Compute Morse energy E = sum_{i<j} exp(rho(1-r_ij))(exp(rho(1-r_ij)) - 2)."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(r, np.inf)
    iu = np.triu_indices(n, k=1)
    r_ij = r[iu]
    x = rho * (1.0 - r_ij)
    e = np.exp(x) * (np.exp(x) - 2.0)
    return float(np.sum(e))


def _morse_energy_and_grad(pos, rho):
    """Compute Morse energy and gradient efficiently."""
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(r, np.inf)
    x = rho * (1.0 - r)
    ex = np.exp(x)
    # dE/dr_ij = exp(x)*(exp(x)-2) * (-rho) = -rho * ex*(ex-2)
    # dE/dpos_i = -sum_j dE/dr_ij * (pos_i - pos_j)/r_ij
    # For upper triangle only, use symmetry
    iu = np.triu_indices(n, k=1)
    r_ij = r[iu]
    x_ij = rho * (1.0 - r_ij)
    ex_ij = np.exp(x_ij)
    e_ij = ex_ij * (ex_ij - 2.0)
    de_dr = -rho * e_ij
    # de_dr has shape (n*(n-1)/2,)
    # Build gradient
    grad = np.zeros_like(pos)
    # For each pair (i,j) with i<j, contribution to grad[i] and grad[j]
    i_idx, j_idx = iu
    # de_dr * (pos_i - pos_j) / r_ij
    diff_ij = pos[i_idx] - pos[j_idx]  # shape (n_pairs, 3)
    # Avoid division by zero (shouldn't happen with valid configs)
    r_safe = np.where(r_ij > 1e-12, r_ij, 1.0)
    factor = de_dr[:, None] / r_safe[:, None]  # shape (n_pairs, 3)
    contrib = factor * diff_ij  # shape (n_pairs, 3)
    # grad[i] += contrib, grad[j] -= contrib
    np.add.at(grad, i_idx, contrib)
    np.add.at(grad, j_idx, -contrib)
    energy = float(np.sum(e_ij))
    return energy, grad


def _generate_fcc(n, spacing=1.0):
    """Generate FCC-like cluster of n atoms."""
    atoms = []
    # FCC lattice vectors
    a = spacing
    basis = [
        [0, 0, 0],
        [0, a/2, a/2],
        [a/2, 0, a/2],
        [a/2, a/2, 0],
    ]
    # Generate points in expanding shells
    max_shell = int(np.ceil(n ** (1/3))) + 2
    candidates = []
    for ix in range(-max_shell, max_shell + 1):
        for iy in range(-max_shell, max_shell + 1):
            for iz in range(-max_shell, max_shell + 1):
                for b in basis:
                    p = np.array([ix * a + b[0], iy * a + b[1], iz * a + b[2]])
                    candidates.append(p)
    candidates = np.array(candidates)
    # Center of mass
    com = candidates.mean(axis=0)
    candidates = candidates - com
    # Sort by distance from center
    dists = np.linalg.norm(candidates, axis=1)
    order = np.argsort(dists)
    selected = candidates[order[:n]]
    return selected


def _generate_bcc(n, spacing=1.0):
    """Generate BCC-like cluster of n atoms."""
    atoms = []
    a = spacing
    basis = [
        [0, 0, 0],
        [a/2, a/2, a/2],
    ]
    max_shell = int(np.ceil(n ** (1/3))) + 2
    candidates = []
    for ix in range(-max_shell, max_shell + 1):
        for iy in range(-max_shell, max_shell + 1):
            for iz in range(-max_shell, max_shell + 1):
                for b in basis:
                    p = np.array([ix * a + b[0], iy * a + b[1], iz * a + b[2]])
                    candidates.append(p)
    candidates = np.array(candidates)
    com = candidates.mean(axis=0)
    candidates = candidates - com
    dists = np.linalg.norm(candidates, axis=1)
    order = np.argsort(dists)
    selected = candidates[order[:n]]
    return selected


def _generate_icosahedral(n, spacing=1.0):
    """Generate icosahedral shell cluster (Mackay icosahedron)."""
    atoms = []
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    r = spacing * 0.5
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    # Normalize to radius r
    norms = np.linalg.norm(verts, axis=1)
    verts = verts * (r / norms[:, None])
    atoms.extend([v.copy() for v in verts])
    
    # Add center
    atoms.append(np.zeros(3))
    
    # Second shell: icosidodecahedron or just scale up
    if n > 13:
        # Add atoms on edges of icosahedron (midpoints)
        # Icosahedron edges: connect vertices that are distance 2 apart (in unit icosahedron)
        # For simplicity, add a second shell at radius 2r
        r2 = r * 2.0
        # Use icosahedral symmetry points - approximate with more vertices
        # Actually, let's just use a denser icosahedral-like set
        # Generate points on sphere with icosahedral symmetry approximation
        theta = np.linspace(0, np.pi, 10)
        phi_arr = np.linspace(0, 2*np.pi, 20)
        for t in theta:
            for p in phi_arr:
                pt = r2 * np.array([np.sin(t)*np.cos(p), np.sin(t)*np.sin(p), np.cos(t)])
                atoms.append(pt)
    
    atoms = np.array(atoms)
    if len(atoms) < n:
        # Pad with small perturbations
        rng = np.random.default_rng(42)
        while len(atoms) < n:
            pt = rng.uniform(-r*3, r*3, 3)
            atoms = np.vstack([atoms, pt])
    else:
        atoms = atoms[:n]
    
    # Center
    com = atoms.mean(axis=0)
    atoms = atoms - com
    return atoms


def _generate_random_compact(n, spacing=1.0, seed=42):
    """Generate a random compact cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0 / 3.0) * 0.8
    points = []
    min_dist = 0.5 * spacing
    max_attempts = 100000
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    if len(points) < n:
        # Fallback: place remaining atoms on a sphere
        for i in range(len(points), n):
            angle = i * 2.399963  # golden angle
            y = 1 - 2 * (i + 0.5) / n
            r = np.sqrt(1 - y*y)
            p = radius * np.array([r*np.cos(angle), r*np.sin(angle), y])
            points.append(p)
    return np.array(points)


def _local_minimize(pos, rho, max_iter=1000, tol=1e-8):
    """Local minimization using gradient descent with adaptive step."""
    n = len(pos)
    pos = pos.copy()
    energy, grad = _morse_energy_and_grad(pos, rho)
    step = 0.01
    for it in range(max_iter):
        gnorm = np.linalg.norm(grad)
        if gnorm < tol:
            break
        # Adaptive step size
        trial = pos - step * grad
        trial_energy, _ = _morse_energy_and_grad(trial, rho)
        if trial_energy < energy:
            pos = trial
            energy = trial_energy
            energy, grad = _morse_energy_and_grad(pos, rho)
            step = min(step * 1.1, 0.1)
        else:
            step *= 0.5
            if step < 1e-10:
                break
    return energy, pos


def _basin_hopping(pos, rho, n_iterations, rng, deadline):
    """Basin hopping: perturb + local minimize."""
    best_energy, best_pos = _local_minimize(pos, rho)
    for it in range(n_iterations):
        if time.time() > deadline:
            break
        # Perturbation: random displacement
        scale = 0.1 + 0.1 * np.exp(-it / 50.0)
        noise = rng.normal(0, scale, size=pos.shape)
        perturbed = pos + noise
        # Ensure minimum distance
        # Simple check: if any pair too close, skip
        valid = True
        for i in range(len(perturbed)):
            for j in range(i+1, len(perturbed)):
                if np.linalg.norm(perturbed[i] - perturbed[j]) < 0.5:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            continue
        energy, minimized = _local_minimize(perturbed, rho)
        if energy < best_energy:
            best_energy = energy
            best_pos = minimized
    return best_energy, best_pos


def construct(instance):
    """Find global minimum of Morse cluster using crystalline seeds + basin hopping."""
    n = instance["n"]
    rho = instance.get("rho", 6)
    potential = instance["potential"]
    
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 300s limit
    
    # Generate diverse starting structures
    seeds = []
    
    # FCC seed
    fcc_pos = _generate_fcc(n, spacing=1.0)
    seeds.append(fcc_pos)
    
    # BCC seed
    bcc_pos = _generate_bcc(n, spacing=1.0)
    seeds.append(bcc_pos)
    
    # Icosahedral seed
    ico_pos = _generate_icosahedral(n, spacing=1.0)
    seeds.append(ico_pos)
    
    # Random compact seeds with different seeds
    for s in [0, 1, 2, 3]:
        rand_pos = _generate_random_compact(n, spacing=1.0, seed=s)
        seeds.append(rand_pos)
    
    # For each seed: local minimize first, then basin hop
    best_energy = np.inf
    best_pos = None
    
    # First pass: local minimize all seeds
    for i, seed_pos in enumerate(seeds):
        if time.time() > deadline:
            break
        energy, minimized = _local_minimize(seed_pos, rho)
        if energy < best_energy:
            best_energy = energy
            best_pos = minimized
    
    if best_pos is None:
        # Fallback
        best_pos = _generate_random_compact(n, spacing=1.0, seed=0)
        best_energy = _morse_energy(best_pos, rho)
    
    # Second pass: basin hopping from best seed
    if time.time() < deadline - 30.0:
        rng = np.random.default_rng(n + rho * 100)
        n_iter = max(10, int((deadline - time.time()) / 2.0))
        n_iter = min(n_iter, 200)
        best_energy, best_pos = _basin_hopping(best_pos, rho, n_iter, rng, deadline)
    
    # Final local polish
    if time.time() < deadline - 5.0:
        best_energy, best_pos = _local_minimize(best_pos, rho)
    
    # Validate: ensure minimum distance >= 0.5
    # If any pair is too close, nudge apart
    for i in range(len(best_pos)):
        for j in range(i + 1, len(best_pos)):
            d = np.linalg.norm(best_pos[i] - best_pos[j])
            if d < 0.5:
                mid = (best_pos[i] + best_pos[j]) / 2
                direction = (best_pos[i] - best_pos[j]) / max(d, 1e-12)
                shift = (0.5 - d) / 2 + 0.01
                best_pos[i] += direction * shift
                best_pos[j] -= direction * shift
    
    return best_pos
# EVOLVE-BLOCK-END
