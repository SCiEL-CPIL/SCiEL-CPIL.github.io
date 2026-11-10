"""Morse cluster global minima: basin hopping with crystalline starts and deadline guard."""
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
    """Compute Morse energy and gradient."""
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(r, np.inf)
    iu = np.triu_indices(n, k=1)
    r_ij = r[iu]
    x = rho * (1.0 - r_ij)
    exp_x = np.exp(x)
    e = exp_x * (exp_x - 2.0)
    energy = float(np.sum(e))
    
    # Gradient: dE/dr_ij = rho * exp_x * (2*exp_x - 1)
    # grad_i = sum_j dE/dr_ij * (pos_i - pos_j)/r_ij
    grad = np.zeros_like(pos)
    de_dr = rho * exp_x * (2.0 * exp_x - 1.0)
    # de_dr is shape (n*(n-1)/2,)
    # For each pair (i,j) with i<j, contribution to grad[i] is de_dr * (pos[i]-pos[j])/r_ij
    # and to grad[j] is -de_dr * (pos[i]-pos[j])/r_ij
    for idx in range(len(iu[0])):
        i, j = iu[0][idx], iu[1][idx]
        rij = r_ij[idx]
        if rij < 1e-10:
            continue
        vec = (pos[i] - pos[j]) / rij
        coeff = de_dr[idx]
        grad[i] += coeff * vec
        grad[j] -= coeff * vec
    return energy, grad


def _is_valid(pos, min_dist=0.5):
    """Check if all pairs are at least min_dist apart."""
    n = len(pos)
    if n < 2:
        return True
    diff = pos[:, None, :] - pos[None, :, :]
    r = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(r, np.inf)
    return np.min(r) >= min_dist


def _generate_fcc(n, spacing=1.0):
    """Generate an FCC-like cluster of n atoms."""
    # FCC lattice vectors
    a = spacing
    lattice = [
        [0, 0, 0],
        [0, a/2, a/2],
        [a/2, 0, a/2],
        [a/2, a/2, 0],
    ]
    # Generate points in FCC lattice
    points = []
    # We'll generate a cubic block of FCC points and take the first n closest to origin
    max_idx = int(np.ceil(n ** (1/3))) + 2
    candidates = []
    for i in range(-max_idx, max_idx+1):
        for j in range(-max_idx, max_idx+1):
            for k in range(-max_idx, max_idx+1):
                for vec in lattice:
                    p = np.array([i*a, j*a, k*a]) + np.array(vec)
                    candidates.append(p)
    candidates = np.array(candidates)
    # Sort by distance from origin
    dists = np.linalg.norm(candidates, axis=1)
    order = np.argsort(dists)
    selected = candidates[order[:n]]
    # Center the cluster
    selected = selected - np.mean(selected, axis=0)
    return selected


def _generate_bcc(n, spacing=1.0):
    """Generate a BCC-like cluster of n atoms."""
    a = spacing
    lattice = [
        [0, 0, 0],
        [a/2, a/2, a/2],
    ]
    max_idx = int(np.ceil(n ** (1/3))) + 2
    candidates = []
    for i in range(-max_idx, max_idx+1):
        for j in range(-max_idx, max_idx+1):
            for k in range(-max_idx, max_idx+1):
                for vec in lattice:
                    p = np.array([i*a, j*a, k*a]) + np.array(vec)
                    candidates.append(p)
    candidates = np.array(candidates)
    dists = np.linalg.norm(candidates, axis=1)
    order = np.argsort(dists)
    selected = candidates[order[:n]]
    selected = selected - np.mean(selected, axis=0)
    return selected


def _basin_hopping_step(pos, rho, rng, step_scale=0.3):
    """Perform one basin hopping step: perturb, relax, return new position and energy."""
    # Random perturbation
    noise = rng.normal(0, step_scale, pos.shape)
    perturbed = pos + noise
    # Relax
    energy, relaxed = relax(perturbed, 'morse', rho)
    return energy, relaxed


def construct(instance):
    """Find low-energy Morse cluster configuration using basin hopping with crystalline starts."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    start_time = time.time()
    deadline = start_time + 280.0  # Safety margin before 330s limit
    
    rng = np.random.default_rng(n)
    
    # Generate initial configurations from different crystal structures
    best_energy = np.inf
    best_pos = None
    
    # Try FCC and BCC starts
    for gen_func in [_generate_fcc, _generate_bcc]:
        pos = gen_func(n, spacing=1.0)
        # Validate and relax
        energy, relaxed = relax(pos, 'morse', rho)
        if _is_valid(relaxed):
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed.copy()
    
    # If no valid start, use random compact
    if best_pos is None:
        radius = 1.0 * n ** (1.0/3.0) * 0.75
        points = []
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 for q in points):
                points.append(p)
        energy, best_pos = relax(np.array(points), 'morse', rho)
        best_energy = energy
    
    # Basin hopping
    step_scale = 0.2
    while time.time() < deadline:
        # Perturb and relax
        noise = rng.normal(0, step_scale, best_pos.shape)
        perturbed = best_pos + noise
        energy, relaxed = relax(perturbed, 'morse', rho)
        
        if _is_valid(relaxed):
            if energy < best_energy:
                best_energy = energy
                best_pos = relaxed.copy()
                # Occasionally reduce step scale on improvement
                step_scale = max(0.1, step_scale * 0.95)
            else:
                # Slightly increase step scale to escape local minima
                step_scale = min(0.5, step_scale * 1.02)
    
    # Final check
    if best_pos is None:
        # Fallback: return a valid compact configuration
        radius = 1.0 * n ** (1.0/3.0) * 0.75
        points = []
        while len(points) < n:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius and all(np.linalg.norm(p - q) >= 0.8 for q in points):
                points.append(p)
        _, best_pos = relax(np.array(points), 'morse', rho)
    
    return best_pos
# EVOLVE-BLOCK-END
