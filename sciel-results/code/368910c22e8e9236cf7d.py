"""Morse cluster global minima search using incremental growth + basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START_TIME = time.time()
_TIME_LIMIT = 290.0  # 300s limit, keep 10s safety margin


def _morse_energy_and_grad(positions, rho):
    """Compute Morse energy and gradient using vectorized operations."""
    n = len(positions)
    if n < 2:
        return 0.0, np.zeros_like(positions)
    
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff * diff, axis=2) + 1e-12)
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    r_masked = r[mask]
    diff_masked = diff[mask]
    
    if len(r_masked) == 0:
        return 0.0, np.zeros_like(positions)
    
    r_safe = np.where(r_masked < 1e-12, 1e-12, r_masked)
    
    exp_term = np.exp(rho * (1.0 - r_safe))
    pair_e = exp_term * (exp_term - 2.0)
    energy = np.sum(pair_e)
    
    dE_dr = exp_term * (2.0 * exp_term - 2.0) * (-rho)
    factor = dE_dr / r_safe
    
    grad = np.zeros_like(positions)
    i_idx, j_idx = np.where(mask)
    np.add.at(grad, i_idx, factor[:, np.newaxis] * diff_masked)
    np.add.at(grad, j_idx, -factor[:, np.newaxis] * diff_masked)
    
    return energy, grad


def _relax_lbfgs(positions, rho, maxiter=100):
    """Relax using L-BFGS-B with Morse energy and gradient."""
    def objective_and_grad(x):
        pos = x.reshape(-1, 3)
        e, g = _morse_energy_and_grad(pos, rho)
        return e, g.flatten()
    
    x0 = positions.flatten()
    try:
        result = minimize(objective_and_grad, x0, jac=True, method='L-BFGS-B',
                          options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 20})
        return result.fun, result.x.reshape(-1, 3)
    except Exception:
        energy, pos = relax(positions, 'morse', rho)
        return energy, pos


def _generate_bcc(n, spacing=1.0):
    """Generate BCC-like cluster."""
    lattice = []
    for i in range(-6, 7):
        for j in range(-6, 7):
            for k in range(-6, 7):
                lattice.append([i * spacing, j * spacing, k * spacing])
                lattice.append([(i + 0.5) * spacing, (j + 0.5) * spacing, (k + 0.5) * spacing])
    lattice.sort(key=lambda p: np.linalg.norm(p))
    points = lattice[:n]
    return np.array(points, dtype=float)


def _generate_fcc(n, spacing=1.0):
    """Generate FCC-like cluster."""
    lattice = []
    for i in range(-7, 8):
        for j in range(-7, 8):
            for k in range(-7, 8):
                lattice.append([i * spacing, j * spacing, k * spacing])
                lattice.append([(i + 0.5) * spacing, (j + 0.5) * spacing, k * spacing])
                lattice.append([(i + 0.5) * spacing, j * spacing, (k + 0.5) * spacing])
                lattice.append([i * spacing, (j + 0.5) * spacing, (k + 0.5) * spacing])
    
    unique = []
    seen = set()
    for p in lattice:
        key = (round(p[0], 3), round(p[1], 3), round(p[2], 3))
        if key not in seen:
            seen.add(key)
            unique.append(p)
    
    unique.sort(key=lambda p: np.linalg.norm(p))
    points = unique[:n]
    return np.array(points, dtype=float)


def _generate_shell(n, spacing=1.0):
    """Generate spherical shell-like cluster (icosahedral-like)."""
    points = []
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    radius = spacing * n ** (1.0 / 3.0) * 0.45
    for i in range(n):
        z = 1.0 - 2.0 * (i + 0.5) / n
        radius_xy = np.sqrt(1.0 - z * z)
        theta = 2.0 * np.pi * i / golden_ratio
        x = radius_xy * np.cos(theta)
        y = radius_xy * np.sin(theta)
        points.append([x * radius, y * radius, z * radius])
    return np.array(points, dtype=float)


def _incremental_growth(n_target, rho, seed=42, max_attempts_per_atom=50):
    """
    Build cluster by incrementally adding atoms, choosing the position
    that minimizes total energy after local relaxation.
    """
    rng = np.random.default_rng(seed)
    
    # Start with a small seed cluster (3-5 atoms)
    n_seed = min(4, n_target)
    if n_seed == 3:
        # Equilateral triangle
        current_pos = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, np.sqrt(3) / 2, 0.0]
        ], dtype=float)
    else:
        # Small tetrahedron or square
        current_pos = _generate_bcc(n_seed, spacing=1.0)
    
    current_energy, current_pos = _relax_lbfgs(current_pos, rho)
    
    # Estimate radius for placement
    base_radius = n_target ** (1.0 / 3.0) * 0.6
    
    for n_current in range(n_seed, n_target):
        # We want to add one atom at a time
        best_new_energy = np.inf
        best_new_pos = None
        
        # Generate candidate positions for the new atom
        # Candidates: on a shell around existing cluster, plus some random
        n_candidates = min(max_attempts_per_atom, 20)
        
        candidate_positions = []
        
        # Shell candidates
        shell_radius = (n_current + 1) ** (1.0 / 3.0) * 0.55
        golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
        for i in range(n_candidates // 2 + 1):
            z = 1.0 - 2.0 * (i + 0.5) / (n_candidates // 2 + 1)
            radius_xy = np.sqrt(1.0 - z * z)
            theta = 2.0 * np.pi * i / golden_ratio
            x = radius_xy * np.cos(theta)
            y = radius_xy * np.sin(theta)
            candidate_positions.append(np.array([x, y, z]) * shell_radius)
        
        # Random candidates
        for i in range(n_candidates // 2):
            p = rng.uniform(-shell_radius, shell_radius, 3)
            if np.linalg.norm(p) <= shell_radius:
                candidate_positions.append(p)
        
        # Evaluate each candidate
        for cand in candidate_positions:
            trial_pos = np.vstack([current_pos, cand.reshape(1, 3)])
            
            # Check minimum distance
            if len(trial_pos) > 1:
                diff = trial_pos[:, np.newaxis, :] - trial_pos[np.newaxis, :, :]
                dist = np.sqrt(np.sum(diff * diff, axis=2))
                np.fill_diagonal(dist, np.inf)
                if np.min(dist) < 0.3:
                    continue
            
            trial_energy, trial_pos_relaxed = _relax_lbfgs(trial_pos, rho, maxiter=50)
            
            if trial_energy < best_new_energy:
                best_new_energy = trial_energy
                best_new_pos = trial_pos_relaxed
        
        # If no valid candidate found, use a random one
        if best_new_pos is None:
            new_atom = rng.uniform(-shell_radius, shell_radius, 3)
            trial_pos = np.vstack([current_pos, new_atom.reshape(1, 3)])
            best_new_energy, best_new_pos = _relax_lbfgs(trial_pos, rho, maxiter=50)
        
        current_pos = best_new_pos
        current_energy = best_new_energy
    
    return current_energy, current_pos


def _basin_hop(positions, rho, n_iter=20, step_size=0.3, rng=None, deadline=None):
    """
    Basin hopping: perturb, relax, accept/reject.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    n = len(positions)
    current_energy, current_pos = _relax_lbfgs(positions, rho)
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    T = 1.0
    
    for i in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Perturb a subset of atoms
        n_perturb = max(1, n // 8)
        indices = rng.choice(n, n_perturb, replace=False)
        
        new_pos = current_pos.copy()
        new_pos[indices] += rng.normal(0, step_size, (n_perturb, 3))
        
        new_energy, new_pos_relaxed = _relax_lbfgs(new_pos, rho)
        
        # Check validity
        if len(new_pos_relaxed) > 1:
            diff = new_pos_relaxed[:, np.newaxis, :] - new_pos_relaxed[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff * diff, axis=2))
            np.fill_diagonal(dist, np.inf)
            if np.min(dist) < 0.5 - 1e-6:
                continue
        
        # Metropolis criterion
        delta_E = new_energy - current_energy
        if delta_E < 0 or rng.random() < np.exp(-delta_E / max(T, 1e-9)):
            current_pos = new_pos_relaxed
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
        
        T *= 0.95
        if T < 0.05:
            break
    
    return best_energy, best_pos


def construct(instance):
    """Incremental growth with basin hopping refinement."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind"))
    
    deadline = _START_TIME + _TIME_LIMIT - 10
    
    if time.time() > deadline:
        return np.zeros((size, 3))
    
    rng = np.random.default_rng(42)
    
    # Strategy 1: Incremental growth
    try:
        e1, pos1 = _incremental_growth(size, rho, seed=42, max_attempts_per_atom=15)
    except Exception:
        e1 = np.inf
        pos1 = _generate_bcc(size, 1.0)
    
    # Strategy 2: Crystalline starting point + basin hopping
    try:
        pos2 = _generate_bcc(size, 1.0)
        e2, pos2 = _basin_hop(pos2, rho, n_iter=15, step_size=0.25, rng=rng, deadline=deadline)
    except Exception:
        e2 = np.inf
        pos2 = pos1.copy()
    
    # Strategy 3: FCC starting point + basin hopping
    try:
        pos3 = _generate_fcc(size, 1.0)
        e3, pos3 = _basin_hop(pos3, rho, n_iter=10, step_size=0.2, rng=rng, deadline=deadline)
    except Exception:
        e3 = np.inf
        pos3 = pos1.copy()
    
    # Strategy 4: Shell starting point + basin hopping
    try:
        pos4 = _generate_shell(size, 1.0)
        e4, pos4 = _basin_hop(pos4, rho, n_iter=10, step_size=0.2, rng=rng, deadline=deadline)
    except Exception:
        e4 = np.inf
        pos4 = pos1.copy()
    
    # Pick the best
    candidates = [(e1, pos1), (e2, pos2), (e3, pos3), (e4, pos4)]
    candidates = [(e, p) for e, p in candidates if e < np.inf]
    
    if not candidates:
        return np.zeros((size, 3))
    
    best_energy, best_pos = min(candidates, key=lambda x: x[0])
    
    # Final relaxation
    final_energy, final_pos = _relax_lbfgs(best_pos, rho, maxiter=100)
    
    return final_pos
# EVOLVE-BLOCK-END
