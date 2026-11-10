"""Morse cluster global minima: supercluster fragment annealing + basin hopping refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START_TIME = time.time()
_TIME_LIMIT = 315.0  # 330s kill, keep 15s safety margin


def _morse_energy_and_grad(pos, rho):
    """Compute Morse energy and gradient using vectorized operations."""
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff * diff, axis=2) + 1e-12)
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    r_masked = r[mask]
    diff_masked = diff[mask]
    
    if len(r_masked) == 0:
        return 0.0, np.zeros_like(pos)
    
    r_safe = np.where(r_masked < 1e-12, 1e-12, r_masked)
    
    exp_term = np.exp(rho * (1.0 - r_safe))
    pair_e = exp_term * (exp_term - 2.0)
    energy = np.sum(pair_e)
    
    dE_dr = exp_term * (2.0 * exp_term - 2.0) * (-rho)
    factor = dE_dr / r_safe
    
    grad = np.zeros_like(pos)
    i_idx, j_idx = np.where(mask)
    np.add.at(grad, i_idx, factor[:, np.newaxis] * diff_masked)
    np.add.at(grad, j_idx, -factor[:, np.newaxis] * diff_masked)
    
    return energy, grad


def _is_valid(pos, min_dist=0.5):
    """Check if all atoms are at least min_dist apart."""
    n = len(pos)
    if n < 2:
        return True
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    return np.min(dist) >= min_dist - 1e-6


def _relax_lbfgs(positions, rho, maxiter=100, deadline=None):
    """Relax using L-BFGS-B with Morse energy and gradient."""
    def objective_and_grad(x):
        if deadline is not None and time.time() > deadline:
            e, g = _morse_energy_and_grad(x.reshape(-1, 3), rho)
            return e, g.flatten()
        pos = x.reshape(-1, 3)
        e, g = _morse_energy_and_grad(pos, rho)
        return e, g.flatten()
    
    x0 = positions.flatten()
    try:
        result = minimize(objective_and_grad, x0, jac=True, method='L-BFGS-B',
                          options={'maxiter': maxiter, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 20})
        new_pos = result.x.reshape(-1, 3)
        if _is_valid(new_pos):
            return result.fun, new_pos
        else:
            energy, pos = relax(positions, 'morse', rho)
            return energy, pos
    except Exception:
        try:
            energy, pos = relax(positions, 'morse', rho)
            return energy, pos
        except Exception:
            energy, _ = _morse_energy_and_grad(positions, rho)
            return energy, positions


def _generate_bcc(n, spacing=1.0):
    """Generate BCC-like cluster."""
    lattice = []
    R = int(np.ceil((n / 2.0) ** (1.0 / 3.0))) + 1
    for i in range(-R, R + 1):
        for j in range(-R, R + 1):
            for k in range(-R, R + 1):
                lattice.append([i * spacing, j * spacing, k * spacing])
                lattice.append([(i + 0.5) * spacing, (j + 0.5) * spacing, (k + 0.5) * spacing])
    lattice.sort(key=lambda p: np.linalg.norm(p))
    points = np.array(lattice[:n], dtype=float)
    return points - points.mean(axis=0)


def _generate_fcc(n, spacing=1.0):
    """Generate FCC-like cluster."""
    basis = [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]
    R = int(np.ceil((n / 4.0) ** (1.0 / 3.0))) + 1
    lattice = []
    for i in range(-R, R + 1):
        for j in range(-R, R + 1):
            for k in range(-R, R + 1):
                for (dx, dy, dz) in basis:
                    lattice.append([a * spacing for a in (i + dx, j + dy, k + dz)])
    
    unique = []
    seen = set()
    for p in lattice:
        key = (round(p[0], 3), round(p[1], 3), round(p[2], 3))
        if key not in seen:
            seen.add(key)
            unique.append(p)
    
    unique.sort(key=lambda p: np.linalg.norm(p))
    points = np.array(unique[:n], dtype=float)
    return points - points.mean(axis=0)


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


def _rotation_matrix_from_quat(q):
    """Convert quaternion to 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])


def _fragment_annealing(n_target, rho, seed=42, deadline=None):
    """
    Build cluster using rigid-body fragment annealing.
    Fragments are small icosahedral/BCC motifs that translate and rotate.
    """
    rng = np.random.default_rng(seed)
    
    # Define fragment sizes
    frag_sizes = []
    remaining = n_target
    while remaining > 0:
        if remaining <= 13:
            frag_sizes.append(remaining)
            remaining = 0
        elif remaining <= 25:
            frag_sizes.append(min(13, remaining))
            remaining -= frag_sizes[-1]
        else:
            frag_sizes.append(13)
            remaining -= 13
    
    # Generate initial fragment positions
    fragments = []
    for fs in frag_sizes:
        if fs <= 4:
            if fs == 1:
                frag_pos = np.zeros((1, 3))
            elif fs == 2:
                frag_pos = np.array([[-0.5, 0, 0], [0.5, 0, 0]])
            elif fs == 3:
                frag_pos = np.array([[0, 0, 0], [1, 0, 0], [0.5, np.sqrt(3)/2, 0]])
            else:
                frag_pos = np.array([[0,0,0],[1,0,0],[0.5,np.sqrt(3)/2,0],[0.5,np.sqrt(3)/6,np.sqrt(2/3)]])
        else:
            # Icosahedral fragment
            phi = (1.0 + np.sqrt(5.0)) / 2.0
            verts = np.array([
                [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
                [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
                [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
            ])
            verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
            frag_pos = verts[:fs]
        
        # Initial position: random compact placement
        radius = n_target ** (1.0 / 3.0) * 0.3
        center = rng.uniform(-radius, radius, 3)
        q = np.array([1.0, 0.0, 0.0, 0.0])  # identity quaternion
        
        fragments.append({
            'pos': frag_pos,
            'center': center,
            'quat': q,
            'size': fs
        })
    
    # Annealing parameters
    T_start = 2.0
    T_end = 0.05
    n_iter = 500
    step_trans = 0.3
    step_rot = 0.1
    
    best_energy = np.inf
    best_positions = None
    
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        T = T_start * (T_end / T_start) ** (it / n_iter)
        
        # Perturb random fragments
        n_perturb = max(1, len(fragments) // 3)
        indices = rng.choice(len(fragments), n_perturb, replace=False)
        
        for idx in indices:
            frag = fragments[idx]
            # Translation perturbation
            frag['center'] += rng.normal(0, step_trans, 3)
            # Rotation perturbation
            dq = rng.normal(0, step_rot, 4)
            frag['quat'] = frag['quat'] + dq
            frag['quat'] = frag['quat'] / np.linalg.norm(frag['quat'])
        
        # Assemble full positions
        all_pos = []
        for frag in fragments:
            R = _rotation_matrix_from_quat(frag['quat'])
            rotated = frag['pos'] @ R.T
            all_pos.append(rotated + frag['center'])
        all_pos = np.vstack(all_pos)
        
        # Check validity
        if not _is_valid(all_pos, 0.3):
            continue
        
        # Compute energy
        energy, _ = _morse_energy_and_grad(all_pos, rho)
        
        # Metropolis accept/reject
        if energy < best_energy:
            best_energy = energy
            best_positions = all_pos.copy()
        
        # Accept based on Metropolis (simplified: always accept if improving, else probabilistic)
        # For fragment annealing, we just track best
    
    if best_positions is not None:
        return best_energy, best_positions
    
    # Fallback
    return np.inf, _generate_bcc(n_target, 1.0)


def _basin_hop(positions, rho, n_iter=20, step_size=0.3, rng=None, deadline=None):
    """
    Basin hopping: perturb, relax, accept/reject.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    n = len(positions)
    current_energy, current_pos = _relax_lbfgs(positions, rho, deadline=deadline)
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
        
        new_energy, new_pos_relaxed = _relax_lbfgs(new_pos, rho, deadline=deadline)
        
        # Check validity
        if not _is_valid(new_pos_relaxed):
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
    """Fragment annealing with basin hopping refinement."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind"))
    
    deadline = _START_TIME + _TIME_LIMIT
    
    if time.time() > deadline:
        return np.zeros((size, 3))
    
    rng = np.random.default_rng(42)
    
    candidates = []
    
    # Strategy 1: Fragment annealing
    try:
        e1, pos1 = _fragment_annealing(size, rho, seed=42, deadline=deadline)
        if e1 < np.inf:
            candidates.append((e1, pos1))
    except Exception:
        pass
    
    # Strategy 2: BCC starting point + basin hopping
    try:
        pos2 = _generate_bcc(size, 1.0)
        e2, pos2 = _basin_hop(pos2, rho, n_iter=15, step_size=0.25, rng=rng, deadline=deadline)
        if e2 < np.inf:
            candidates.append((e2, pos2))
    except Exception:
        pass
    
    # Strategy 3: FCC starting point + basin hopping
    try:
        pos3 = _generate_fcc(size, 1.0)
        e3, pos3 = _basin_hop(pos3, rho, n_iter=10, step_size=0.2, rng=rng, deadline=deadline)
        if e3 < np.inf:
            candidates.append((e3, pos3))
    except Exception:
        pass
    
    # Strategy 4: Shell starting point + basin hopping
    try:
        pos4 = _generate_shell(size, 1.0)
        e4, pos4 = _basin_hop(pos4, rho, n_iter=10, step_size=0.2, rng=rng, deadline=deadline)
        if e4 < np.inf:
            candidates.append((e4, pos4))
    except Exception:
        pass
    
    # Pick the best
    if not candidates:
        return np.zeros((size, 3))
    
    best_energy, best_pos = min(candidates, key=lambda x: x[0])
    
    # Final relaxation
    final_energy, final_pos = _relax_lbfgs(best_pos, rho, maxiter=100, deadline=deadline)
    
    if not _is_valid(final_pos):
        # Fallback to best valid candidate
        for e, p in candidates:
            if _is_valid(p):
                return p
        return best_pos
    
    return final_pos
# EVOLVE-BLOCK-END
