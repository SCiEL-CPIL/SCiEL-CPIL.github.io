"""Sutton-Chen cluster global minima search with basin-hopping and structured seeds."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from autoresearch_suite.chem_clusters import relax, sutton_chen_energy_gradient

# Global deadline guard
_DEADLINE = time.time() + 280  # 280s out of 330s budget, leaving 50s margin


def _check_deadline():
    return time.time() > _DEADLINE


def _sc_energy(pos, n_exp, m_exp, c):
    """Compute Sutton-Chen energy."""
    n = len(pos)
    if n < 2:
        return 0.0
    d = cdist(pos, pos)
    np.fill_diagonal(d, np.inf)
    # Avoid division by zero
    d = np.maximum(d, 1e-10)
    r_inv_n = np.sum(d ** (-n_exp), axis=1)
    r_inv_m = np.sum(d ** (-m_exp), axis=1)
    e = 0.5 * np.sum(r_inv_n) - c * np.sum(np.sqrt(r_inv_m))
    return float(e)


def _sc_gradient(pos, n_exp, m_exp, c):
    """Compute Sutton-Chen energy and gradient."""
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    d = cdist(pos, pos)
    np.fill_diagonal(d, np.inf)
    d = np.maximum(d, 1e-10)
    
    r_inv_n = np.sum(d ** (-n_exp), axis=1)
    r_inv_m = np.sum(d ** (-m_exp), axis=1)
    e = 0.5 * np.sum(r_inv_n) - c * np.sum(np.sqrt(r_inv_m))
    
    # Gradient computation
    grad = np.zeros_like(pos)
    for i in range(n):
        for j in range(i + 1, n):
            rij = pos[i] - pos[j]
            r = np.linalg.norm(rij)
            if r < 1e-10:
                continue
            # dE/dr for pair (i,j)
            r_inv_n_pair = r ** (-n_exp)
            r_inv_m_pair = r ** (-m_exp)
            
            # Sum over k!=i of r_ik^-m for atom i
            r_inv_m_i = np.sum((cdist(pos[i:i+1], pos)) ** (-m_exp))
            r_inv_m_i -= r ** (-m_exp)  # remove self
            r_inv_m_i = max(r_inv_m_i, 1e-30)
            
            r_inv_m_j = np.sum((cdist(pos[j:j+1], pos)) ** (-m_exp))
            r_inv_m_j -= r ** (-m_exp)
            r_inv_m_j = max(r_inv_m_j, 1e-30)
            
            # dE/dr = -n_exp * r^(-n_exp-1) + c * (m_exp / 2) * r^(-m_exp-1) * (1/sqrt(r_inv_m_i) + 1/sqrt(r_inv_m_j))
            dE_dr = -n_exp * r ** (-n_exp - 1) + c * (m_exp / 2) * r ** (-m_exp - 1) * (1/np.sqrt(r_inv_m_i) + 1/np.sqrt(r_inv_m_j))
            
            # Force on i: -dE/dr * (rij/r)
            f = -dE_dr * rij / r
            grad[i] += f
            grad[j] -= f
    
    return float(e), grad


def _generate_fcc_seed(n, spacing=1.0):
    """Generate FCC-like cluster seed."""
    basis = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
    max_idx = int(np.ceil(n ** (1/3))) + 3
    points = []
    for i in range(-max_idx, max_idx + 1):
        for j in range(-max_idx, max_idx + 1):
            for k in range(-max_idx, max_idx + 1):
                for b in basis:
                    p = np.array([i, j, k]) + b
                    points.append(p)
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx] * spacing


def _generate_hcp_seed(n, spacing=1.0):
    """Generate HCP-like cluster seed."""
    a = spacing
    c_hcp = a * np.sqrt(2.0 / 3.0)
    points = []
    # Layer 1: hexagonal
    for i in range(-3, 4):
        for j in range(-3, 4):
            x = a * (i + 0.5 * (j % 2))
            y = a * np.sqrt(3) / 2 * j
            z = 0
            points.append([x, y, z])
    # Layer 2: shifted
    shift = np.array([a / 2, a / (2 * np.sqrt(3)), c_hcp])
    for p in points[:]:
        points.append([p[0] + shift[0], p[1] + shift[1], p[2] + shift[2]])
    # Layer 3: same as layer 1
    for p in points[:len(points) // 2]:
        points.append([p[0], p[1], p[2] + 2 * c_hcp])
    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    return points[idx]


def _generate_icosahedral_seed(n, spacing=1.0):
    """Generate icosahedral-like cluster seed."""
    phi = (1 + np.sqrt(5)) / 2
    # Icosahedron vertices
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]) * spacing / np.sqrt(1 + phi**2)
    
    points = [np.zeros(3)]
    for v in verts:
        points.append(v)
    
    # Add second shell (octahedron positions)
    octa = np.array([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
    ]) * spacing * 0.7
    for v in octa:
        points.append(v)
    
    # Add more if needed
    rng = np.random.default_rng(42)
    while len(points) < n:
        p = rng.normal(0, spacing * 0.5, 3)
        points.append(p)
    
    points = np.array(points[:n])
    return points


def _generate_random_compact(n, spacing=1.0, seed=0):
    """Generate random compact cluster."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1/3) * 0.8
    points = []
    for _ in range(n):
        for _ in range(50):
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                points.append(p)
                break
    return np.array(points)


def _local_relax(pos, n_exp, m_exp, c, maxiter=500):
    """Local relaxation using L-BFGS-B."""
    if _check_deadline():
        return _sc_energy(pos, n_exp, m_exp, c), pos
    
    def objective(x):
        e, g = _sc_gradient(x, n_exp, m_exp, c)
        return e, g
    
    try:
        res = minimize(objective, pos, jac=True, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'gtol': 1e-6})
        if res.success:
            return float(res.fun), res.x
        else:
            e, g = _sc_gradient(res.x, n_exp, m_exp, c)
            return e, res.x
    except Exception:
        return _sc_energy(pos, n_exp, m_exp, c), pos


def _basin_hopping(pos, n_exp, m_exp, c, n_iter=50, step_size=0.3, temp_start=1.0, temp_end=0.01):
    """Basin-hopping with cooling."""
    if _check_deadline():
        return _sc_energy(pos, n_exp, m_exp, c), pos
    
    best_e, best_pos = _local_relax(pos, n_exp, m_exp, c)
    current_e, current_pos = best_e, best_pos.copy()
    rng = np.random.default_rng(int(abs(hash(tuple(map(tuple, pos[:3])))) % 10000))
    
    for i in range(n_iter):
        if _check_deadline():
            break
        temp = temp_start * (temp_end / temp_start) ** (i / max(n_iter - 1, 1))
        
        # Perturbation
        delta = rng.normal(0, step_size, size=current_pos.shape)
        trial_pos = current_pos + delta
        
        # Local relax
        trial_e, trial_pos = _local_relax(trial_pos, n_exp, m_exp, c, maxiter=300)
        
        # Metropolis accept
        if trial_e < current_e:
            current_e, current_pos = trial_e, trial_pos
            if trial_e < best_e:
                best_e, best_pos = trial_e, trial_pos.copy()
        else:
            delta_e = trial_e - current_e
            if delta_e / max(temp, 1e-10) < np.log(rng.random()):
                current_e, current_pos = trial_e, trial_pos
    
    return best_e, best_pos


def construct(instance):
    """Construct cluster configuration for Sutton-Chen potential."""
    global _DEADLINE
    _DEADLINE = time.time() + 280
    
    n = instance["n"]
    kind = instance["kind"]
    
    # Map kind to parameters
    params = {
        '12-6': (12, 6, 144.41),
        '9-6': (9, 6, 39.432),
        '10-8': (10, 8, 34.408)
    }
    n_exp, m_exp, c = params[kind]
    
    spacing = 1.0
    
    # Generate diverse seeds
    seeds = []
    seeds.append(_generate_fcc_seed(n, spacing))
    seeds.append(_generate_hcp_seed(n, spacing))
    seeds.append(_generate_icosahedral_seed(n, spacing))
    for s in range(3):
        seeds.append(_generate_random_compact(n, spacing, seed=s * 100 + n))
    
    best_e = np.inf
    best_pos = None
    
    for seed_pos in seeds:
        if _check_deadline():
            break
        
        # First local relax
        e, pos = _local_relax(seed_pos, n_exp, m_exp, c, maxiter=300)
        if e < best_e:
            best_e, best_pos = e, pos.copy()
        
        # Basin-hopping from this relaxed structure
        if not _check_deadline():
            bh_e, bh_pos = _basin_hopping(pos, n_exp, m_exp, c, 
                                           n_iter=30, step_size=0.25, 
                                           temp_start=0.5, temp_end=0.01)
            if bh_e < best_e:
                best_e, best_pos = bh_e, bh_pos.copy()
    
    # Final local relax on best
    if best_pos is not None:
        final_e, final_pos = _local_relax(best_pos, n_exp, m_exp, c, maxiter=1000)
        if final_e < best_e:
            best_e, best_pos = final_e, final_pos
    
    # Validate: check no atoms closer than 0.35
    if best_pos is not None and len(best_pos) >= 2:
        d = cdist(best_pos, best_pos)
        np.fill_diagonal(d, np.inf)
        if np.min(d) < 0.35:
            # Nudge apart
            for i in range(len(best_pos)):
                for j in range(i + 1, len(best_pos)):
                    rij = best_pos[i] - best_pos[j]
                    r = np.linalg.norm(rij)
                    if r < 0.35 and r > 1e-10:
                        push = (0.35 - r) / 2
                        best_pos[i] += push * rij / r
                        best_pos[j] -= push * rij / r
    
    if best_pos is None:
        best_pos = _generate_random_compact(n, spacing, seed=12345)
    
    return best_pos
# EVOLVE-BLOCK-END
