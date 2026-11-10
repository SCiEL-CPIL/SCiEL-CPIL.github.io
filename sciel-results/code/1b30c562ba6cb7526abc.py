"""Morse cluster global minima: Multi-start optimization with adaptive basin hopping."""
import numpy as np
import time
import sys
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global time tracking
_TOTAL_DEADLINE = 310.0
_start_time = None


def _elapsed():
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    return time.time() - _start_time


def _remaining():
    return _TOTAL_DEADLINE - _elapsed()


def _morse_energy(positions, rho):
    """Compute Morse potential energy."""
    n = positions.shape[0]
    d = squareform(pdist(positions))
    np.fill_diagonal(d, 1.0)
    r = np.where(d > 0, d, 1.0)
    x = rho * (1.0 - r)
    ex = np.exp(x)
    v = ex * (ex - 2.0)
    energy = 0.5 * np.sum(v)
    return energy


def _morse_energy_grad(positions, rho):
    """Compute Morse potential energy and gradient (vectorized)."""
    n = positions.shape[0]
    d = squareform(pdist(positions))
    np.fill_diagonal(d, 1.0)
    r = np.where(d > 0, d, 1.0)
    x = rho * (1.0 - r)
    ex = np.exp(x)
    v = ex * (ex - 2.0)
    energy = 0.5 * np.sum(v)
    
    # Vectorized gradient
    dv_dr = -rho * ex * (2.0 * ex - 1.0)
    # r_vec[i,j] = positions[i] - positions[j]
    r_vec = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    r_mat = np.where(d > 1e-10, d, 1.0)
    unit_vec = r_vec / r_mat[:, :, np.newaxis]
    grad = np.einsum('ij,ijk->ik', dv_dr, unit_vec)
    return energy, grad


def _fcc_cluster(n, spacing=1.0):
    """Generate FCC-like cluster."""
    a = spacing
    points = []
    # Use smaller grid to avoid generating too many points
    for i in range(4):
        for j in range(4):
            for k in range(4):
                p = np.array([i, j, k]) * a
                points.append(p)
                points.append(p + np.array([a/2, a/2, 0]))
                points.append(p + np.array([a/2, 0, a/2]))
                points.append(p + np.array([0, a/2, a/2]))
    points = np.array(points)
    points -= points.mean(axis=0)
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n]]
    return points


def _bcc_cluster(n, spacing=1.0):
    """Generate BCC-like cluster."""
    a = spacing
    points = []
    for i in range(4):
        for j in range(4):
            for k in range(4):
                p = np.array([i, j, k]) * a
                points.append(p)
                points.append(p + np.array([a/2, a/2, a/2]))
    points = np.array(points)
    points -= points.mean(axis=0)
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n]]
    return points


def _hcp_cluster(n, spacing=1.0):
    """Generate HCP-like cluster."""
    a = spacing
    c = a * np.sqrt(8.0/3.0)
    points = []
    for layer in range(4):
        z = layer * c
        if layer % 2 == 0:
            for i in range(4):
                for j in range(4):
                    x = i * a + (j % 2) * a/2
                    y = j * a * np.sqrt(3.0)/2.0
                    points.append(np.array([x, y, z]))
        else:
            for i in range(4):
                for j in range(4):
                    x = i * a + a/2
                    y = j * a * np.sqrt(3.0)/2.0 + a * np.sqrt(3.0)/6.0
                    points.append(np.array([x, y, z]))
    points = np.array(points)
    points -= points.mean(axis=0)
    dists = np.linalg.norm(points, axis=1)
    order = np.argsort(dists)
    points = points[order[:n]]
    return points


def _icosahedral_shell(n, spacing=1.0):
    """Generate icosahedral shell cluster."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = [
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ]
    verts = np.array(verts, dtype=float)
    verts /= np.linalg.norm(verts[0]) * spacing
    
    points = [np.zeros(3)]
    points.extend(verts)
    
    # Add second shell with proper icosahedral symmetry
    s = spacing * 2.2
    # 30 vertices of second shell (edges of icosahedron)
    for i in range(12):
        for j in range(12):
            if i != j:
                v1 = verts[i]
                v2 = verts[j]
                mid = (v1 + v2) * 0.5
                if np.linalg.norm(mid) > 0.1:
                    points.append(mid * (s / np.linalg.norm(mid)))
    
    # Deduplicate by keeping only unique points
    unique_points = [points[0]]
    for p in points[1:]:
        if all(np.linalg.norm(p - q) > 0.3 * spacing for q in unique_points):
            unique_points.append(p)
    points = unique_points
    
    while len(points) < n:
        theta = np.random.uniform(0, 2*np.pi)
        z = np.random.uniform(-1, 1)
        r = np.sqrt(1-z*z)
        p = np.array([r*np.cos(theta), r*np.sin(theta), z]) * spacing * 3.5
        points.append(p)
    
    points = np.array(points[:n])
    points -= points.mean(axis=0)
    return points


def _random_cluster(n, spacing=1.0, seed=None):
    """Generate random cluster with minimum distance constraint."""
    rng = np.random.default_rng(seed)
    radius = spacing * n ** (1.0/3.0) * 1.2
    points = []
    attempts = 0
    max_attempts = 50000
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            # Vectorized distance check
            if len(points) == 0:
                points.append(p)
            else:
                existing = np.array(points)
                dists = np.linalg.norm(existing - p, axis=1)
                if np.min(dists) >= 0.5 * spacing:
                    points.append(p)
        attempts += 1
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    points = np.array(points)
    points -= points.mean(axis=0)
    return points


def _lbfgs_refine(positions, rho, maxiter=200, ftol=1e-10, gtol=1e-8):
    """Refine positions using L-BFGS-B with vectorized gradient."""
    n = positions.shape[0]
    
    def fun_and_grad(x):
        pos = x.reshape(n, 3)
        e, g = _morse_energy_grad(pos, rho)
        return e, g
    
    try:
        result = minimize(fun_and_grad, positions.ravel(), method='L-BFGS-B',
                          jac=True, options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
        return result.x.reshape(n, 3), result.fun
    except:
        return positions, _morse_energy(positions, rho)


def _adaptive_basin_hopping(positions, rho, n_iterations=100, base_step=0.3, seed=None, time_limit=20.0):
    """Adaptive basin hopping with step size that decreases over time."""
    rng = np.random.default_rng(seed)
    n = positions.shape[0]
    current_pos = positions.copy()
    current_energy, _ = _morse_energy_grad(current_pos, rho)
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    start = time.time()
    for i in range(n_iterations):
        if _remaining() < 5.0:
            break
        if time.time() - start > time_limit:
            break
        
        # Adaptive step size: starts large, decreases
        step_scale = max(0.05, base_step * (1.0 - i / n_iterations))
        
        # Use subset perturbation for larger clusters
        if n > 50 and rng.random() < 0.3:
            n_perturb = max(3, n // 4)
            idx = rng.choice(n, size=n_perturb, replace=False)
            perturbed = current_pos.copy()
            perturbed[idx] += rng.normal(0, step_scale * 1.5, size=(n_perturb, 3))
        else:
            perturbed = current_pos + rng.normal(0, step_scale, size=current_pos.shape)
        
        # Relax
        relaxed, e = _lbfgs_refine(perturbed, rho, maxiter=80)
        
        # Metropolis accept with adaptive temperature
        temp = 0.2 * (1.0 - i / n_iterations) + 0.05
        delta = e - current_energy
        if delta < 0 or rng.random() < np.exp(-delta / temp):
            current_pos = relaxed
            current_energy = e
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_pos = current_pos.copy()
    
    return best_pos, best_energy


def _kick_basin_hopping(positions, rho, n_iterations=50, seed=None, time_limit=15.0):
    """Basin hopping with occasional large kicks to escape deep local minima."""
    rng = np.random.default_rng(seed)
    n = positions.shape[0]
    current_pos = positions.copy()
    current_energy, _ = _morse_energy_grad(current_pos, rho)
    best_pos = current_pos.copy()
    best_energy = current_energy
    
    start = time.time()
    for i in range(n_iterations):
        if _remaining() < 5.0:
            break
        if time.time() - start > time_limit:
            break
        
        # Every 5th iteration, do a large kick
        if i % 5 == 4:
            step_scale = 0.8
        else:
            step_scale = 0.25
        
        perturbed = current_pos + rng.normal(0, step_scale, size=current_pos.shape)
        
        # Relax
        relaxed, e = _lbfgs_refine(perturbed, rho, maxiter=80)
        
        # Metropolis accept
        temp = 0.3
        delta = e - current_energy
        if delta < 0 or rng.random() < np.exp(-delta / temp):
            current_pos = relaxed
            current_energy = e
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_pos = current_pos.copy()
    
    return best_pos, best_energy


# _basin_hopping removed, replaced by _adaptive_basin_hopping


def _check_valid(positions, min_dist=0.5):
    """Check if all pairs are at least min_dist apart."""
    d = pdist(positions)
    return np.min(d) >= min_dist


def construct(instance):
    """Find global minimum for Morse cluster."""
    global _start_time
    if _start_time is None:
        _start_time = time.time()
    
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    
    # Generate diverse initial structures
    seeds = []
    seeds.append(_fcc_cluster(n, spacing=1.0))
    seeds.append(_bcc_cluster(n, spacing=1.0))
    seeds.append(_hcp_cluster(n, spacing=1.0))
    seeds.append(_icosahedral_shell(n, spacing=1.0))
    for i in range(3):
        seeds.append(_random_cluster(n, spacing=1.0, seed=42+i))
    
    best_energy = np.inf
    best_pos = None
    
    # First pass: relax all seeds with time budget
    for pos in seeds:
        if _remaining() < 10.0:
            break
        try:
            relaxed, e = _lbfgs_refine(pos, rho, maxiter=150)
            if _check_valid(relaxed, 0.5) and e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()
        except:
            pass
    
    # Second pass: adaptive basin hopping on multiple starts
    if _remaining() > 20.0 and best_pos is not None:
        bh_starts = [best_pos]
        if _remaining() > 40.0:
            bh_starts.append(_fcc_cluster(n, spacing=1.0))
            bh_starts.append(_bcc_cluster(n, spacing=1.0))
            bh_starts.append(_random_cluster(n, spacing=1.0, seed=99))
        
        for start_pos in bh_starts:
            if _remaining() < 10.0:
                break
            try:
                bh_pos, bh_e = _adaptive_basin_hopping(start_pos, rho, n_iterations=60, base_step=0.35, seed=0, time_limit=15.0)
                if _check_valid(bh_pos, 0.5) and bh_e < best_energy:
                    best_energy = bh_e
                    best_pos = bh_pos.copy()
            except:
                pass
    
    # Third pass: kick basin hopping for additional exploration
    if _remaining() > 15.0 and best_pos is not None:
        try:
            kick_pos, kick_e = _kick_basin_hopping(best_pos, rho, n_iterations=40, seed=7, time_limit=12.0)
            if _check_valid(kick_pos, 0.5) and kick_e < best_energy:
                best_energy = kick_e
                best_pos = kick_pos.copy()
        except:
            pass
    
    # Final high-precision polish
    if best_pos is not None and _remaining() > 5.0:
        try:
            final_pos, final_e = _lbfgs_refine(best_pos, rho, maxiter=300, ftol=1e-12, gtol=1e-10)
            if _check_valid(final_pos, 0.5):
                best_pos = final_pos
                best_energy = final_e
        except:
            pass
    
    if best_pos is None:
        best_pos = _random_cluster(n, spacing=1.0, seed=123)
    
    return best_pos
