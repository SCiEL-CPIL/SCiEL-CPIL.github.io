"""Morse cluster global minima finder - Deterministic Multi-Start Optimization.

Strategy:
1. For each instance, generate a set of diverse, structured initial configurations 
   (FCC, HCP, BCC, Icosahedral shells, Random dense).
2. Relax each configuration using L-BFGS-B with high precision.
3. Keep the lowest energy configuration found.
4. Use a strict time budget to ensure all 6 instances complete within 300s.
"""
import numpy as np
import time
import sys

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Try to import chem_clusters if available, as per task description
try:
    import chem_clusters
    CHEM_CLUSTERS_AVAILABLE = True
except ImportError:
    CHEM_CLUSTERS_AVAILABLE = False

# Global time management
_TOTAL_START_TIME = time.time()
_TOTAL_DEADLINE = _TOTAL_START_TIME + 290.0  # 10s safety margin

def _morse_energy(pos, rho):
    """Compute Morse energy for a configuration."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    upper = np.triu_indices(n, 1)
    r = dists[upper]
    val = rho * (1.0 - r)
    exp_val = np.exp(val)
    e = np.sum(exp_val * (exp_val - 2.0))
    return float(e)

def _morse_gradient(pos, rho):
    """Compute gradient of Morse energy (fully vectorized)."""
    n = len(pos)
    grad = np.zeros_like(pos)
    if n < 2:
        return grad
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    exp_val = np.exp(rho * (1.0 - dists))
    pair_force = -2.0 * rho * exp_val * (exp_val - 1.0)
    np.fill_diagonal(pair_force, 0.0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        W = pair_force / dists
    W = np.where(dists > 1e-10, W, 0.0)
    Ws = W.sum(axis=1)
    grad = Ws[:, None] * pos - W @ pos
    return grad

def _min_dist_check(pos, min_d=0.5):
    """Check minimum pairwise distance."""
    n = len(pos)
    if n < 2:
        return True
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= min_d - 1e-9

def _ensure_min_dist(pos, min_d=0.51, max_iter=200):
    """Separate atoms that are too close using a simple repulsion scheme."""
    pos = pos.copy()
    n = len(pos)
    for _ in range(max_iter):
        if not _min_dist_check(pos, min_d):
            diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff ** 2, axis=-1))
            np.fill_diagonal(dists, np.inf)
            idx_i, idx_j = np.unravel_index(np.argmin(dists), dists.shape)
            if idx_i > idx_j:
                idx_i, idx_j = idx_j, idx_i
            rij = pos[idx_i] - pos[idx_j]
            r = np.linalg.norm(rij)
            if r < 1e-10:
                rij = np.array([1.0, 0, 0])
                r = 1.0
            push = (min_d - r + 0.01) / 2.0
            pos[idx_i] += push * rij / r
            pos[idx_j] -= push * rij / r
        else:
            break
    return pos

def _fcc_points(n, spacing=1.0):
    """Generate FCC lattice points."""
    pts = []
    a = spacing
    max_r = int(np.ceil(n ** (1/3))) + 4
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                for dx, dy, dz in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    pts.append(p)
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx].copy()

def _hcp_points(n, spacing=1.0):
    """Generate HCP lattice points."""
    pts = []
    a = spacing
    c_over_a = np.sqrt(8.0 / 3.0)
    max_r = int(np.ceil(n ** (1/3))) + 4
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p1 = np.array([i, j, k * c_over_a]) * a
                p2 = np.array([i + 1/3.0, j + 2/3.0, (k + 0.5) * c_over_a]) * a
                pts.append(p1)
                pts.append(p2)
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx].copy()

def _bcc_points(n, spacing=1.0):
    """Generate BCC lattice points."""
    pts = []
    a = spacing
    max_r = int(np.ceil(n ** (1/3))) + 4
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p1 = np.array([i, j, k]) * a
                p2 = np.array([i + 0.5, j + 0.5, k + 0.5]) * a
                pts.append(p1)
                pts.append(p2)
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx].copy()

def _mckay_icosahedron(n, spacing=1.0):
    """Generate Mackay icosahedron cluster."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    pts = [np.zeros(3)]
    
    # First shell: 12 vertices
    for v in verts:
        pts.append(v.copy())
        
    # Second shell: 30 face centers
    faces = [
        [0, 4, 8], [0, 9, 4], [0, 1, 5], [0, 5, 10],
        [0, 10, 9], [0, 8, 7], [0, 7, 11], [0, 11, 9],
        [1, 6, 8], [1, 8, 4], [1, 2, 3], [1, 3, 5],
        [2, 6, 7], [2, 7, 3], [2, 3, 11], [2, 11, 6],
        [3, 7, 11], [3, 11, 5], [3, 5, 2], [4, 9, 8]
    ]
    for f in faces:
        center = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
        r = np.linalg.norm(center)
        if r > 1e-10:
            center = center / r * 1.15
            pts.append(center)
            
    # Third shell: 60 edge midpoints (approx)
    if n > 43:
        edges = [
            (0,1), (0,2), (0,3), (0,4), (0,5), (0,6), (0,7), (0,8), (0,9), (0,10), (0,11),
            (1,2), (1,3), (1,4), (1,5), (1,6), (1,7), (1,8), (1,9), (1,10), (1,11),
            (2,3), (2,4), (2,5), (2,6), (2,7), (2,8), (2,9), (2,10), (2,11),
            (3,4), (3,5), (3,6), (3,7), (3,8), (3,9), (3,10), (3,11),
            (4,5), (4,6), (4,7), (4,8), (4,9), (4,10), (4,11),
            (5,6), (5,7), (5,8), (5,9), (5,10), (5,11),
            (6,7), (6,8), (6,9), (6,10), (6,11),
            (7,8), (7,9), (7,10), (7,11),
            (8,9), (8,10), (8,11),
            (9,10), (9,11),
            (10,11)
        ]
        # Simplified: just add more vertices scaled up
        for v in verts:
            pts.append(v * 2.0)
        for f in faces:
            center = (verts[f[0]] + verts[f[1]] + verts[f[2]]) / 3.0
            pts.append(center * 2.0)
            
    while len(pts) < n:
        # Fill with random points in a sphere
        r = np.random.uniform(0, 3.0)
        theta = np.random.uniform(0, 2*np.pi)
        z = np.random.uniform(-1, 1)
        p = np.array([r*np.sqrt(1-z**2)*np.cos(theta), r*np.sqrt(1-z**2)*np.sin(theta), r*z])
        pts.append(p)
        
    pts = np.array(pts[:n])
    if len(pts) > 1:
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_d = np.min(dists)
        if min_d > 1e-10:
            pts = pts * (spacing / min_d)
            
    return pts

def _random_dense(n, rng, spacing=1.0):
    """Generate random dense packing."""
    pts = []
    radius = spacing * (n ** (1/3)) * 0.8
    min_d = 0.55 * spacing
    max_attempts = 5000
    attempts = 0
    while len(pts) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            valid = True
            for q in pts:
                if np.linalg.norm(p - q) < min_d:
                    valid = False
                    break
            if valid:
                pts.append(p)
        attempts += 1
    
    while len(pts) < n:
        p = rng.uniform(-radius, radius, 3)
        pts.append(p)
        
    return np.array(pts)

def _relax_lbfgs(pos, rho, maxiter=500):
    """Relax using L-BFGS-B."""
    n = len(pos)
    x0 = pos.flatten()

    def objective(x):
        p = x.reshape(n, 3)
        return _morse_energy(p, rho)

    def gradient(x):
        p = x.reshape(n, 3)
        return _morse_gradient(p, rho).flatten()

    try:
        result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                          options={'maxiter': maxiter, 'gtol': 1e-8, 'ftol': 1e-10, 'maxls': 50})
        if result.fun < _morse_energy(pos, rho):
            return result.x.reshape(n, 3), result.fun
    except Exception:
        pass

    return pos, _morse_energy(pos, rho)

def construct(instance):
    """Find low-energy Morse cluster configuration."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 10))
    
    # Check time
    if time.time() > _TOTAL_DEADLINE:
        # Fallback: return a simple valid configuration
        return _fcc_points(size)

    # Try to use chem_clusters if available
    if CHEM_CLUSTERS_AVAILABLE:
        try:
            # Generate a good initial guess
            pos = _fcc_points(size)
            pos = _ensure_min_dist(pos)
            relaxed_pos = chem_clusters.relax(pos, 'morse', rho)
            if relaxed_pos is not None and _min_dist_check(relaxed_pos):
                return relaxed_pos
        except Exception:
            pass

    rng = np.random.default_rng(size * 13 + int(rho) * 7 + 42)
    
    best_energy = np.inf
    best_pos = None
    
    # Generate initial configurations based on rho
    initial_configs = []
    
    if rho <= 6:
        initial_configs.append(_fcc_points(size))
        initial_configs.append(_hcp_points(size))
        initial_configs.append(_bcc_points(size))
        initial_configs.append(_random_dense(size, rng))
    else:
        initial_configs.append(_mckay_icosahedron(size))
        initial_configs.append(_fcc_points(size))
        initial_configs.append(_hcp_points(size))
        initial_configs.append(_random_dense(size, rng))
    
    # Add perturbed versions of the best structured ones
    for base in initial_configs[:2]:
        perturbed = base + rng.normal(0, 0.1, base.shape)
        initial_configs.append(perturbed)
    
    for pos in initial_configs:
        if time.time() > _TOTAL_DEADLINE - 10.0:
            break
            
        pos = _ensure_min_dist(pos)
        if not _min_dist_check(pos):
            continue
            
        try:
            relaxed_pos, energy = _relax_lbfgs(pos, rho, maxiter=1000)
            if _min_dist_check(relaxed_pos):
                if energy < best_energy:
                    best_energy = energy
                    best_pos = relaxed_pos.copy()
        except Exception:
            continue
            
    if best_pos is not None and time.time() < _TOTAL_DEADLINE - 5.0:
        # Final polish
        try:
            relaxed_pos, energy = _relax_lbfgs(best_pos, rho, maxiter=2000)
            if _min_dist_check(relaxed_pos) and energy < best_energy:
                best_pos = relaxed_pos
                best_energy = energy
        except Exception:
            pass
            
    if best_pos is None:
        best_pos = _random_dense(size, rng)
        best_pos = _ensure_min_dist(best_pos)
        
    return best_pos
