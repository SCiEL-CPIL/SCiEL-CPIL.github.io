"""Morse cluster global minima search with adaptive multi-start optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global time tracking across all 6 instances
_GLOBAL_START_TIME = None
_GLOBAL_BUDGET = 295.0  # Leave 5s safety margin for 300s limit


def _get_global_remaining_time():
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
    elapsed = time.time() - _GLOBAL_START_TIME
    return max(0.0, _GLOBAL_BUDGET - elapsed)


def _morse_energy(pos, rho):
    """Compute Morse potential energy: sum_{i<j} exp(rho(1-r_ij))(exp(rho(1-r_ij))-2)."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, 1.0)
    x = rho * (1.0 - dists)
    # Clip to avoid overflow
    x = np.clip(x, -50, 50)
    ex = np.exp(x)
    energy = np.sum(ex * (ex - 2.0)) / 2.0
    return energy


def _fix_min_dist(pos, min_d=0.5, max_iter=100):
    """Push apart atoms that are too close to ensure validity."""
    pos = pos.copy()
    n = len(pos)
    for _ in range(max_iter):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_d:
                    if d < 1e-10:
                        dir_vec = np.array([1.0, 0.0, 0.0])
                    else:
                        dir_vec = (pos[i] - pos[j]) / d
                    shift = (min_d - d) / 2.0 * 1.02
                    pos[i] += dir_vec * shift
                    pos[j] -= dir_vec * shift
                    moved = True
        if not moved:
            break
    return pos


def _fcc_cluster(n, spacing=1.0):
    """Generate FCC lattice fragment centered at origin."""
    a = spacing
    points = []
    # Generate FCC points in a cube around origin
    side = int(np.ceil((0.25 * n) ** (1.0/3.0))) + 3
    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                for offset in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    p = np.array([i, j, k]) * a + np.array(offset) * a
                    points.append(p)
    points = np.array(points)
    points -= points.mean(axis=0)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    return points


def _bcc_cluster(n, spacing=1.0):
    """Generate BCC lattice fragment centered at origin."""
    a = spacing
    points = []
    side = int(np.ceil((0.5 * n) ** (1.0/3.0))) + 3
    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                for offset in [(0, 0, 0), (0.5, 0.5, 0.5)]:
                    p = np.array([i, j, k]) * a + np.array(offset) * a
                    points.append(p)
    points = np.array(points)
    points -= points.mean(axis=0)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    return points


def _hcp_cluster(n, spacing=1.0):
    """Generate HCP lattice fragment centered at origin."""
    a = spacing
    c = a * np.sqrt(8.0 / 3.0)
    points = []
    side = int(np.ceil((0.3 * n) ** (1.0/3.0))) + 3
    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                # HCP: two atoms per unit cell
                p1 = np.array([i + 0.5 * (j % 2), np.sqrt(3) / 2 * j, k * c]) * a
                p2 = p1 + np.array([1.0 / 3.0, np.sqrt(3) / 3.0, c / 2.0]) * a
                points.append(p1)
                points.append(p2)
    points = np.array(points)
    points -= points.mean(axis=0)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    return points


def _icosahedron_shell(n, rng=None):
    """Generate icosahedral shell configuration (Mackay-like)."""
    if rng is None:
        rng = np.random.default_rng()
    
    phi = (1 + np.sqrt(5)) / 2
    # 12 vertices of icosahedron
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1]
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    points = [np.zeros(3)]
    
    # Shell 1: 12 atoms at radius 1
    for v in verts:
        points.append(v.copy())
    
    # Shell 2: 42 atoms at radius ~1.7 (approx)
    # We generate random points on sphere and keep those that are "close" to icosahedral symmetry
    # or just fill with random points on sphere with min distance constraint
    target_count = 55
    min_dist = 0.55
    target_r = 1.7
    
    attempts = 0
    while len(points) < target_count and len(points) < n and attempts < 5000:
        p = rng.normal(0, 1, 3)
        p = p / np.linalg.norm(p) * target_r
        if all(np.linalg.norm(p - q) >= min_dist for q in points):
            points.append(p)
        attempts += 1
        
    # Shell 3: Fill remaining
    if n > 55:
        target_r = 2.4
        attempts = 0
        while len(points) < n and attempts < 10000:
            p = rng.normal(0, 1, 3)
            p = p / np.linalg.norm(p) * target_r
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
            attempts += 1
            
    # Fallback if not enough
    while len(points) < n:
        p = rng.normal(0, 1, 3)
        p = p / np.linalg.norm(p) * 3.0
        points.append(p)
        
    return np.array(points[:n])


def _random_compact(n, spacing=1.0, rng=None):
    """Generate random compact configuration."""
    if rng is None:
        rng = np.random.default_rng()
    
    radius = spacing * n ** (1.0 / 3.0) * 0.9
    min_dist = 0.52 * spacing
    points = []
    
    attempts = 0
    max_attempts = 50000
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    
    # If not enough, relax constraint
    if len(points) < n:
        min_dist = 0.50 * spacing
        attempts = 0
        while len(points) < n and attempts < max_attempts:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                if all(np.linalg.norm(p - q) >= min_dist for q in points):
                    points.append(p)
            attempts += 1
            
    # Final fallback
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
        
    return np.array(points[:n])


def _optimize_adaptive(pos, rho, deadline):
    """
    Multi-stage optimization with time checks.
    Stage 1: relax (fast, robust)
    Stage 2: L-BFGS-B (gradient-based, good for polishing)
    Stage 3: BFGS (more aggressive, can escape shallow local mins)
    """
    pos = pos.copy()
    e = _morse_energy(pos, rho)
    
    # Check time before starting
    if time.time() > deadline:
        return e, pos
    
    # Stage 1: Built-in relax
    try:
        e1, p1 = relax(pos, 'morse', rho)
        if e1 < e:
            e = e1
            pos = p1
    except Exception:
        pass
    
    if time.time() > deadline:
        return e, pos
    
    # Stage 2: L-BFGS-B
    try:
        def fun_and_grad(x):
            pos_2d = x.reshape(-1, 3)
            e_g, g = morse_energy_gradient(pos_2d, rho)
            return e_g, g.reshape(-1)
        
        result = minimize(fun_and_grad, pos.flatten(), method='L-BFGS-B',
                         jac=True, options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
        if result.fun < e:
            e = result.fun
            pos = result.x.reshape(-1, 3)
    except Exception:
        pass
    
    if time.time() > deadline:
        return e, pos
    
    # Stage 3: BFGS
    try:
        def fun_and_grad_bfgs(x):
            pos_2d = x.reshape(-1, 3)
            e_g, g = morse_energy_gradient(pos_2d, rho)
            return e_g, g.reshape(-1)
            
        result = minimize(fun_and_grad_bfgs, pos.flatten(), method='BFGS',
                         jac=True, options={'maxiter': 500, 'gtol': 1e-10})
        if result.fun < e:
            e = result.fun
            pos = result.x.reshape(-1, 3)
    except Exception:
        pass
        
    return e, pos


def construct(instance):
    """Find global minimum of Morse cluster."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 6))
    
    # Initialize global timer
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
    
    deadline = time.time() + _get_global_remaining_time()
    
    # If we have very little time, return a quick valid guess
    if deadline - time.time() < 1.0:
        init = _fcc_cluster(size, spacing=1.0)
        return _fix_min_dist(init)
    
    rng = np.random.default_rng(hash((size, rho)) % (2**32))
    
    # Generate diverse initial configurations
    inits = []
    
    # 1. Lattice-based inits (FCC, BCC, HCP)
    for lattice_func in [_fcc_cluster, _bcc_cluster, _hcp_cluster]:
        try:
            init = lattice_func(size, spacing=1.0)
            # Add small perturbation to break symmetry
            init = init + rng.normal(0, 0.05, init.shape)
            inits.append(init)
        except Exception:
            pass
            
    # 2. Icosahedral shell inits
    for _ in range(2):
        try:
            init = _icosahedron_shell(size, rng=rng)
            init = init + rng.normal(0, 0.02, init.shape)
            inits.append(init)
        except Exception:
            pass
            
    # 3. Random compact inits
    # Estimate how many random inits we can afford
    # Assume ~5-10s per full optimization cycle
    time_per_init = 8.0
    n_random = max(2, int((deadline - time.time()) / time_per_init))
    n_random = min(n_random, 10)
    
    for _ in range(n_random):
        if time.time() > deadline - 2.0:
            break
        try:
            init = _random_compact(size, spacing=1.0, rng=rng)
            inits.append(init)
        except Exception:
            pass
            
    # 4. Perturbed lattice inits (larger perturbations)
    for _ in range(2):
        if time.time() > deadline - 2.0:
            break
        try:
            init = _fcc_cluster(size, spacing=1.0)
            init = init + rng.normal(0, 0.2, init.shape)
            inits.append(init)
        except Exception:
            pass
            
    # Run multi-start optimization
    best_energy = np.inf
    best_positions = None
    
    for i, init in enumerate(inits):
        if time.time() > deadline - 1.0:
            break
            
        e, p = _optimize_adaptive(init, rho, deadline)
        
        # Ensure min distance constraint
        p = _fix_min_dist(p, min_d=0.5)
        e = _morse_energy(p, rho)
        
        if e < best_energy:
            best_energy = e
            best_positions = p
            
    # Final polish if time permits
    if best_positions is not None and time.time() < deadline - 5.0:
        e, p = _optimize_adaptive(best_positions, rho, deadline)
        p = _fix_min_dist(p, min_d=0.5)
        e = _morse_energy(p, rho)
        if e < best_energy:
            best_energy = e
            best_positions = p
            
    if best_positions is None:
        # Fallback
        best_positions = _fcc_cluster(size, spacing=1.0)
        best_positions = _fix_min_dist(best_positions, min_d=0.5)
        
    return best_positions
# EVOLVE-BLOCK-END
