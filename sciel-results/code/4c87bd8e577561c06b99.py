"""Morse cluster global minima search with multi-start optimization."""
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


def _min_dist_check(pos, min_d=0.5):
    """Check if all pairs are at least min_d apart."""
    n = len(pos)
    if n < 2:
        return True
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists) >= min_d


def _fix_min_dist(pos, min_d=0.5, max_iter=50):
    """Push apart atoms that are too close."""
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
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
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
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
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
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-3, 4):
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


def _icosahedron_shell(n, shell_radii=None, rng=None):
    """Generate icosahedral shell configuration."""
    if rng is None:
        rng = np.random.default_rng()
    
    if shell_radii is None:
        # Default shell radii based on n
        if n <= 13:
            shell_radii = [0, 1.0]
        elif n <= 55:
            shell_radii = [0, 1.0, 1.7]
        elif n <= 147:
            shell_radii = [0, 1.0, 1.7, 2.4]
        else:
            shell_radii = [0, 1.0, 1.7, 2.4, 3.1]
    
    points = []
    # Center atom
    points.append(np.zeros(3))
    
    # Generate points on shells using icosahedral directions
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    ico_vertices = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0]
    ])
    # Normalize
    ico_vertices = ico_vertices / np.linalg.norm(ico_vertices, axis=1, keepdims=True)
    
    # Generate more directions by rotating
    all_dirs = [ico_vertices.copy()]
    for angle in [np.pi / 5, 2 * np.pi / 5, np.pi / 3, 2 * np.pi / 3]:
        for axis in [[0, 0, 1], [0, 1, 0], [1, 0, 0]]:
            axis = np.array(axis, dtype=float)
            axis = axis / np.linalg.norm(axis)
            # Rotation matrix
            K = np.array([[0, -axis[2], axis[1]],
                          [axis[2], 0, -axis[0]],
                          [-axis[1], axis[0], 0]])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            rotated = ico_vertices @ R.T
            all_dirs.append(rotated)
    
    all_dirs = np.vstack(all_dirs)
    # Normalize and remove duplicates
    all_dirs = all_dirs / np.linalg.norm(all_dirs, axis=1, keepdims=True)
    
    # Place atoms on shells
    for shell_idx, radius in enumerate(shell_radii):
        if radius == 0:
            continue
        # Number of atoms on this shell
        if shell_idx == 0:
            n_shell = 12
        else:
            n_shell = min(30, n - len(points))
        
        for i in range(n_shell):
            if len(points) >= n:
                break
            # Pick a direction
            d = all_dirs[i % len(all_dirs)]
            # Add small perturbation
            d = d + rng.normal(0, 0.01, 3)
            d = d / np.linalg.norm(d)
            points.append(d * radius)
    
    # If we need more atoms, add on outermost shell
    while len(points) < n:
        d = all_dirs[len(points) % len(all_dirs)] + rng.normal(0, 0.05, 3)
        d = d / np.linalg.norm(d)
        r = shell_radii[-1] + rng.uniform(0, 0.3)
        points.append(d * r)
    
    return np.array(points[:n])


def _random_compact(n, spacing=1.0, rng=None):
    """Generate random compact configuration."""
    if rng is None:
        rng = np.random.default_rng()
    
    radius = spacing * n ** (1.0 / 3.0) * 0.85
    min_dist = 0.55 * spacing
    points = []
    
    attempts = 0
    max_attempts = 30000
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_dist for q in points):
                points.append(p)
        attempts += 1
    
    # If not enough, relax constraint
    if len(points) < n:
        min_dist = 0.51 * spacing
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


def _optimize_relax(pos, rho):
    """Optimize using built-in relax."""
    try:
        energy, positions = relax(pos, 'morse', rho)
        return energy, positions
    except Exception:
        return _morse_energy(pos, rho), pos


def _optimize_lbfgs(pos, rho, maxiter=2000):
    """Optimize using L-BFGS-B with gradient."""
    try:
        def fun_and_grad(x):
            pos_2d = x.reshape(-1, 3)
            e, g = morse_energy_gradient(pos_2d, rho)
            return e, g.reshape(-1)
        
        result = minimize(fun_and_grad, pos.flatten(), method='L-BFGS-B',
                         jac=True, options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50})
        positions = result.x.reshape(-1, 3)
        return result.fun, positions
    except Exception:
        return _morse_energy(pos, rho), pos


def _optimize_bfgs(pos, rho, maxiter=2000):
    """Optimize using BFGS with gradient."""
    try:
        def fun_and_grad(x):
            pos_2d = x.reshape(-1, 3)
            e, g = morse_energy_gradient(pos_2d, rho)
            return e, g.reshape(-1)
        
        result = minimize(fun_and_grad, pos.flatten(), method='BFGS',
                         jac=True, options={'maxiter': maxiter, 'gtol': 1e-12, 'maxls': 50})
        positions = result.x.reshape(-1, 3)
        return result.fun, positions
    except Exception:
        return _morse_energy(pos, rho), pos


def _multi_start_optimize(inits, rho, deadline):
    """Run multi-start optimization over multiple initial configurations."""
    best_energy = np.inf
    best_positions = None
    
    for i, init in enumerate(inits):
        remaining = _get_global_remaining_time()
        if remaining < 5.0:
            break
        
        # Try relax first (fast)
        e1, p1 = _optimize_relax(init, rho)
        if e1 < best_energy:
            best_energy = e1
            best_positions = p1
        
        remaining = _get_global_remaining_time()
        if remaining < 10.0:
            continue
        
        # Try L-BFGS-B
        e2, p2 = _optimize_lbfgs(init, rho, maxiter=1500)
        if e2 < best_energy:
            best_energy = e2
            best_positions = p2
        
        remaining = _get_global_remaining_time()
        if remaining < 15.0:
            continue
        
        # Try BFGS from the best of the two
        start_for_bfgs = p2 if e2 < e1 else p1
        e3, p3 = _optimize_bfgs(start_for_bfgs, rho, maxiter=1500)
        if e3 < best_energy:
            best_energy = e3
            best_positions = p3
    
    return best_energy, best_positions


def construct(instance):
    """Find global minimum of Morse cluster."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 6))
    
    # Initialize global timer
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
    
    remaining = _get_global_remaining_time()
    if remaining < 10.0:
        # Very little time left, return quick answer
        init = _fcc_cluster(size, spacing=1.0)
        return _fix_min_dist(init)
    
    rng = np.random.default_rng(hash((size, rho)) % (2**32))
    
    # Generate diverse initial configurations
    inits = []
    
    # Lattice-based inits
    for lattice_func in [_fcc_cluster, _bcc_cluster, _hcp_cluster]:
        try:
            init = lattice_func(size, spacing=1.0)
            # Add small perturbation to break symmetry
            init = init + rng.normal(0, 0.03, init.shape)
            inits.append(init)
        except Exception:
            pass
    
    # Icosahedral shell inits
    for _ in range(3):
        try:
            init = _icosahedron_shell(size, rng=rng)
            init = init + rng.normal(0, 0.02, init.shape)
            inits.append(init)
        except Exception:
            pass
    
    # Random compact inits
    n_random = max(4, int(remaining / 30))
    n_random = min(n_random, 12)
    for _ in range(n_random):
        try:
            init = _random_compact(size, spacing=1.0, rng=rng)
            inits.append(init)
        except Exception:
            pass
    
    # Perturbed lattice inits
    for _ in range(3):
        try:
            init = _fcc_cluster(size, spacing=1.0)
            init = init + rng.normal(0, 0.15, init.shape)
            inits.append(init)
        except Exception:
            pass
    
    # Run multi-start optimization
    best_energy, best_positions = _multi_start_optimize(inits, rho, deadline=_get_global_remaining_time())
    
    # Final polish if time permits
    if best_positions is not None and _get_global_remaining_time() > 15.0:
        # Re-optimize the best with more iterations
        e, p = _optimize_lbfgs(best_positions, rho, maxiter=3000)
        if e < best_energy:
            best_energy = e
            best_positions = p
        
        if _get_global_remaining_time() > 10.0:
            e, p = _optimize_bfgs(best_positions, rho, maxiter=3000)
            if e < best_energy:
                best_energy = e
                best_positions = p
    
    # Ensure minimum distance constraint
    if best_positions is not None:
        best_positions = _fix_min_dist(best_positions, min_d=0.5)
        best_energy = _morse_energy(best_positions, rho)
    
    if best_positions is None:
        # Fallback
        best_positions = _fcc_cluster(size, spacing=1.0)
        best_positions = _fix_min_dist(best_positions, min_d=0.5)
    
    return best_positions
# EVOLVE-BLOCK-END
