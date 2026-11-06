"""Morse cluster global minima search with structural optimization and basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

# Global time tracking
_GLOBAL_START_TIME = None
_GLOBAL_BUDGET = 290.0  # Safety margin before 330s kill


def _get_global_remaining_time():
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
    elapsed = time.time() - _GLOBAL_START_TIME
    return max(0.0, _GLOBAL_BUDGET - elapsed)


def _morse_energy(pos, rho):
    """Compute Morse potential energy."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, 1.0)
    x = rho * (1.0 - dists)
    x = np.clip(x, -50, 50)
    ex = np.exp(x)
    energy = np.sum(ex * (ex - 2.0)) / 2.0
    return energy


def _fix_min_dist(pos, min_dist=0.5, max_iter=200):
    """Ensure all pairs are at least min_dist apart."""
    pos = pos.copy()
    n = len(pos)
    for _ in range(max_iter):
        fixed = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pos[i] - pos[j])
                if d < min_dist:
                    if d < 1e-10:
                        dir_vec = np.array([1.0, 0.0, 0.0])
                    else:
                        dir_vec = (pos[i] - pos[j]) / d
                    shift = (min_dist - d) / 2 * 1.02
                    pos[i] += dir_vec * shift
                    pos[j] -= dir_vec * shift
                    fixed = True
        if not fixed:
            break
    return pos


def _fcc_cluster(n, spacing=1.0):
    """Generate FCC lattice fragment."""
    points = []
    a = spacing
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


def _hcp_cluster(n, spacing=1.0):
    """Generate HCP-like cluster."""
    points = []
    a = spacing
    layer_height = a * np.sqrt(2.0 / 3.0)
    for layer in range(-3, 4):
        z = layer * layer_height
        offset_x = (layer % 2) * 0.5 * a
        for i in range(-4, 5):
            for j in range(-4, 5):
                if (i + j) % 2 == 0:
                    p = np.array([i * a * 0.866 + offset_x, j * a, z])
                    points.append(p)
    points = np.array(points)
    points -= points.mean(axis=0)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)
    points = points[idx[:n]]
    return points


def _icosahedron_shell(n):
    """Generate icosahedral shell configuration."""
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
        [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ])
    verts /= np.linalg.norm(verts[0])
    
    points = []
    # Core
    points.append(np.array([0.0, 0.0, 0.0]))
    
    # First shell (icosahedron)
    for v in verts:
        points.append(v * 1.0)
    
    # Second shell (subdivided icosahedron)
    if n > 13:
        # Add midpoints of edges
        from itertools import combinations
        edge_mids = []
        for i, j in combinations(range(12), 2):
            mid = (verts[i] + verts[j]) / 2
            mid = mid / np.linalg.norm(mid)
            edge_mids.append(mid * 1.6)
        for p in edge_mids:
            if len(points) < n:
                points.append(p)
    
    # Third shell
    if n > len(points):
        radius = 2.2
        for v in verts:
            if len(points) < n:
                points.append(v * radius)
        # Add more
        for i in range(12):
            for j in range(12):
                if i != j and len(points) < n:
                    mid = (verts[i] + verts[j]) / 2
                    mid = mid / np.linalg.norm(mid)
                    points.append(mid * radius)
    
    points = np.array(points[:n])
    return points


def _random_compact(n, rng, spacing=1.0):
    """Generate random compact configuration."""
    radius = spacing * n ** (1.0 / 3.0) * 0.85
    min_dist = 0.55 * spacing
    points = []
    attempts = 0
    max_attempts = 80000
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            ok = True
            for q in points:
                if np.linalg.norm(p - q) < min_dist:
                    ok = False
                    break
            if ok:
                points.append(p)
        attempts += 1
    
    if len(points) < n:
        min_dist2 = 0.51 * spacing
        attempts = 0
        while len(points) < n and attempts < max_attempts:
            p = rng.uniform(-radius, radius, 3)
            if np.linalg.norm(p) <= radius:
                ok = True
                for q in points:
                    if np.linalg.norm(p - q) < min_dist2:
                        ok = False
                        break
                if ok:
                    points.append(p)
            attempts += 1
    
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    return np.array(points[:n])


def _fibonacci_sphere(n, radius=1.0):
    """Generate points on a sphere using Fibonacci spiral."""
    points = []
    golden_angle = np.pi * (3 - np.sqrt(5))
    for i in range(n):
        z = 1 - 2 * i / (n - 1) if n > 1 else 0
        r = np.sqrt(1 - z * z)
        theta = golden_angle * i
        p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * radius
        points.append(p)
    return np.array(points)


def _optimize_lbfgs(pos, rho, maxiter=2000):
    """Optimize using L-BFGS-B with exact gradient."""
    def fun_and_grad(x):
        pos_2d = x.reshape(-1, 3)
        e, g = morse_energy_gradient(pos_2d, rho)
        return e, g.reshape(-1)
    
    try:
        result = minimize(fun_and_grad, pos.flatten(), method='L-BFGS-B',
                          jac=True, options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10})
        positions = result.x.reshape(-1, 3)
        return result.fun, positions
    except Exception:
        return _morse_energy(pos, rho), pos


def _optimize_bfgs(pos, rho, maxiter=2000):
    """Optimize using BFGS with exact gradient."""
    def fun_and_grad(x):
        pos_2d = x.reshape(-1, 3)
        e, g = morse_energy_gradient(pos_2d, rho)
        return e, g.reshape(-1)
    
    try:
        result = minimize(fun_and_grad, pos.flatten(), method='BFGS',
                          jac=True, options={'maxiter': maxiter, 'gtol': 1e-10})
        positions = result.x.reshape(-1, 3)
        return result.fun, positions
    except Exception:
        return _morse_energy(pos, rho), pos


def _optimize_relax(pos, rho):
    """Optimize using built-in relax."""
    try:
        energy, positions = relax(pos, 'morse', rho)
        return energy, positions
    except Exception:
        return _morse_energy(pos, rho), pos


def _basin_hopping_step(pos, rho, step_size, rng):
    """Perform a single basin hopping step: perturb + optimize."""
    perturbed = pos + rng.normal(0, step_size, pos.shape)
    perturbed = _fix_min_dist(perturbed)
    
    # Try multiple optimizers
    best_e = np.inf
    best_p = perturbed
    
    e, p = _optimize_lbfgs(perturbed, rho, maxiter=1000)
    if e < best_e:
        best_e = e
        best_p = p
    
    e, p = _optimize_relax(perturbed, rho)
    if e < best_e:
        best_e = e
        best_p = p
    
    return best_e, best_p


def construct(instance):
    """Find global minimum of Morse cluster."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 6))
    
    remaining = _get_global_remaining_time()
    if remaining < 5:
        # Emergency fallback
        return _fix_min_dist(_fcc_cluster(size))
    
    rng = np.random.default_rng(size * 1000 + int(rho))
    
    best_energy = np.inf
    best_positions = None
    
    # Generate diverse initial configurations
    inits = []
    
    # Structural inits
    for init_func in [_fcc_cluster, _hcp_cluster, _icosahedron_shell]:
        try:
            init = init_func(size)
            init = init + rng.normal(0, 0.03, init.shape)
            inits.append(init)
        except Exception:
            pass
    
    # Fibonacci sphere
    try:
        init = _fibonacci_sphere(size, radius=size ** (1.0/3.0) * 0.5)
        init = init + rng.normal(0, 0.02, init.shape)
        inits.append(init)
    except Exception:
        pass
    
    # Random compact
    for _ in range(4):
        try:
            init = _random_compact(size, rng)
            inits.append(init)
        except Exception:
            pass
    
    # Optimize each initial configuration
    for i, init in enumerate(inits):
        remaining = _get_global_remaining_time()
        if remaining < 10:
            break
        
        # L-BFGS-B
        e, p = _optimize_lbfgs(init, rho, maxiter=1500)
        if e < best_energy:
            best_energy = e
            best_positions = p
        
        # Built-in relax
        remaining = _get_global_remaining_time()
        if remaining < 10:
            break
        e, p = _optimize_relax(init, rho)
        if e < best_energy:
            best_energy = e
            best_positions = p
    
    # Basin hopping from best configuration
    if best_positions is not None:
        remaining = _get_global_remaining_time()
        if remaining > 30:
            # Adaptive step size
            step_sizes = [0.3, 0.2, 0.15, 0.1, 0.05]
            current_pos = best_positions.copy()
            current_e = best_energy
            
            for step_size in step_sizes:
                remaining = _get_global_remaining_time()
                if remaining < 15:
                    break
                
                n_attempts = min(5, int(remaining / 5))
                for _ in range(n_attempts):
                    remaining = _get_global_remaining_time()
                    if remaining < 10:
                        break
                    
                    new_e, new_p = _basin_hopping_step(current_pos, rho, step_size, rng)
                    if new_e < current_e:
                        current_e = new_e
                        current_pos = new_p
                        if new_e < best_energy:
                            best_energy = new_e
                            best_positions = new_p
    
    # Final polish
    if best_positions is not None:
        remaining = _get_global_remaining_time()
        if remaining > 15:
            e, p = _optimize_lbfgs(best_positions, rho, maxiter=3000)
            if e < best_energy:
                best_energy = e
                best_positions = p
        
        remaining = _get_global_remaining_time()
        if remaining > 10:
            e, p = _optimize_bfgs(best_positions, rho, maxiter=2000)
            if e < best_energy:
                best_energy = e
                best_positions = p
        
        remaining = _get_global_remaining_time()
        if remaining > 5:
            e, p = _optimize_relax(best_positions, rho)
            if e < best_energy:
                best_energy = e
                best_positions = p
    
    # Ensure minimum distance constraint
    if best_positions is not None:
        best_positions = _fix_min_dist(best_positions)
        best_energy = _morse_energy(best_positions, rho)
    
    if best_positions is None:
        best_positions = _fix_min_dist(_fcc_cluster(size))
    
    return best_positions
# EVOLVE-BLOCK-END
