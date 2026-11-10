"""Morse cluster global minima: diverse crystalline seeds + multi-scale basin hopping."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START_TIME = time.time()
_TIME_LIMIT = 320.0  # 330s kill, keep 10s safety margin


def _morse_energy_and_grad(pos, rho):
    """Compute Morse energy and gradient using vectorized operations."""
    n = len(pos)
    if n < 2:
        return 0.0, np.zeros_like(pos)
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff * diff, axis=2) + 1e-15)
    
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    r_masked = r[mask]
    diff_masked = diff[mask]
    
    if len(r_masked) == 0:
        return 0.0, np.zeros_like(pos)
    
    r_safe = np.where(r_masked < 1e-12, 1e-12, r_masked)
    
    exp_term = np.exp(rho * (1.0 - r_safe))
    pair_e = exp_term * (exp_term - 2.0)
    energy = float(np.sum(pair_e))
    
    dE_dr = exp_term * (2.0 * exp_term - 2.0) * (-rho)
    factor = dE_dr / r_safe
    
    grad = np.zeros_like(pos)
    i_idx, j_idx = np.where(mask)
    np.add.at(grad, i_idx, factor[:, np.newaxis] * diff_masked)
    np.add.at(grad, j_idx, -factor[:, np.newaxis] * diff_masked)
    
    return energy, grad


def _is_valid(pos, min_dist=0.5):
    """Check if all pairwise distances are at least min_dist."""
    n = len(pos)
    if n < 2:
        return True
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(r, np.inf)
    return float(np.min(r)) >= min_dist - 1e-9


def _local_minimize(pos, rho, deadline, maxiter=200, ftol=1e-10):
    """Local minimization using L-BFGS-B with deadline guard."""
    pos = pos.astype(np.float64)
    
    def fun_g(x):
        if time.time() > deadline:
            return 1e30, np.zeros_like(x)
        e, g = _morse_energy_and_grad(x.reshape(-1, 3), rho)
        return e, g.flatten()
    
    try:
        res = minimize(fun_g, pos.ravel(), method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': ftol, 'gtol': 1e-8, 'maxls': 20})
        if res.fun < 1e29:
            newpos = res.x.reshape(-1, 3)
            if _is_valid(newpos):
                return newpos
    except Exception:
        pass
    
    # Fallback to provided relax
    try:
        e, newpos = relax(pos, 'morse', rho)
        if _is_valid(newpos):
            return newpos
    except Exception:
        pass
    
    return pos


def _fcc_cluster(n, a=1.0):
    """Generate n points on an FCC lattice, centered."""
    basis = [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]
    R = int(np.ceil((n / 4.0) ** (1.0 / 3.0))) + 1
    coords = []
    for i in range(-R, R + 1):
        for j in range(-R, R + 1):
            for k in range(-R, R + 1):
                for (dx, dy, dz) in basis:
                    coords.append((a * (i + dx), a * (j + dy), a * (k + dz)))
    coords = np.array(coords)
    d = np.linalg.norm(coords, axis=1)
    order = np.argsort(d)
    pts = coords[order[:n]]
    pts = pts - pts.mean(axis=0)
    return pts


def _bcc_cluster(n, a=1.0):
    """Generate n points on a BCC lattice, centered."""
    basis = [(0, 0, 0), (0.5, 0.5, 0.5)]
    R = int(np.ceil((n / 2.0) ** (1.0 / 3.0))) + 1
    coords = []
    for i in range(-R, R + 1):
        for j in range(-R, R + 1):
            for k in range(-R, R + 1):
                for (dx, dy, dz) in basis:
                    coords.append((a * (i + dx), a * (j + dy), a * (k + dz)))
    coords = np.array(coords)
    d = np.linalg.norm(coords, axis=1)
    order = np.argsort(d)
    pts = coords[order[:n]]
    pts = pts - pts.mean(axis=0)
    return pts


def _hcp_cluster(n, a=1.0):
    """Generate n points on an HCP-like lattice, centered."""
    # HCP: two interpenetrating triangular lattices
    pts = []
    a = 1.0
    c = np.sqrt(8.0 / 3.0) * a  # ideal HCP c/a ratio
    layers = int(np.ceil((n / 6.0) ** (1.0 / 3.0))) + 2
    for layer in range(-layers, layers + 1):
        z = layer * c
        offset = (layer % 2) * 0.5
        for i in range(-layers, layers + 1):
            for j in range(-layers, layers + 1):
                x = (i + offset) * a
                y = j * a * np.sqrt(3.0) / 2.0
                pts.append([x, y, z])
    pts = np.array(pts)
    d = np.linalg.norm(pts, axis=1)
    order = np.argsort(d)
    pts = pts[order[:n]]
    pts = pts - pts.mean(axis=0)
    return pts


def _icosahedral_cluster(n):
    """Icosahedral shell construction (Mackay-like), centered."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ])
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)
    
    pts = [np.zeros(3)]  # center
    shell_radii = [1.0, np.sqrt(2.0 + 2.0 / phi), 2.0, np.sqrt(5.0 + 2.0 * phi)]
    
    # Shell 1: 12 vertices
    for v in verts:
        pts.append(shell_radii[0] * v)
    if len(pts) >= n:
        return np.array(pts[:n])
    
    # Shell 2: 30 points - use permutations of (0, 1, phi) type directions
    shell2_dirs = []
    base_dirs = [(0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi),
                 (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
                 (phi, 0, 1), (-phi, 0, 1), (phi, 0, -1), (-phi, 0, -1),
                 (1, 0, phi), (-1, 0, phi), (1, 0, -phi), (-1, 0, -phi),
                 (phi, 1, 0), (-phi, 1, 0), (phi, -1, 0), (-phi, -1, 0)]
    for d in base_dirs:
        v = np.array(d, dtype=float)
        v = v / np.linalg.norm(v)
        shell2_dirs.append(v)
    # Add more by symmetry operations
    for v in shell2_dirs[:]:
        for perm_idx in range(3):
            perm = np.roll(v, perm_idx)
            shell2_dirs.append(perm / np.linalg.norm(perm))
    
    # Take up to 30 unique directions
    seen = set()
    shell2_pts = []
    for v in shell2_dirs:
        key = (round(v[0], 3), round(v[1], 3), round(v[2], 3))
        if key not in seen:
            seen.add(key)
            shell2_pts.append(shell_radii[1] * v)
            if len(shell2_pts) == 30:
                break
    pts.extend(shell2_pts)
    if len(pts) >= n:
        return np.array(pts[:n])
    
    # Shell 3: 20 points - face centers
    shell3_pts = []
    for i in range(12):
        for j in range(i + 1, 12):
            for k in range(j + 1, 12):
                p = verts[i] + verts[j] + verts[k]
                norm = np.linalg.norm(p)
                if norm > 1e-6:
                    p = p / norm
                    shell3_pts.append(shell_radii[2] * p)
                    if len(shell3_pts) == 20:
                        break
            if len(shell3_pts) == 20:
                break
        if len(shell3_pts) == 20:
            break
    pts.extend(shell3_pts)
    
    if len(pts) >= n:
        return np.array(pts[:n])
    
    # Pad with random near outer shell
    rng = np.random.default_rng(42)
    while len(pts) < n:
        d = rng.normal(size=3)
        d = d / np.linalg.norm(d)
        pts.append(shell_radii[3] * d * rng.uniform(0.95, 1.05))
    
    pts = np.array(pts[:n])
    return pts - pts.mean(axis=0)


def _spherical_shell(n):
    """Fibonacci sphere, centered, radius chosen for packing."""
    if n <= 1:
        return np.zeros((n, 3))
    radius = 0.5 * n ** (1.0 / 3.0) + 0.3
    pts = [np.zeros(3)]
    m = n - 1
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    for i in range(m):
        z = 1.0 - 2.0 * (i + 0.5) / m
        r = np.sqrt(max(0.0, 1.0 - z * z))
        theta = 2.0 * np.pi * i / golden
        pts.append(radius * np.array([r * np.cos(theta), r * np.sin(theta), z]))
    pts = np.array(pts)
    return pts - pts.mean(axis=0)


def _random_compact(n, seed=0):
    """Random compact cluster with minimum distance enforcement."""
    rng = np.random.default_rng(seed)
    radius = 0.8 * n ** (1.0 / 3.0)
    points = []
    min_dist = 0.55
    tries = 0
    while len(points) < n and tries < 50000:
        tries += 1
        p = rng.uniform(-radius, radius, 3)
        if len(points) == 0:
            points.append(p)
            continue
        pts = np.array(points)
        dists = np.linalg.norm(pts - p, axis=1)
        if np.min(dists) >= min_dist:
            points.append(p)
    # If we couldn't place enough, relax the constraint
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    pts = np.array(points)
    return pts - pts.mean(axis=0)


def _basin_hop(positions, rho, n_iter, step_size, rng, deadline):
    """
    Basin hopping with adaptive step size and multi-scale perturbations.
    """
    n = len(positions)
    current_energy, current_pos = _morse_energy_and_grad(positions, rho)
    current_pos = _local_minimize(current_pos, rho, deadline, maxiter=100)
    current_energy, _ = _morse_energy_and_grad(current_pos, rho)
    
    best_energy = current_energy
    best_pos = current_pos.copy()
    
    T = 2.0
    
    for i in range(n_iter):
        if time.time() > deadline:
            break
        
        # Adaptive step size: start large, decrease
        scale = step_size * (1.0 + 0.3 * np.cos(2.0 * np.pi * i / max(n_iter, 1)))
        
        # Multi-scale: perturb different numbers of atoms
        if i % 3 == 0:
            n_perturb = max(1, n // 10)
        elif i % 3 == 1:
            n_perturb = max(1, n // 5)
        else:
            n_perturb = max(1, n // 3)
        
        indices = rng.choice(n, n_perturb, replace=False)
        
        new_pos = current_pos.copy()
        new_pos[indices] += rng.normal(0, scale, (n_perturb, 3))
        
        # Relax
        new_pos_relaxed = _local_minimize(new_pos, rho, deadline, maxiter=80)
        
        if not _is_valid(new_pos_relaxed):
            continue
        
        new_energy, _ = _morse_energy_and_grad(new_pos_relaxed, rho)
        
        # Metropolis criterion
        delta_E = new_energy - current_energy
        if delta_E < 0 or rng.random() < np.exp(-delta_E / max(T, 1e-9)):
            current_pos = new_pos_relaxed
            current_energy = new_energy
            
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
        
        T *= 0.92
        if T < 0.1:
            break
    
    return best_energy, best_pos


def construct(instance):
    """Diverse crystalline seeds + multi-scale basin hopping refinement."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind"))
    
    deadline = _START_TIME + _TIME_LIMIT - 15  # Extra safety margin
    rng = np.random.default_rng(42)
    
    # Generate diverse starting points
    starts = []
    
    # FCC
    try:
        starts.append(('fcc', _fcc_cluster(size)))
    except Exception:
        pass
    
    # BCC
    try:
        starts.append(('bcc', _bcc_cluster(size)))
    except Exception:
        pass
    
    # HCP
    try:
        starts.append(('hcp', _hcp_cluster(size)))
    except Exception:
        pass
    
    # Icosahedral
    try:
        starts.append(('ico', _icosahedral_cluster(size)))
    except Exception:
        pass
    
    # Spherical shell
    try:
        starts.append(('shell', _spherical_shell(size)))
    except Exception:
        pass
    
    # Random compact
    try:
        starts.append(('rand1', _random_compact(size, seed=1)))
        starts.append(('rand2', _random_compact(size, seed=2)))
    except Exception:
        pass
    
    # If no valid starts, use a simple fallback
    if not starts:
        starts.append(('fallback', _fcc_cluster(size)))
    
    # Run basin hopping on each start with time allocation
    n_starts = len(starts)
    time_per_start = max(5.0, (deadline - time.time()) / n_starts - 5.0)
    
    best_energy = np.inf
    best_pos = None
    
    for idx, (name, start_pos) in enumerate(starts):
        if time.time() > deadline:
            break
        
        # Allocate more iterations to earlier (typically better) starts
        if idx < 3:
            n_iter = max(15, int(time_per_start / 2.0))
            step_size = 0.3
        elif idx < 5:
            n_iter = max(10, int(time_per_start / 3.0))
            step_size = 0.25
        else:
            n_iter = max(8, int(time_per_start / 4.0))
            step_size = 0.2
        
        try:
            e, pos = _basin_hop(start_pos, rho, n_iter, step_size, rng, deadline)
            if e < best_energy:
                best_energy = e
                best_pos = pos.copy()
        except Exception:
            continue
    
    # If basin hopping didn't find anything good, try direct relaxation of each start
    if best_energy >= np.inf or best_pos is None:
        for name, start_pos in starts:
            if time.time() > deadline:
                break
            try:
                relaxed = _local_minimize(start_pos, rho, deadline, maxiter=100)
                e, _ = _morse_energy_and_grad(relaxed, rho)
                if e < best_energy:
                    best_energy = e
                    best_pos = relaxed.copy()
            except Exception:
                continue
    
    # Final polish: tight relaxation on the best
    if best_pos is not None and time.time() < deadline - 5:
        final_pos = _local_minimize(best_pos, rho, deadline, maxiter=300, ftol=1e-12)
        if _is_valid(final_pos):
            best_pos = final_pos
    
    if best_pos is None:
        return _fcc_cluster(size)
    
    return best_pos
# EVOLVE-BLOCK-END
