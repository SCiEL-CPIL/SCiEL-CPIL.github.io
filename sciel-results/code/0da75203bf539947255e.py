# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _energy_morse(pos, rho):
    """Compute Morse energy E = sum_{i<j} exp(rho(1-r)) (exp(rho(1-r))-2)."""
    n = pos.shape[0]
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    iu = np.triu_indices(n, k=1)
    r = dists[iu]
    e = np.exp(rho * (1.0 - r))
    return np.sum(e * (e - 2.0))


def _min_distance(pos):
    """Compute minimum interatomic distance."""
    if pos.shape[0] < 2:
        return np.inf
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    return np.min(dists)


def _gradient_morse(pos, rho):
    """Compute gradient of Morse energy."""
    n = pos.shape[0]
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-15)
    e = np.exp(rho * (1.0 - dists))
    dE_dr = rho * e * (2.0 * e - 2.0)
    mask = ~np.eye(n, dtype=bool)
    inv_r = np.where(mask, 1.0 / dists, 0.0)
    factor = -dE_dr * inv_r
    factor = factor * mask
    grad = np.einsum('ij,ijk->ik', factor, diff)
    return grad


def _bfgs_local(pos, rho, max_iter=500, tol=1e-12):
    """Local optimization using L-BFGS-B via scipy."""
    from scipy.optimize import minimize
    
    def fun(x):
        p = x.reshape(-1, 3)
        return _energy_morse(p, rho)
    
    def jac(x):
        p = x.reshape(-1, 3)
        g = _gradient_morse(p, rho)
        return g.ravel()
    
    x0 = pos.ravel()
    res = minimize(fun, x0, jac=jac, method='L-BFGS-B', 
                   options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-10})
    return res.fun, res.x.reshape(-1, 3)


def _mckay_icosahedron(n, rng):
    """Generate Mackay icosahedral cluster with shells.
    Shell sizes: 1, 12, 42, 92, 165, 261, 380, 522, 689, 880, ...
    Formula for shell k (k>=1): 10*k*(k+1) + 1
    """
    # Build icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    base = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    base = base / np.linalg.norm(base[0])
    
    points = [np.zeros(3)]
    
    # Shell 1: 12 vertices at unit distance
    points.extend(base.tolist())
    
    # Shell 2: 42 points (30 mid-edge + 12 face centers)
    if n > 13:
        # Edge midpoints: 30 edges of icosahedron
        edges = []
        verts = base
        for i in range(12):
            for j in range(i+1, 12):
                d = np.linalg.norm(verts[i] - verts[j])
                if d < 1.01:  # close to unit distance = edge
                    mid = (verts[i] + verts[j]) / 2
                    mid = mid / np.linalg.norm(mid) * np.sqrt(3)/2
                    edges.append(mid)
        points.extend(edges[:min(30, n - len(points))])
    
    # Shell 3: 92 points
    if n > 55:
        # Use Fibonacci sphere for remaining
        remaining = n - len(points)
        shell_r = np.sqrt(3)
        for k in range(remaining):
            t = (k + 0.5) / remaining
            z = 1 - 2 * t
            r_z = np.sqrt(max(0, 1 - z*z))
            theta = 2 * np.pi * k * (1 + np.sqrt(5)) / 2
            p = shell_r * np.array([r_z * np.cos(theta), r_z * np.sin(theta), z])
            points.append(p)
    
    return np.array(points[:n])


def _fcc_cluster(n, rng):
    """Generate FCC-like cluster by taking closest n points from FCC lattice."""
    a = 1.0
    max_coord = int(np.ceil(n ** (1/3))) + 3
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for offset in [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]:
                    p = np.array([i, j, k]) * a + np.array(offset) * a
                    points.append(p)
    points = np.array(points)
    norms = np.linalg.norm(points, axis=1)
    idx = np.argsort(norms)
    points = points[idx[:n]]
    # Center
    points -= points.mean(axis=0)
    return points


def _bcc_cluster(n, rng):
    """Generate BCC-like cluster."""
    a = 1.0
    max_coord = int(np.ceil(n ** (1/3))) + 3
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                p = np.array([i, j, k]) * a
                points.append(p)
                p2 = np.array([i, j, k]) * a + np.array([0.5, 0.5, 0.5]) * a
                points.append(p2)
    points = np.array(points)
    norms = np.linalg.norm(points, axis=1)
    idx = np.argsort(norms)
    points = points[idx[:n]]
    points -= points.mean(axis=0)
    return points


def _decahedral_cluster(n, rng):
    """Generate decahedral-like cluster (5-fold symmetry)."""
    points = [np.zeros(3)]
    for layer in range(1, 15):
        r = layer * 0.7
        n_in_ring = 10
        for k in range(n_in_ring):
            theta = 2 * np.pi * k / n_in_ring + (np.pi / 10 if layer % 2 else 0)
            p = np.array([r * np.cos(theta), r * np.sin(theta), 0])
            points.append(p)
            if len(points) >= n:
                break
        if len(points) >= n:
            break
    # Add some 3D structure for larger n
    if len(points) < n:
        remaining = n - len(points)
        for k in range(remaining):
            theta = 2 * np.pi * k / max(1, remaining)
            phi_angle = np.pi * (k + 0.5) / max(1, remaining)
            r = 0.7 * np.sqrt(remaining)
            p = r * np.array([np.sin(phi_angle) * np.cos(theta), 
                              np.sin(phi_angle) * np.sin(theta), 
                              np.cos(phi_angle)])
            points.append(p)
    return np.array(points[:n])


def _random_compact(n, rng, spacing=1.0):
    """Generate a random compact cluster with minimum distance constraint."""
    radius = spacing * n ** (1/3) * 1.0
    points = []
    attempts = 0
    max_attempts = n * 500
    while len(points) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.55 * spacing for q in points):
                points.append(p)
        attempts += 1
    while len(points) < n:
        p = rng.uniform(-radius * 1.5, radius * 1.5, 3)
        points.append(p)
    return np.array(points)


def _shell_fibonacci(n, rng):
    """Generate shell-like cluster with Fibonacci spheres."""
    shell_pts = [np.zeros(3)]
    shell_r = 1.0
    golden = (1 + np.sqrt(5)) / 2
    while len(shell_pts) < n:
        # Estimate how many fit on this shell
        num_on_shell = min(n - len(shell_pts), max(2, int(12 * shell_r**2)))
        for m in range(num_on_shell):
            z = 1 - 2 * (m + 0.5) / num_on_shell
            r_z = np.sqrt(max(0, 1 - z*z))
            theta = golden * m * 2 * np.pi
            p = shell_r * np.array([r_z * np.cos(theta), r_z * np.sin(theta), z])
            shell_pts.append(p)
        shell_r += 1.2
    return np.array(shell_pts[:n])


def _make_seeds(n, rho, rng):
    """Generate diverse initial configurations."""
    seeds = []
    
    # Structured seeds
    try:
        seeds.append(_mckay_icosahedron(n, rng))
    except Exception:
        pass
    
    try:
        seeds.append(_fcc_cluster(n, rng))
    except Exception:
        pass
    
    try:
        seeds.append(_bcc_cluster(n, rng))
    except Exception:
        pass
    
    try:
        seeds.append(_decahedral_cluster(n, rng))
    except Exception:
        pass
    
    try:
        seeds.append(_shell_fibonacci(n, rng))
    except Exception:
        pass
    
    # Random compact seeds
    for i in range(8):
        try:
            seeds.append(_random_compact(n, rng))
        except Exception:
            pass
    
    return seeds


def _relax_and_check(pos, rho, potential, min_d_thresh=0.5 - 1e-6):
    """Relax a configuration and check validity."""
    try:
        energy, positions = relax(pos, potential, rho)
        min_d = _min_distance(positions)
        if min_d >= min_d_thresh:
            return energy, positions
    except Exception:
        pass
    return None, None


def construct(instance):
    """Find low-energy Morse cluster configurations using multi-start basin hopping."""
    start_time = time.time()
    deadline = 325.0  # Safety margin before 330s kill
    
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 6))
    
    rng = np.random.default_rng(size * 7919 + int(rho) * 131)
    
    best_energy = np.inf
    best_pos = None
    
    # Phase 1: Multi-start local relaxation
    seeds = _make_seeds(size, rho, rng)
    
    for i, seed in enumerate(seeds):
        if time.time() - start_time > deadline:
            break
        
        energy, positions = _relax_and_check(seed, rho, potential)
        if energy is not None and energy < best_energy:
            best_energy = energy
            best_pos = positions.copy()
        
        # Also try BFGS directly on some seeds
        if i < 3:
            try:
                e2, p2 = _bfgs_local(seed, rho, max_iter=300)
                min_d = _min_distance(p2)
                if min_d >= 0.5 - 1e-6 and e2 < best_energy:
                    best_energy = e2
                    best_pos = p2.copy()
            except Exception:
                pass
    
    if best_pos is None:
        # Fallback
        best_pos = _random_compact(size, rng)
        energy, positions = _relax_and_check(best_pos, rho, potential)
        if energy is not None:
            best_energy = energy
            best_pos = positions
        else:
            best_energy = _energy_morse(best_pos, rho)
    
    # Phase 2: Basin hopping from best configuration
    current_pos = best_pos.copy()
    current_energy = best_energy
    
    T = 0.3
    T_min = 0.01
    cooling = 0.985
    max_bh_steps = 500
    
    for step in range(max_bh_steps):
        if time.time() - start_time > deadline:
            break
        if T < T_min:
            break
        
        # Perturb a random subset of atoms
        n_perturb = max(1, size // 8)
        idx = rng.choice(size, n_perturb, replace=False)
        noise_scale = 0.1 * (1 + 2 * T)
        perturbed = current_pos.copy()
        perturbed[idx] += rng.normal(0, noise_scale, (n_perturb, 3))
        
        energy, positions = _relax_and_check(perturbed, rho, potential)
        if energy is None:
            continue
        
        delta_e = energy - current_energy
        if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
            current_pos = positions.copy()
            current_energy = energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pos = current_pos.copy()
        
        T *= cooling
    
    # Phase 3: Targeted large perturbations from best
    for trial in range(30):
        if time.time() - start_time > deadline:
            break
        # Kick one atom far away
        idx = rng.integers(0, size)
        perturbed = best_pos.copy()
        perturbed[idx] = best_pos[idx] + rng.normal(0, 1.5, 3)
        
        energy, positions = _relax_and_check(perturbed, rho, potential)
        if energy is not None and energy < best_energy:
            best_energy = energy
            best_pos = positions.copy()
    
    # Phase 4: Additional random restarts if time permits
    if time.time() - start_time < deadline - 30:
        for trial in range(15):
            if time.time() - start_time > deadline:
                break
            seed = _random_compact(size, rng)
            energy, positions = _relax_and_check(seed, rho, potential)
            if energy is not None and energy < best_energy:
                best_energy = energy
                best_pos = positions.copy()
    
    if best_pos is None:
        best_pos = np.zeros((size, 3))
        for i in range(size):
            best_pos[i, 0] = i * 1.0
        best_pos -= best_pos.mean(axis=0)
    
    return best_pos
# EVOLVE-BLOCK-END
