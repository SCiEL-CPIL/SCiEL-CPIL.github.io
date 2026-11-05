# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient


def _energy_morse(pos, rho):
    """Compute Morse energy E = sum_{i<j} exp(rho(1-r)) (exp(rho(1-r))-2)."""
    n = pos.shape[0]
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


def _bfgs_local(pos, rho, max_iter=200, tol=1e-10):
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
                   options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8})
    return res.fun, res.x.reshape(-1, 3)


def _icosahedron_seed(n, rng):
    """Generate an icosahedral-like cluster with shells."""
    phi = (1 + np.sqrt(5)) / 2
    base = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
    ], dtype=float)
    base = base / 2.0
    
    points = [np.zeros(3)]
    points.extend(base.tolist())
    
    if n > 13:
        remaining = n - len(points)
        shell_radius = 1.0 * np.sqrt(2)
        n_shell = min(remaining, 30)
        for k in range(n_shell):
            t = (k + 0.5) / n_shell
            z = 1 - 2 * t
            r = np.sqrt(max(0, 1 - z * z))
            theta = 2 * np.pi * k / n_shell
            p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * shell_radius
            points.append(p)
        remaining -= n_shell
        if remaining > 0:
            shell_radius2 = 1.0 * np.sqrt(3)
            for k in range(remaining):
                t = (k + 0.5) / remaining
                z = 1 - 2 * t
                r = np.sqrt(max(0, 1 - z * z))
                theta = 2 * np.pi * k / remaining
                p = np.array([r * np.cos(theta), r * np.sin(theta), z]) * shell_radius2
                points.append(p)
    
    return np.array(points[:n])


def _fcc_seed(n, rho):
    """Generate an FCC-like cluster."""
    a = 1.0
    max_coord = int(np.ceil(n ** (1/3))) + 2
    points = []
    for i in range(-max_coord, max_coord + 1):
        for j in range(-max_coord, max_coord + 1):
            for k in range(-max_coord, max_coord + 1):
                for offset in [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]:
                    p = np.array([i, j, k]) * a + np.array(offset) * a
                    if np.linalg.norm(p) < 1.5 * n ** (1/3):
                        points.append(p)
    if len(points) > 0:
        points = np.array(points)
        norms = np.linalg.norm(points, axis=1)
        idx = np.argsort(norms)
        points = points[idx[:n]]
    else:
        points = np.zeros((n, 3))
    return points


def _decahedral_seed(n, rng):
    """Generate a decahedral-like cluster (5-fold symmetric)."""
    points = [np.zeros(3)]
    for layer in range(1, 10):
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
    return np.array(points[:n])


def _random_compact(n, rng, spacing=1.0):
    """Generate a random compact cluster."""
    radius = spacing * n ** (1/3) * 0.8
    points = []
    attempts = 0
    while len(points) < n and attempts < n * 200:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= 0.55 * spacing for q in points):
                points.append(p)
        attempts += 1
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    return np.array(points)


def _shell_seed(n, rng):
    """Generate a shell-like cluster with Fibonacci spheres."""
    shell_pts = [np.zeros(3)]
    shell_r = 1.0
    golden = (1 + np.sqrt(5)) / 2
    while len(shell_pts) < n:
        num_on_shell = min(n - len(shell_pts), int(12 * shell_r**2))
        if num_on_shell < 2:
            num_on_shell = 2
        for m in range(num_on_shell):
            z = 1 - 2 * (m + 0.5) / num_on_shell
            r_z = np.sqrt(max(0, 1 - z*z))
            theta = golden * m * 2 * np.pi
            p = shell_r * np.array([r_z * np.cos(theta), r_z * np.sin(theta), z])
            shell_pts.append(p)
        shell_r += 1.5
    return np.array(shell_pts[:n])


def _make_initial_guesses(n, rho, rng):
    """Generate multiple initial configurations."""
    guesses = []
    
    try:
        guesses.append(_icosahedron_seed(n, rng))
    except Exception:
        pass
    
    try:
        guesses.append(_fcc_seed(n, rho))
    except Exception:
        pass
    
    try:
        guesses.append(_decahedral_seed(n, rng))
    except Exception:
        pass
    
    try:
        guesses.append(_shell_seed(n, rng))
    except Exception:
        pass
    
    for _ in range(6):
        try:
            guesses.append(_random_compact(n, rng))
        except Exception:
            pass
    
    return guesses


def construct(instance):
    """Find low-energy Morse cluster configurations using basin hopping."""
    start_time = time.time()
    deadline = 330.0
    safety_margin = 5.0
    
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 6))
    
    rng = np.random.default_rng(size * 7919 + int(rho))
    
    best_energy = np.inf
    best_pos = None
    
    guesses = _make_initial_guesses(size, rho, rng)
    
    # Relax each initial guess
    for i, guess in enumerate(guesses):
        if time.time() - start_time > deadline - safety_margin:
            break
        
        try:
            energy, positions = relax(guess, potential, rho)
            min_d = _min_distance(positions)
            if min_d >= 0.5 - 1e-6:
                if energy < best_energy:
                    best_energy = energy
                    best_pos = positions.copy()
        except Exception:
            pass
        
        if time.time() - start_time > deadline - safety_margin:
            break
        try:
            e2, p2 = _bfgs_local(guess, rho, max_iter=300)
            min_d = _min_distance(p2)
            if min_d >= 0.5 - 1e-6:
                if e2 < best_energy:
                    best_energy = e2
                    best_pos = p2.copy()
        except Exception:
            pass
    
    if best_pos is None:
        best_pos = _random_compact(size, rng)
        try:
            best_energy, best_pos = relax(best_pos, potential, rho)
        except Exception:
            pass
    
    # Basin hopping
    if best_pos is not None:
        T = 0.5
        T_min = 0.005
        cooling = 0.97
        max_sa_steps = 800
        current_pos = best_pos.copy()
        current_energy = best_energy
        
        for step in range(max_sa_steps):
            if time.time() - start_time > deadline - safety_margin:
                break
            if T < T_min:
                break
            
            n_perturb = max(1, size // 5)
            idx = rng.choice(size, n_perturb, replace=False)
            noise_scale = 0.15 * (1 + T)
            perturbed = current_pos.copy()
            perturbed[idx] += rng.normal(0, noise_scale, (n_perturb, 3))
            
            try:
                energy, positions = relax(perturbed, potential, rho)
                min_d = _min_distance(positions)
                if min_d < 0.5 - 1e-6:
                    continue
                
                delta_e = energy - current_energy
                if delta_e < 0 or rng.random() < np.exp(-delta_e / T):
                    current_pos = positions.copy()
                    current_energy = energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_pos = current_pos.copy()
            except Exception:
                pass
            
            T *= cooling
        
        # Additional targeted perturbations
        for trial in range(20):
            if time.time() - start_time > deadline - safety_margin:
                break
            noise_scale = 0.1 + 0.05 * trial
            perturbed = best_pos + rng.normal(0, noise_scale, best_pos.shape)
            try:
                energy, positions = relax(perturbed, potential, rho)
                min_d = _min_distance(positions)
                if min_d >= 0.5 - 1e-6:
                    if energy < best_energy:
                        best_energy = energy
                        best_pos = positions.copy()
            except Exception:
                pass
        
        # Larger perturbations
        for trial in range(15):
            if time.time() - start_time > deadline - safety_margin:
                break
            noise_scale = 0.5 + 0.15 * trial
            perturbed = best_pos + rng.normal(0, noise_scale, best_pos.shape)
            try:
                energy, positions = relax(perturbed, potential, rho)
                min_d = _min_distance(positions)
                if min_d >= 0.5 - 1e-6:
                    if energy < best_energy:
                        best_energy = energy
                        best_pos = positions.copy()
            except Exception:
                pass
        
        # Kick-out perturbations
        for trial in range(10):
            if time.time() - start_time > deadline - safety_margin:
                break
            idx = rng.integers(0, size)
            perturbed = best_pos.copy()
            perturbed[idx] = best_pos[idx] + rng.normal(0, 1.2, 3)
            try:
                energy, positions = relax(perturbed, potential, rho)
                min_d = _min_distance(positions)
                if min_d >= 0.5 - 1e-6:
                    if energy < best_energy:
                        best_energy = energy
                        best_pos = positions.copy()
            except Exception:
                pass
    
    if best_pos is None:
        best_pos = np.zeros((size, 3))
        for i in range(size):
            best_pos[i, 0] = i * 1.0
        best_pos -= best_pos.mean(axis=0)
    
    return best_pos
# EVOLVE-BLOCK-END
