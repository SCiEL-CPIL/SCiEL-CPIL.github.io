"""Morse cluster global minima finder using basin hopping with diverse structural starts."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient
from scipy.optimize import minimize

# Global time tracking across all six instances
_GLOBAL_START = None
_GLOBAL_DEADLINE = None


def _get_global_deadline():
    """Get or initialize the global deadline."""
    global _GLOBAL_START, _GLOBAL_DEADLINE
    if _GLOBAL_START is None:
        _GLOBAL_START = time.time()
        _GLOBAL_DEADLINE = _GLOBAL_START + 280.0  # safety margin
    return _GLOBAL_DEADLINE


def _time_remaining():
    """Get remaining time in seconds."""
    return _get_global_deadline() - time.time()


def _morse_energy(pos, rho):
    """Compute Morse energy for a configuration using vectorized operations."""
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


def _morse_energy_gradient(pos, rho):
    """Compute Morse energy gradient for a configuration."""
    n = len(pos)
    grad = np.zeros_like(pos)
    if n < 2:
        return grad
    for i in range(n):
        for j in range(i + 1, n):
            rij = pos[i] - pos[j]
            r = np.linalg.norm(rij)
            if r < 1e-12:
                continue
            val = rho * (1.0 - r)
            exp_val = np.exp(val)
            dEdr = -rho * exp_val * (2.0 * exp_val - 2.0)
            grad[i] += dEdr * rij / r
            grad[j] -= dEdr * rij / r
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


def _fix_min_dist(pos, min_d=0.5, max_iters=200):
    """Try to fix minimum distance violations by moving atoms apart."""
    pos = pos.copy()
    n = len(pos)
    if n < 2:
        return pos
    for _ in range(max_iters):
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_d_actual = np.min(dists)
        if min_d_actual >= min_d - 1e-9:
            break
        i, j = np.unravel_index(np.argmin(dists), dists.shape)
        rij = pos[i] - pos[j]
        r = np.linalg.norm(rij)
        if r < 1e-12:
            pos[i] += np.array([0.01, 0.01, 0.01])
        else:
            direction = rij / r
            needed = (min_d - r) / 2.0 + 0.01
            pos[i] += direction * needed
            pos[j] -= direction * needed
    return pos


def _generate_fcc(n, spacing=1.0):
    """Generate an FCC-like cluster of n atoms."""
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
    return pts[idx]


def _generate_bcc(n, spacing=1.0):
    """Generate a BCC-like cluster of n atoms."""
    pts = []
    a = spacing
    max_r = int(np.ceil(n ** (1/3))) + 4
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                pts.append(np.array([i, j, k]) * a)
                pts.append(np.array([i + 0.5, j + 0.5, k + 0.5]) * a)
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx]


def _generate_hcp(n, spacing=1.0):
    """Generate an HCP-like cluster of n atoms."""
    pts = []
    a = spacing
    c = a * np.sqrt(8.0 / 3.0)
    max_r = int(np.ceil(n ** (1/3))) + 4
    for i in range(-max_r, max_r + 1):
        for j in range(-max_r, max_r + 1):
            for k in range(-max_r, max_r + 1):
                p1 = np.array([i, j, k * c / a])
                p2 = np.array([i + 1/3, j + 2/3, (k + 0.5) * c / a])
                pts.append(p1 * a)
                pts.append(p2 * a)
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:n]
    return pts[idx]


def _generate_icosahedral(n, spacing=1.0):
    """Generate icosahedral cluster."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True) * spacing
    
    pts = [np.zeros(3)]  # center
    pts.extend([v.copy() for v in verts])  # first shell: 12
    
    # Second shell: 30 atoms at edge midpoints and face centers
    if n > 13:
        for i in range(12):
            for j in range(i + 1, 12):
                d = np.linalg.norm(verts[i] - verts[j])
                if d < 1.8 * spacing:
                    mid = (verts[i] + verts[j]) / 2.0
                    mid = mid / np.linalg.norm(mid) * spacing * 1.4
                    if len(pts) < n:
                        pts.append(mid)
    
    # Fill remaining with random points on sphere
    rng = np.random.default_rng(42)
    while len(pts) < n:
        theta = rng.uniform(0, 2 * np.pi)
        z = rng.uniform(-1, 1)
        r_xy = np.sqrt(1 - z * z)
        p = np.array([r_xy * np.cos(theta), r_xy * np.sin(theta), z]) * spacing * 2.0
        pts.append(p)
    
    return np.array(pts[:n])


def _generate_random_compact(n, rng, spacing=1.0):
    """Generate a random compact cluster using rejection sampling."""
    radius = spacing * (n / (4/3 * np.pi * 0.5)) ** (1/3) * 1.5
    pts = []
    min_d = 0.55
    attempts = 0
    max_attempts = 100000
    while len(pts) < n and attempts < max_attempts:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            if all(np.linalg.norm(p - q) >= min_d for q in pts):
                pts.append(p)
        attempts += 1
    while len(pts) < n:
        p = rng.uniform(-radius, radius, 3)
        pts.append(p)
    return np.array(pts)


def _generate_perturbed_crystal(n, crystal_type, rng, spacing=1.0, pert_scale=0.1):
    """Generate a perturbed crystal structure."""
    if crystal_type == 'fcc':
        base = _generate_fcc(n, spacing)
    elif crystal_type == 'bcc':
        base = _generate_bcc(n, spacing)
    elif crystal_type == 'hcp':
        base = _generate_hcp(n, spacing)
    else:
        base = _generate_fcc(n, spacing)
    
    perturbed = base + rng.normal(0, pert_scale, base.shape)
    return perturbed


def _local_optimize(pos, rho, potential, max_time=10.0):
    """Local optimization using L-BFGS-B with time limit."""
    start = time.time()
    
    def objective(x):
        p = x.reshape(-1, 3)
        return _morse_energy(p, rho)
    
    def gradient(x):
        p = x.reshape(-1, 3)
        return _morse_energy_gradient(p, rho).flatten()
    
    x0 = pos.flatten()
    n = len(pos)
    radius = 3.0 * n ** (1/3)
    bounds = [(-radius, radius)] * (3 * n)
    
    try:
        result = minimize(
            objective, x0, jac=gradient,
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 300, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        if time.time() - start > max_time:
            return _morse_energy(pos, rho), pos
        if result.success or result.fun < _morse_energy(pos, rho):
            return float(result.fun), result.x.reshape(-1, 3)
        else:
            return _morse_energy(pos, rho), pos
    except Exception:
        return _morse_energy(pos, rho), pos


def _basin_hopping_step(pos, rho, potential, step_size=0.3, rng=None):
    """Perform one basin hopping step: perturb then locally optimize."""
    if rng is None:
        rng = np.random.default_rng()
    
    # Perturb
    perturbed = pos + rng.normal(0, step_size, pos.shape)
    
    # Fix minimum distance if violated
    if not _min_dist_check(perturbed):
        perturbed = _fix_min_dist(perturbed)
    
    # Local optimize
    energy, optimized = _local_optimize(perturbed, rho, potential, max_time=5.0)
    
    return energy, optimized


def _basin_hopping(pos, rho, potential, n_steps=20, step_size=0.3, rng=None):
    """Basin hopping algorithm."""
    if rng is None:
        rng = np.random.default_rng()
    
    best_energy = _morse_energy(pos, rho)
    best_pos = pos.copy()
    current_energy = best_energy
    current_pos = pos.copy()
    
    for step in range(n_steps):
        if _time_remaining() < 10:
            break
        
        # Basin hopping step
        energy, optimized = _basin_hopping_step(current_pos, rho, potential, step_size, rng)
        
        if not _min_dist_check(optimized):
            continue
        
        # Accept if better or with Metropolis criterion
        if energy < current_energy:
            current_energy = energy
            current_pos = optimized.copy()
            if energy < best_energy:
                best_energy = energy
                best_pos = optimized.copy()
        else:
            # Metropolis acceptance
            delta_e = energy - current_energy
            if delta_e < 0:
                current_energy = energy
                current_pos = optimized.copy()
            else:
                # Simple acceptance: accept with probability exp(-delta_e / T)
                # Use a fixed temperature
                T = 0.1
                if rng.random() < np.exp(-delta_e / T):
                    current_energy = energy
                    current_pos = optimized.copy()
    
    return best_energy, best_pos


def construct(instance):
    """Find low-energy Morse cluster configuration."""
    size = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind"))
    
    _get_global_deadline()
    rng = np.random.default_rng(size * 13 + 7)
    
    best_energy = np.inf
    best_pos = None
    
    # Generate diverse initial configurations
    initial_configs = []
    
    # 1. FCC-based
    try:
        fcc = _generate_fcc(size, spacing=1.0)
        if _min_dist_check(fcc):
            initial_configs.append(('fcc', fcc))
    except Exception:
        pass
    
    # 2. BCC-based
    try:
        bcc = _generate_bcc(size, spacing=1.0)
        if _min_dist_check(bcc):
            initial_configs.append(('bcc', bcc))
    except Exception:
        pass
    
    # 3. HCP-based
    try:
        hcp = _generate_hcp(size, spacing=1.0)
        if _min_dist_check(hcp):
            initial_configs.append(('hcp', hcp))
    except Exception:
        pass
    
    # 4. Icosahedral
    try:
        ico = _generate_icosahedral(size, spacing=1.0)
        if _min_dist_check(ico):
            initial_configs.append(('ico', ico))
    except Exception:
        pass
    
    # 5. Random compact (multiple variants)
    for i in range(6):
        try:
            rc = _generate_random_compact(size, rng, spacing=1.0)
            if _min_dist_check(rc):
                initial_configs.append((f'rand_compact{i}', rc))
            else:
                fixed = _fix_min_dist(rc)
                if _min_dist_check(fixed):
                    initial_configs.append((f'rand_compact{i}_fix', fixed))
        except Exception:
            pass
    
    # 6. Perturbed crystals
    for crystal_type in ['fcc', 'bcc', 'hcp']:
        for pert_scale in [0.05, 0.15, 0.3]:
            try:
                pert = _generate_perturbed_crystal(size, crystal_type, rng, spacing=1.0, pert_scale=pert_scale)
                if _min_dist_check(pert):
                    initial_configs.append((f'{crystal_type}_pert{pert_scale}', pert))
                else:
                    fixed = _fix_min_dist(pert)
                    if _min_dist_check(fixed):
                        initial_configs.append((f'{crystal_type}_pert{pert_scale}_fix', fixed))
            except Exception:
                pass
    
    # Sort initial configs by initial energy (best first)
    def _config_energy(name_pos):
        try:
            return _morse_energy(name_pos[1], rho)
        except Exception:
            return np.inf
    
    initial_configs.sort(key=_config_energy)
    
    # Phase 1: Local optimization on each initial config
    for name, pos in initial_configs:
        if _time_remaining() < 15:
            break
        
        try:
            e1, p1 = _local_optimize(pos, rho, potential, max_time=min(10.0, _time_remaining() * 0.3))
            if _min_dist_check(p1) and e1 < best_energy:
                best_energy = e1
                best_pos = p1.copy()
        except Exception:
            pass
    
    # Phase 2: Basin hopping from best configurations
    if best_pos is not None and _time_remaining() > 20:
        # Determine number of basin hopping iterations based on remaining time
        n_bh_steps = min(30, int(_time_remaining() / 8))
        
        # Try basin hopping from the best position
        try:
            e_bh, p_bh = _basin_hopping(best_pos, rho, potential, n_steps=n_bh_steps, step_size=0.3, rng=rng)
            if _min_dist_check(p_bh) and e_bh < best_energy:
                best_energy = e_bh
                best_pos = p_bh.copy()
        except Exception:
            pass
        
        # Try basin hopping from other good starting points
        for name, pos in initial_configs[:5]:
            if _time_remaining() < 15:
                break
            if name == 'fcc' or name == 'bcc' or name == 'hcp' or name == 'ico':
                continue  # Already tried the best ones
            try:
                e_bh, p_bh = _basin_hopping(pos, rho, potential, n_steps=max(5, n_bh_steps // 2), step_size=0.3, rng=rng)
                if _min_dist_check(p_bh) and e_bh < best_energy:
                    best_energy = e_bh
                    best_pos = p_bh.copy()
            except Exception:
                pass
    
    # Phase 3: Additional random restarts with basin hopping
    if best_pos is not None and _time_remaining() > 15:
        n_extra = min(10, int(_time_remaining() / 10))
        for i in range(n_extra):
            if _time_remaining() < 10:
                break
            try:
                rp = _generate_random_compact(size, rng, spacing=1.0)
                if not _min_dist_check(rp):
                    rp = _fix_min_dist(rp)
                if not _min_dist_check(rp):
                    continue
                
                # Quick local optimization
                e_local, p_local = _local_optimize(rp, rho, potential, max_time=min(5.0, _time_remaining() * 0.3))
                if not _min_dist_check(p_local):
                    continue
                
                # Basin hopping from this local minimum
                n_bh = max(3, min(10, int(_time_remaining() / 10)))
                e_bh, p_bh = _basin_hopping(p_local, rho, potential, n_steps=n_bh, step_size=0.3, rng=rng)
                if _min_dist_check(p_bh) and e_bh < best_energy:
                    best_energy = e_bh
                    best_pos = p_bh.copy()
            except Exception:
                continue
    
    # Phase 4: Final polish with small perturbations
    if best_pos is not None and _time_remaining() > 5:
        rng_polish = np.random.default_rng(999)
        for _ in range(5):
            if _time_remaining() < 3:
                break
            perturbed = best_pos + rng_polish.normal(0, 0.02, best_pos.shape)
            if not _min_dist_check(perturbed):
                perturbed = _fix_min_dist(perturbed)
            try:
                e_polish, p_polish = _local_optimize(perturbed, rho, potential, max_time=min(3.0, _time_remaining() * 0.4))
                if _min_dist_check(p_polish) and e_polish < best_energy:
                    best_energy = e_polish
                    best_pos = p_polish.copy()
            except Exception:
                pass
    
    # Final fallback: if no valid result, create a simple valid config
    if best_pos is None:
        # Create a linear chain
        spacing = 0.6
        best_pos = np.zeros((size, 3))
        for i in range(size):
            best_pos[i] = [i * spacing, 0, 0]
        if not _min_dist_check(best_pos):
            # Spread them out more
            radius = 0.6 * size ** (1/3) * 1.5
            valid = [best_pos[0]]
            for i in range(1, size):
                attempts = 0
                while attempts < 2000:
                    p = rng.uniform(-radius, radius, 3)
                    if all(np.linalg.norm(p - v) >= 0.5 for v in valid):
                        valid.append(p)
                        break
                    attempts += 1
                else:
                    valid.append(rng.uniform(-radius, radius, 3))
            best_pos = np.array(valid[:size])
    
    return best_pos
# EVOLVE-BLOCK-END
