"""Morse cluster global minima: Parallel Adaptive Basin Hopping with L-BFGS-B."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START = time.time()
_BUDGET = 315.0  # Safety margin before 330s kill
_DEADLINE = _START + _BUDGET

def _remaining():
    return max(0.0, _DEADLINE - time.time())

def _morse_energy(rho, coords):
    """Exact Morse energy; returns large penalty if any pair < 0.5."""
    n = len(coords)
    if n < 2:
        return 0.0
    d = cdist(coords, coords)
    np.fill_diagonal(d, np.inf)
    if np.min(d) < 0.5:
        return 1e10
    e = np.exp(rho * (1.0 - d))
    return np.sum(e * (e - 2.0)) / 2.0

def _fix_overlaps(coords, min_dist=0.5, max_iter=500):
    """Push apart any atoms closer than min_dist."""
    coords = coords.copy()
    n = len(coords)
    for _ in range(max_iter):
        d = cdist(coords, coords)
        np.fill_diagonal(d, np.inf)
        min_d = np.min(d)
        if min_d >= min_dist:
            break
        idx = np.unravel_index(np.argmin(d), d.shape)
        i, j = idx
        if i == j:
            continue
        diff = coords[i] - coords[j]
        dist = np.linalg.norm(diff)
        if dist < 1e-8:
            coords[i] += 0.01 * np.array([1, 0, 0])
            continue
        diff /= dist
        push = (min_dist - dist) * 0.5 + 0.01
        coords[i] += push * diff
        coords[j] -= push * diff
    return coords

def _local_minimize(rho, x0):
    """
    Perform local minimization using L-BFGS-B primarily, falling back to relax.
    Returns (energy, coords).
    """
    n = len(x0)
    x0 = _fix_overlaps(x0)
    
    # Try L-BFGS-B with gradient
    x0_flat = x0.flatten()
    
    def func_and_grad(x_flat):
        x = x_flat.reshape(n, 3)
        # Quick overlap check to avoid NaNs in gradient if needed, though L-BFGS-B handles it somewhat
        # But morse_energy_gradient might behave badly if d < 0.5
        d = cdist(x, x)
        np.fill_diagonal(d, np.inf)
        if np.min(d) < 0.5:
             # Return a high energy and zero gradient to push away? 
             # Actually, just return high energy. L-BFGS-B will struggle.
             return 1e10, np.zeros_like(x_flat)
        
        e, g = morse_energy_gradient(x, rho)
        return e, g.flatten()
        
    try:
        res = minimize(func_and_grad, x0_flat, method='L-BFGS-B', 
                       options={'maxiter': 200, 'gtol': 1e-5, 'ftol': 1e-8})
        x_opt = res.x.reshape(n, 3)
        e_opt = _morse_energy(rho, x_opt)
        if e_opt < 1e8:
            # Verify it's better than or equal to initial relaxed state if we had one?
            # Just return it.
            return e_opt, x_opt
    except Exception:
        pass
    
    # Fallback to relax if L-BFGS-B fails or produces invalid result
    try:
        e_rel, x_rel = relax(x0, 'morse', rho)
        e_check = _morse_energy(rho, x_rel)
        if e_check < 1e8:
            return e_check, x_rel
        return e_rel, x_rel
    except Exception:
        return 1e10, x0

def _generate_fcc_seeds(n, rng, count=2):
    """Generate FCC-like seeds with perturbations and rotations."""
    seeds = []
    a = 1.0
    pts = []
    limit = int(np.ceil((n ** (1/3)) * 1.8)) + 2
    for i in range(-limit, limit + 1):
        for j in range(-limit, limit + 1):
            for k in range(-limit, limit + 1):
                pts.append([i*a, j*a, k*a])
                pts.append([(i+0.5)*a, (j+0.5)*a, k*a])
                pts.append([(i+0.5)*a, j*a, (k+0.5)*a])
                pts.append([i*a, (j+0.5)*a, (k+0.5)*a])
    
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    base_fcc = pts[order[:n]].copy()
    
    for _ in range(count):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
        rotated = base_fcc @ R.T
        noisy = rotated + rng.normal(0, 0.1, rotated.shape)
        seeds.append(_fix_overlaps(noisy))
        
    return seeds

def _generate_hcp_seeds(n, rng, count=2):
    """Generate HCP-like seeds."""
    seeds = []
    a = 1.0
    pts = []
    limit = int(np.ceil((n ** (1/3)) * 1.8)) + 2
    # HCP structure: ABAB... stacking
    # Layer A: (i, j, 0)
    # Layer B: (i+0.5, j+0.5/sqrt(3)*?, 1/sqrt(2/3)) -> simplified for seed generation
    # Just use a distorted FCC or a specific HCP lattice generation
    # For simplicity, generate a hexagonal lattice in 2D and stack with offset
    
    hex_a = 1.0
    hex_c = 1.633 # Ideal HCP c/a ratio is ~1.633, but for Morse rho=3-10, optimal distance is 1.0.
    # Let's just use a standard hexagonal close packing geometry
    
    for k in range(-limit, limit + 1):
        z = k * hex_c
        for i in range(-limit, limit + 1):
            for j in range(-limit, limit + 1):
                # Triangular lattice in xy plane
                x = i * hex_a + (j % 2) * hex_a * 0.5
                y = j * hex_a * np.sqrt(3) / 2
                pts.append([x, y, z])
                
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    base_hcp = pts[order[:n]].copy()
    
    for _ in range(count):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
        rotated = base_hcp @ R.T
        noisy = rotated + rng.normal(0, 0.1, rotated.shape)
        seeds.append(_fix_overlaps(noisy))
        
    return seeds

def _generate_icosahedral_seeds(n, rng, count=2):
    """Generate icosahedral-like seeds."""
    seeds = []
    phi = (1 + np.sqrt(5)) / 2
    base = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
        [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]
    ], dtype=float)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    
    pts = [np.zeros(3)]
    radii = [1.0, 1.7, 2.4, 3.1, 3.8, 4.5]
    for r in radii:
        for p in base:
            pts.append(p * r)
            pts.append(-p * r)
    
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    base_icosa = pts[order[:n]].copy()
    
    for _ in range(count):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
        rotated = base_icosa @ R.T
        noisy = rotated + rng.normal(0, 0.08, rotated.shape)
        seeds.append(_fix_overlaps(noisy))
        
    return seeds

def _generate_spherical_shell_seeds(n, rng, count=2):
    """Generate spherical shell seeds."""
    seeds = []
    def fibonacci_sphere(n_points):
        points = []
        golden_ratio = (1 + np.sqrt(5)) / 2
        for i in range(n_points):
            z = 1 - 2 * i / max(1, n_points - 1)
            radius_at_z = np.sqrt(1 - z * z)
            theta = 2 * np.pi * i / golden_ratio
            x = np.cos(theta) * radius_at_z
            y = np.sin(theta) * radius_at_z
            points.append([x, y, z])
        return np.array(points)
    
    configs = [
        (1, 0.8),
        (2, 1.5),
        (3, 2.2),
    ]
    
    for cfg_idx in range(min(count, len(configs))):
        n_shells, shell_spacing = configs[cfg_idx]
        pts = [np.zeros(3)]
        
        atoms_per_shell = []
        remaining = n - 1
        for s in range(n_shells):
            if s == n_shells - 1:
                atoms_per_shell.append(remaining)
            else:
                frac = (s + 1) / n_shells
                count_s = max(1, int(remaining * frac * 0.7))
                atoms_per_shell.append(count_s)
                remaining -= count_s
        
        for s, count_s in enumerate(atoms_per_shell):
            if count_s <= 0:
                continue
            r = shell_spacing * (s + 1)
            shell_pts = fibonacci_sphere(count_s) * r
            pts.extend(shell_pts.tolist())
        
        pts = np.array(pts[:n])
        pts -= pts.mean(axis=0)
        
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
        rotated = pts @ R.T
        noisy = rotated + rng.normal(0, 0.05, rotated.shape)
        seeds.append(_fix_overlaps(noisy))
        
    return seeds

def _generate_random_compact_seeds(n, rng, count=2):
    """Generate random compact seeds."""
    seeds = []
    for _ in range(count):
        box_size = (n * 0.3) ** (1/3) * 3.0
        pts = rng.uniform(-box_size, box_size, size=(n, 3))
        seeds.append(_fix_overlaps(pts))
    return seeds

def _basin_hopping_chain(rho, x0, rng, duration, seed_id):
    """
    Run a single basin hopping chain for a specified duration.
    """
    e_rel, x_rel = _local_minimize(rho, x0)
    if e_rel > 1e8:
        return x0, 1e10

    cur_e = e_rel
    best_e = e_rel
    best_x = x_rel.copy()
    cur_x = x_rel.copy()
    
    temp = 0.3
    t_start = time.time()
    n = len(x0)
    accepted = 0
    total = 0
    
    while _remaining() > 1.0 and (time.time() - t_start) < duration:
        shake_scale = 0.08 + 0.15 * np.sqrt(temp)
        
        n_perturb = max(1, int(n * (0.08 + 0.08 * np.sqrt(temp))))
        idx = rng.choice(n, size=n_perturb, replace=False)
        noise = rng.normal(0, shake_scale, size=(n_perturb, 3))
        x_new = cur_x.copy()
        x_new[idx] += noise
        
        e_new, x_new_rel = _local_minimize(rho, x_new)
        
        if e_new > 1e8:
            continue

        total += 1
        dE = e_new - cur_e
        if dE < 0 or (temp > 0.01 and rng.random() < np.exp(-dE / temp)):
            cur_e = e_new
            cur_x = x_new_rel
            if cur_e < best_e:
                best_e = cur_e
                best_x = x_new_rel.copy()
                accepted += 1
        
        temp *= 0.96
        if total > 30 and accepted == 0:
            temp = max(temp, 0.2)
            accepted = 0
            total = 0
    
    return best_x, best_e

def construct(instance):
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    rng = np.random.default_rng(n * 1000 + int(rho))
    
    # Generate diverse seeds
    fcc_seeds = _generate_fcc_seeds(n, rng, count=2)
    hcp_seeds = _generate_hcp_seeds(n, rng, count=2)
    icosa_seeds = _generate_icosahedral_seeds(n, rng, count=2)
    shell_seeds = _generate_spherical_shell_seeds(n, rng, count=2)
    random_seeds = _generate_random_compact_seeds(n, rng, count=2)
    
    # Select 4 diverse starting points (one per thread)
    # Mix of close-packed and other structures
    chains_starts = [fcc_seeds[0], hcp_seeds[0], icosa_seeds[0], shell_seeds[0]]
    
    total_time = _remaining()
    polish_time = max(8.0, total_time * 0.10)
    search_time = total_time - polish_time
    time_per_chain = search_time / 4.0
    
    best_x_global = None
    best_e_global = np.inf
    
    def run_chain(args):
        x0, t_dur, seed_id = args
        x_res, e_res = _basin_hopping_chain(rho, x0, np.random.default_rng(seed_id), t_dur, seed_id)
        return x_res, e_res

    args = []
    for i, start in enumerate(chains_starts):
        args.append((start, time_per_chain, n * 100 + i))
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_chain, arg) for arg in args]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                x_res, e_res = future.result()
                if e_res < best_e_global:
                    best_e_global = e_res
                    best_x_global = x_res
            except Exception:
                pass
                
    if best_x_global is None:
        return _fix_overlaps(chains_starts[0])
        
    # Polish phase
    if _remaining() > polish_time:
        rng_polish = np.random.default_rng(42 + n)
        best_x_global, best_e_global = _basin_hopping_chain(rho, best_x_global, rng_polish, polish_time, 999)
        
    e_final = _morse_energy(rho, best_x_global)
    if e_final < 1e8:
        return best_x_global
    else:
        best_x_global = _fix_overlaps(best_x_global)
        e_final = _morse_energy(rho, best_x_global)
        if e_final < 1e8:
            return best_x_global
        return _fix_overlaps(chains_starts[0])

# EVOLVE-BLOCK-END
