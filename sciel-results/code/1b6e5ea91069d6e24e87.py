"""Morse cluster global minima: Parallel multi-start with L-BFGS-B polishing."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import concurrent.futures
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START = time.time()
_BUDGET = 295.0
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


def _validate_and_fix(coords):
    """Ensure minimum distance is respected by pushing apart overlapping atoms."""
    n = len(coords)
    if n < 2:
        return coords
    d = cdist(coords, coords)
    np.fill_diagonal(d, np.inf)
    min_d = np.min(d)
    if min_d < 0.5:
        for _ in range(300):
            d = cdist(coords, coords)
            np.fill_diagonal(d, np.inf)
            min_d = np.min(d)
            if min_d >= 0.5:
                break
            idx = np.unravel_index(np.argmin(d), d.shape)
            i, j = idx
            if i == j:
                continue
            diff = coords[i] - coords[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-6:
                diff = np.array([0.01, 0.01, 0.01])
                dist = np.linalg.norm(diff)
            diff /= dist
            coords[i] += 0.01 * diff
            coords[j] -= 0.01 * diff
    return coords


def _morse_lbfgs_objective(rho, flat_x, n):
    """Objective for L-BFGS-B: Morse energy with distance constraint penalty."""
    coords = flat_x.reshape(n, 3)
    d = cdist(coords, coords)
    np.fill_diagonal(d, np.inf)
    min_d = np.min(d)
    if min_d < 0.5:
        # Penalty for violating distance constraint
        penalty = 1e4 * (0.5 - min_d) ** 2
    else:
        penalty = 0.0
    e = np.exp(rho * (1.0 - d))
    energy = np.sum(e * (e - 2.0)) / 2.0
    return energy + penalty


def _morse_lbfgs_gradient(rho, flat_x, n):
    """Gradient for L-BFGS-B."""
    coords = flat_x.reshape(n, 3)
    d = cdist(coords, coords)
    np.fill_diagonal(d, np.inf)
    e = np.exp(rho * (1.0 - d))
    # dE/dr = rho * exp(rho*(1-r)) * (2*exp(rho*(1-r)) - 2)
    de_dr = rho * e * (2.0 * e - 2.0)
    # Gradient w.r.t. coordinates
    grad = np.zeros_like(coords)
    for i in range(n):
        for j in range(i + 1, n):
            rij = coords[i] - coords[j]
            r = np.linalg.norm(rij)
            if r > 1e-10:
                grad[i] += de_dr[i, j] * rij / r
                grad[j] -= de_dr[i, j] * rij / r
    # Add penalty gradient for distance constraint
    min_d = np.min(d)
    if min_d < 0.5:
        idx = np.unravel_index(np.argmin(d), d.shape)
        i, j = idx
        rij = coords[i] - coords[j]
        r = np.linalg.norm(rij)
        if r > 1e-10:
            penalty_grad = -2e4 * (0.5 - min_d) * rij / r
            grad[i] += penalty_grad
            grad[j] -= penalty_grad
    return grad.ravel()


def _lbfgs_polish(rho, x, max_iter=200):
    """Polish using L-BFGS-B with distance constraint."""
    n = len(x)
    flat_x = x.copy().ravel()
    try:
        result = minimize(
            _morse_lbfgs_objective,
            flat_x,
            args=(rho, n),
            method='L-BFGS-B',
            jac=lambda fx: _morse_lbfgs_gradient(rho, fx, n),
            options={'maxiter': max_iter, 'gtol': 1e-8, 'maxls': 50}
        )
        if result.success or result.fun < _morse_energy(rho, x):
            x_new = result.x.reshape(n, 3)
            e_new = _morse_energy(rho, x_new)
            if e_new < 1e8:
                return x_new, e_new
    except Exception:
        pass
    return x, _morse_energy(rho, x)


def _single_optimize(rho, x0, seed_idx):
    """Single optimization chain: relax then L-BFGS polish."""
    rng_local = np.random.default_rng(seed_idx * 7919 + int(rho))
    best_x = x0.copy()
    best_e = _morse_energy(rho, best_x)
    
    # Step 1: Use provided relax
    try:
        e_rel, x_rel = relax(x0, 'morse', rho)
        if e_rel < 1e8:
            best_x = x_rel
            best_e = e_rel
        else:
            x_fixed = _validate_and_fix(x_rel)
            e_rel, x_rel = relax(x_fixed, 'morse', rho)
            if e_rel < 1e8:
                best_x = x_rel
                best_e = e_rel
    except Exception:
        pass
    
    # Step 2: L-BFGS polish
    if _remaining() > 2.0:
        best_x, best_e = _lbfgs_polish(rho, best_x, max_iter=150)
    
    # Step 3: Small perturb + relax + polish
    if _remaining() > 2.0:
        for _ in range(3):
            if _remaining() < 2.0:
                break
            scale = 0.05 + 0.05 * rng_local.random()
            k = max(1, int(0.2 * len(best_x)))
            idx = rng_local.choice(len(best_x), size=k, replace=False)
            x_pert = best_x.copy()
            x_pert[idx] += rng_local.normal(0, scale, (k, 3))
            try:
                e_p, x_p = relax(x_pert, 'morse', rho)
                if e_p < 1e8:
                    x_p, e_p = _lbfgs_polish(rho, x_p, max_iter=100)
                    if e_p < best_e:
                        best_x = x_p
                        best_e = e_p
            except Exception:
                pass
    
    return best_x, best_e


def _generate_seeds(n, rng):
    """Generate diverse initial configurations."""
    seeds = []
    
    # 1. FCC-like packing
    a = 1.0
    pts = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            for k in range(-4, 5):
                pts.append([i*a, j*a, k*a])
                pts.append([(i+0.5)*a, (j+0.5)*a, k*a])
                pts.append([(i+0.5)*a, j*a, (k+0.5)*a])
                pts.append([i*a, (j+0.5)*a, (k+0.5)*a])
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    fcc_seed = pts[order[:n]].copy()
    fcc_seed += rng.normal(0, 0.03, fcc_seed.shape)
    seeds.append(fcc_seed)

    # 2. BCC-like packing
    pts_bcc = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                pts_bcc.append([i*1.0, j*1.0, k*1.0])
                pts_bcc.append([(i+0.5), (j+0.5), (k+0.5)])
    pts_bcc = np.array(pts_bcc)
    pts_bcc -= pts_bcc.mean(axis=0)
    dists_bcc = np.linalg.norm(pts_bcc, axis=1)
    order_bcc = np.argsort(dists_bcc)
    bcc_seed = pts_bcc[order_bcc[:n]].copy()
    bcc_seed += rng.normal(0, 0.03, bcc_seed.shape)
    seeds.append(bcc_seed)

    # 3. Icosahedral shell
    t = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1],
    ], dtype=float)
    ico /= np.linalg.norm(ico[0])
    
    if n <= 12:
        ico_seed = ico[:n].copy()
    else:
        inner_n = n - 12
        inner_r = 0.5
        inner_pts = []
        for _ in range(inner_n):
            while True:
                p = rng.uniform(-inner_r, inner_r, 3)
                if np.linalg.norm(p) < inner_r:
                    break
            inner_pts.append(p)
        inner_pts = np.array(inner_pts)
        ico_seed = np.vstack([ico, inner_pts])
    
    ico_seed += rng.normal(0, 0.03, ico_seed.shape)
    seeds.append(ico_seed)

    # 4. Random Compact
    radius = n ** (1/3) * 0.7
    random_seed = rng.uniform(-radius, radius, size=(n, 3))
    seeds.append(random_seed)

    # 5. Spherical Shell (Fibonacci)
    golden = (1 + np.sqrt(5)) / 2
    shell_seed = []
    r = n ** (1/3) * 0.45
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        rr = np.sqrt(max(0, 1 - z * z))
        theta = 2 * np.pi * i / golden
        shell_seed.append([rr * np.cos(theta), rr * np.sin(theta), z])
    shell_seed = np.array(shell_seed) * r
    shell_seed += rng.normal(0, 0.03, shell_seed.shape)
    seeds.append(shell_seed)

    # 6. Two-Shell Structure
    if n > 10:
        outer_n = max(10, int(n * 0.55))
        inner_n = n - outer_n
        outer_r = 1.1
        inner_r = 0.4
        
        outer_pts = []
        golden = (1 + np.sqrt(5)) / 2
        for i in range(outer_n):
            z = 1 - 2 * (i + 0.5) / outer_n
            rr = np.sqrt(max(0, 1 - z * z))
            theta = 2 * np.pi * i / golden
            outer_pts.append([rr * np.cos(theta), rr * np.sin(theta), z])
        outer_pts = np.array(outer_pts) * outer_r
        
        inner_pts = rng.uniform(-inner_r, inner_r, size=(inner_n, 3))
        two_shell = np.vstack([outer_pts, inner_pts])
        two_shell += rng.normal(0, 0.03, two_shell.shape)
        seeds.append(two_shell)

    # 7. Cubic shell
    if n > 8:
        cube_pts = []
        side = max(2, int(round(n ** (1/3))))
        for i in range(side):
            for j in range(side):
                for k in range(side):
                    cube_pts.append([i - side/2 + 0.5, j - side/2 + 0.5, k - side/2 + 0.5])
        cube_pts = np.array(cube_pts[:n])
        cube_pts += rng.normal(0, 0.03, cube_pts.shape)
        seeds.append(cube_pts)

    # 8. Additional random starts
    for _ in range(5):
        r = n ** (1/3) * 0.6
        seeds.append(rng.uniform(-r, r, size=(n, 3)))

    return seeds


def construct(instance):
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    rng = np.random.default_rng(n * 1000 + int(rho))
    
    # Generate seeds
    seeds = _generate_seeds(n, rng)
    
    # Parallel multi-start optimization
    best_x = None
    best_e = np.inf
    results = []
    
    # Use up to 4 threads
    max_workers = min(4, len(seeds))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i, seed in enumerate(seeds):
            if _remaining() < 5.0:
                break
            f = executor.submit(_single_optimize, rho, seed.copy(), i)
            futures[f] = i
        
        for f in concurrent.futures.as_completed(futures):
            if _remaining() < 3.0:
                break
            try:
                x_res, e_res = f.result(timeout=_remaining() - 2.0)
                if e_res < best_e:
                    best_e = e_res
                    best_x = x_res.copy()
                results.append((x_res, e_res))
            except Exception:
                pass
    
    # If no results, fall back to sequential
    if best_x is None:
        for seed in seeds[:3]:
            try:
                x_res, e_res = _single_optimize(rho, seed, 0)
                if e_res < best_e:
                    best_e = e_res
                    best_x = x_res.copy()
            except Exception:
                continue
    
    if best_x is None:
        return seeds[0]
    
    # Final polish on best result
    if _remaining() > 3.0:
        best_x, best_e = _lbfgs_polish(rho, best_x, max_iter=200)
        
        # Try one more small perturbation
        if _remaining() > 2.0:
            x_pert = best_x.copy()
            k = max(1, int(0.1 * n))
            idx = rng.choice(n, size=k, replace=False)
            x_pert[idx] += rng.normal(0, 0.03, (k, 3))
            try:
                e_p, x_p = relax(x_pert, 'morse', rho)
                if e_p < 1e8:
                    x_p, e_p = _lbfgs_polish(rho, x_p, max_iter=100)
                    if e_p < best_e:
                        best_e = e_p
                        best_x = x_p.copy()
            except Exception:
                pass
    
    # Final validation
    final_e = _morse_energy(rho, best_x)
    if final_e > 1e8:
        r = n ** (1/3) * 0.5
        fallback = rng.uniform(-r, r, size=(n, 3))
        fallback = _validate_and_fix(fallback)
        return fallback
    
    return best_x
# EVOLVE-BLOCK-END
