"""Morse cluster global minima: Multi-start L-BFGS-B with structural prototypes."""
# EVOLVE-BLOCK-START
import numpy as np
import time
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


def _morse_energy_flat(x, n, rho):
    """Energy for flat array (for scipy minimize)."""
    coords = x.reshape(-1, 3)
    return _morse_energy(rho, coords)


def _morse_grad_flat(x, n, rho):
    """Gradient for flat array (for scipy minimize)."""
    coords = x.reshape(-1, 3)
    try:
        _, grad = morse_energy_gradient(coords, rho)
        return grad.flatten()
    except Exception:
        return np.zeros_like(x)


def _validate_and_fix(coords):
    """Ensure minimum distance is respected by pushing apart overlapping atoms."""
    n = len(coords)
    if n < 2:
        return coords
    d = cdist(coords, coords)
    np.fill_diagonal(d, np.inf)
    min_d = np.min(d)
    if min_d < 0.5:
        for _ in range(200):
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
            coords[i] += 0.005 * diff
            coords[j] -= 0.005 * diff
    return coords


def _generate_structural_seeds(n, rng):
    """Generate physically motivated structural prototypes."""
    seeds = []
    
    # 1. FCC packing
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
    fcc_seed += rng.normal(0, 0.02, fcc_seed.shape)
    seeds.append(fcc_seed)

    # 2. HCP packing
    pts_hcp = []
    for i in range(-3, 4):
        for j in range(-3, 4):
            for k in range(-3, 4):
                pts_hcp.append([i*1.0, j*1.0, k*1.0])
                pts_hcp.append([(i+0.5), (j+0.5), (k+0.5)])
    pts_hcp = np.array(pts_hcp)
    pts_hcp -= pts_hcp.mean(axis=0)
    dists_hcp = np.linalg.norm(pts_hcp, axis=1)
    order_hcp = np.argsort(dists_hcp)
    hcp_seed = pts_hcp[order_hcp[:n]].copy()
    hcp_seed += rng.normal(0, 0.02, hcp_seed.shape)
    seeds.append(hcp_seed)

    # 3. Icosahedral
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
        inner_r = 0.4
        inner_pts = []
        for _ in range(inner_n):
            while True:
                p = rng.uniform(-inner_r, inner_r, 3)
                if np.linalg.norm(p) < inner_r:
                    break
            inner_pts.append(p)
        inner_pts = np.array(inner_pts)
        ico_seed = np.vstack([ico, inner_pts])
    
    ico_seed += rng.normal(0, 0.02, ico_seed.shape)
    seeds.append(ico_seed)

    # 4. Two-shell (Fibonacci outer)
    if n > 10:
        outer_n = max(10, int(n * 0.6))
        inner_n = n - outer_n
        outer_r = 1.0
        inner_r = 0.3
        
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
        two_shell += rng.normal(0, 0.02, two_shell.shape)
        seeds.append(two_shell)

    # 5. Random compact
    radius = n ** (1/3) * 0.6
    random_seed = rng.uniform(-radius, radius, size=(n, 3))
    seeds.append(random_seed)

    # 6. Spherical shell
    golden = (1 + np.sqrt(5)) / 2
    shell_seed = []
    r = n ** (1/3) * 0.4
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        rr = np.sqrt(max(0, 1 - z * z))
        theta = 2 * np.pi * i / golden
        shell_seed.append([rr * np.cos(theta), rr * np.sin(theta), z])
    shell_seed = np.array(shell_seed) * r
    shell_seed += rng.normal(0, 0.02, shell_seed.shape)
    seeds.append(shell_seed)

    return seeds


def _local_minimize_lbfgs(rho, coords, max_iter=200):
    """Local minimization using L-BFGS-B."""
    n = len(coords)
    x0 = coords.flatten()
    
    def func(x):
        return _morse_energy_flat(x, n, rho)
    
    def grad(x):
        return _morse_grad_flat(x, n, rho)
    
    try:
        result = minimize(
            func, x0, jac=grad,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        x_opt = result.x.reshape(-1, 3)
        e_opt = _morse_energy(rho, x_opt)
        if e_opt > 1e8:
            x_opt = _validate_and_fix(x_opt)
            e_opt = _morse_energy(rho, x_opt)
        return x_opt, e_opt
    except Exception:
        return coords, _morse_energy(rho, coords)


def construct(instance):
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    rng = np.random.default_rng(n * 1000 + int(rho))
    
    # Generate structural seeds
    seeds = _generate_structural_seeds(n, rng)
    
    # Add more random perturbations for diversity
    extra_seeds = []
    for _ in range(4):
        r = n ** (1/3) * 0.6
        extra_seeds.append(rng.uniform(-r, r, size=(n, 3)))
    seeds.extend(extra_seeds)
    
    best_x = None
    best_e = np.inf
    
    # Phase 1: Local minimize each seed with L-BFGS-B
    for seed in seeds:
        if _remaining() < 5.0:
            break
        x_opt, e_opt = _local_minimize_lbfgs(rho, seed, max_iter=150)
        if e_opt < best_e:
            best_e = e_opt
            best_x = x_opt.copy()
    
    if best_x is None:
        return seeds[0]
    
    # Phase 2: Basin hopping from the best local minimum
    if _remaining() > 15.0:
        cur_x = best_x.copy()
        cur_e = best_e
        temp = 0.05
        scale = 0.1
        frac = 0.2
        stuck = 0
        
        while _remaining() > 5.0:
            # Perturb
            k = max(1, int(n * frac))
            idx = rng.choice(n, size=k, replace=False)
            noise = rng.normal(0.0, scale, size=(k, 3))
            x_new = cur_x.copy()
            x_new[idx] += noise
            
            # Local minimize
            x_opt, e_opt = _local_minimize_lbfgs(rho, x_new, max_iter=100)
            
            if e_opt > 1e8:
                continue
            
            dE = e_opt - cur_e
            accept = dE < 0 or rng.random() < np.exp(-dE / temp)
            
            if accept:
                cur_x = x_opt
                cur_e = e_opt
                stuck = 0
                if cur_e < best_e:
                    best_e = cur_e
                    best_x = x_opt.copy()
            else:
                stuck += 1
                if stuck > 10:
                    scale *= 1.5
                    frac = min(0.5, frac + 0.05)
                    stuck = 0
            
            temp *= 0.97
    
    # Phase 3: Final high-precision polish
    if _remaining() > 3.0:
        x_polished, e_polished = _local_minimize_lbfgs(rho, best_x, max_iter=300)
        if e_polished < best_e:
            best_e = e_polished
            best_x = x_polished.copy()
    
    # Final validation
    final_e = _morse_energy(rho, best_x)
    if final_e > 1e8:
        r = n ** (1/3) * 0.5
        fallback = rng.uniform(-r, r, size=(n, 3))
        fallback = _validate_and_fix(fallback)
        return fallback
    
    return best_x
# EVOLVE-BLOCK-END
