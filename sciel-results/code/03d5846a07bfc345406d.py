"""Morse cluster global minima: multi-start basin hopping with diverse seeds and L-BFGS refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START_TIME = time.time()
_DEADLINE = 315.0  # safety margin before 330 s kill

def _remaining():
    return _DEADLINE - time.time()

def _morse_energy(pos, rho):
    pos = np.asarray(pos, dtype=float)
    n = pos.shape[0]
    if n < 2:
        return 0.0
    r = np.sqrt(((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(r, 0.0)
    mask = r > 1e-12
    x = np.exp(rho * (1.0 - r[mask]))
    return float((x * (x - 2.0)).sum())

def _fcc_cluster(n, a=1.0):
    """Build a compact FCC cluster of exactly n atoms."""
    pts = []
    maxcoord = int(3.0 * n ** (1.0 / 3.0) + 6.0)
    for i in range(-maxcoord, maxcoord + 1):
        for j in range(-maxcoord, maxcoord + 1):
            for k in range(-maxcoord, maxcoord + 1):
                r2 = i * i + j * j + k * k
                if r2 > maxcoord * maxcoord:
                    continue
                if (i + j + k) % 2 == 0:
                    pts.append([i * a / 2.0, j * a / 2.0, k * a / 2.0])
    pts = np.array(pts, dtype=float)
    d2 = (pts ** 2).sum(1)
    order = np.argsort(d2)
    return pts[order[:n]].copy()

def _bcc_cluster(n, a=1.0):
    """Build a compact BCC cluster of exactly n atoms."""
    pts = []
    maxcoord = int(3.0 * n ** (1.0 / 3.0) + 6.0)
    for i in range(-maxcoord, maxcoord + 1):
        for j in range(-maxcoord, maxcoord + 1):
            for k in range(-maxcoord, maxcoord + 1):
                if (i + j + k) % 2 == 0 or (i == j == k == 0):
                    pts.append([i * a / 2.0, j * a / 2.0, k * a / 2.0])
    pts = np.array(pts, dtype=float)
    d2 = (pts ** 2).sum(1)
    order = np.argsort(d2)
    return pts[order[:n]].copy()

def _hcp_cluster(n, a=1.0):
    """Build a compact HCP cluster."""
    pts = []
    maxcoord = int(3.0 * n ** (1.0 / 3.0) + 6.0)
    for i in range(-maxcoord, maxcoord + 1):
        for j in range(-maxcoord, maxcoord + 1):
            for k in range(-maxcoord, maxcoord + 1):
                x = i * a / 2.0
                y = j * a / 2.0
                z = k * a * np.sqrt(2.0 / 3.0)
                if (i + j + k) % 2 == 0:
                    pts.append([x, y, z])
    pts = np.array(pts, dtype=float)
    d2 = (pts ** 2).sum(1)
    order = np.argsort(d2)
    return pts[order[:n]].copy()

def _icosahedral_shell(n, a=1.0):
    """Build an icosahedral-like cluster with shells."""
    pts = []
    # Core
    pts.append([0.0, 0.0, 0.0])
    # First shell: 12 vertices of icosahedron
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [0, 1, t], [0, 1, -t], [0, -1, t], [0, -1, -t],
        [1, t, 0], [1, -t, 0], [-1, t, 0], [-1, -t, 0],
        [t, 0, 1], [t, 0, -1], [-t, 0, 1], [-t, 0, -1]
    ], dtype=float)
    ico /= np.linalg.norm(ico[0])
    for v in ico:
        pts.append(v * a)
    
    # Second shell: 30 vertices (edges of icosahedron)
    if n > 13:
        # Generate points on a sphere using Fibonacci lattice
        remaining = n - len(pts)
        radius = a * 2.0
        for i in range(remaining):
            phi = np.pi * (3.0 - np.sqrt(5.0))
            y = 1 - (i + 0.5) / remaining * 2
            radius_y = np.sqrt(1 - y**2)
            theta = phi * i
            x = np.cos(theta) * radius_y
            z = np.sin(theta) * radius_y
            pts.append([x * radius, y * radius, z * radius])
    
    return np.array(pts[:n], dtype=float)

def _random_cluster(n, rng, spacing=1.0):
    """Random compact cluster with minimum separation."""
    radius = spacing * n ** (1.0 / 3.0) * 1.3
    pts = []
    minsep = 0.51
    maxtries = 80000
    t = 0
    while len(pts) < n and t < maxtries:
        t += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if all(np.linalg.norm(p - q) >= minsep for q in pts):
            pts.append(p)
    if len(pts) < n:
        base = _fcc_cluster(n)
        base = base + rng.normal(0, 0.1, base.shape)
        return base
    return np.array(pts, dtype=float)

def _gas_cluster(n, rng, spacing=1.0):
    """High-temperature gas-like initial configuration."""
    radius = spacing * n ** (1.0 / 3.0) * 2.0
    pts = []
    minsep = 0.51
    maxtries = 100000
    t = 0
    while len(pts) < n and t < maxtries:
        t += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if all(np.linalg.norm(p - q) >= minsep for q in pts):
            pts.append(p)
    if len(pts) < n:
        base = _fcc_cluster(n)
        base = base + rng.normal(0, 0.3, base.shape)
        return base
    return np.array(pts, dtype=float)

def _check_valid(pos, minsep=0.5):
    n = pos.shape[0]
    if n == 0:
        return False
    d2 = ((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    min_d = np.sqrt(d2.min())
    return min_d >= minsep - 1e-6 and np.all(np.isfinite(pos))

def _lbfgs_refine(pos, potential, rho, maxiter=100):
    """Use L-BFGS-B for local refinement."""
    from scipy.optimize import minimize
    
    def obj(x):
        return _morse_energy(x, rho)
    
    def grad(x):
        e, g = morse_energy_gradient(x, rho)
        return g
    
    try:
        res = minimize(obj, pos, method='L-BFGS-B', jac=grad, 
                       options={'maxiter': maxiter, 'gtol': 1e-8})
        if res.success and _check_valid(res.x):
            return res.x, float(res.fun)
    except Exception:
        pass
    return pos, _morse_energy(pos, rho)

def _basin_hopping(pos, potential, rho, n_trials=20, step_scale=0.5):
    """Basin hopping: random kicks followed by local minimization."""
    best_pos = pos.copy()
    best_e = _morse_energy(pos, rho)
    rng = np.random.default_rng(hash(tuple(pos.ravel())) % 2**32)
    
    for trial in range(n_trials):
        if _remaining() < 15.0:
            break
        # Random perturbation
        kick = rng.normal(0, step_scale, pos.shape)
        new_pos = pos + kick
        # Ensure minimum separation
        if not _check_valid(new_pos, 0.45):
            continue
        # Local minimization
        try:
            e, relaxed = relax(new_pos, potential, rho)
            e = float(e)
            if np.isfinite(e) and _check_valid(relaxed):
                if e < best_e:
                    best_e = e
                    best_pos = relaxed.copy()
                    pos = relaxed.copy()
        except Exception:
            pass
    
    return best_pos, best_e

def construct(instance):
    """Build diverse initial structures and use basin hopping to find global minimum."""
    n = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 10.0))
    
    seeds = []
    rng = np.random.default_rng(n * 1000 + int(rho))
    
    # 1. FCC
    seeds.append(_fcc_cluster(n))
    # 2. BCC
    seeds.append(_bcc_cluster(n))
    # 3. HCP
    seeds.append(_hcp_cluster(n))
    # 4. Icosahedral
    seeds.append(_icosahedral_shell(n))
    # 5. Jittered FCC
    seeds.append(_fcc_cluster(n) + rng.normal(0, 0.05, (n, 3)))
    # 6. Jittered BCC
    seeds.append(_bcc_cluster(n) + rng.normal(0, 0.05, (n, 3)))
    # 7. Random compact
    seeds.append(_random_cluster(n, rng))
    # 8. Gas-like
    seeds.append(_gas_cluster(n, rng))
    
    best_pos = None
    best_e = float("inf")
    
    def _process_seed(seed, idx):
        try:
            if _remaining() < 20.0:
                return None, float("inf")
            # First, do a quick relax
            e, pos = relax(np.asarray(seed, dtype=float), potential, rho)
            e = float(e)
            if not (np.isfinite(e) and _check_valid(pos)):
                return None, float("inf")
            
            # Then basin hopping with limited trials
            n_trials = max(5, int(20 * _remaining() / 100.0))
            pos, e = _basin_hopping(pos, potential, rho, n_trials=n_trials, 
                                    step_scale=0.3 + 0.2 * idx)
            
            # Final L-BFGS polish
            pos, e = _lbfgs_refine(pos, potential, rho, maxiter=50)
            
            if np.isfinite(e) and _check_valid(pos):
                return pos, e
            return None, float("inf")
        except Exception:
            return None, float("inf")
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_process_seed, s, i): i for i, s in enumerate(seeds)}
        for future in as_completed(futures):
            if _remaining() < 10.0:
                break
            try:
                pos, e = future.result()
                if pos is not None and e < best_e:
                    best_e = e
                    best_pos = pos
            except Exception:
                pass
    
    if best_pos is not None:
        return best_pos
    
    # Fallback: return the best seed (unrelaxed) if relax failed completely
    return seeds[0]

# EVOLVE-BLOCK-END
