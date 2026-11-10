"""Combined program: Basin Hopping global search over diverse structural seeds."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START_TIME = time.time()
_DEADLINE = 310.0  # safety margin before 330 s kill


def _remaining():
    return _DEADLINE - time.time()


def _morse_energy(pos, rho):
    """Compute exact Morse energy: sum exp(rho(1-r))(exp(rho(1-r))-2) for all pairs."""
    pos = np.asarray(pos, dtype=float)
    n = pos.shape[0]
    if n < 2:
        return 0.0
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(r, np.inf)
    x = np.exp(rho * (1.0 - r))
    e = x * (x - 2.0)
    iu = np.triu_indices(n, k=1)
    return float(np.sum(e[iu]))


def _morse_grad(pos, rho):
    """Compute gradient of Morse energy."""
    pos = np.asarray(pos, dtype=float)
    n = pos.shape[0]
    diff = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
    r2 = (diff * diff).sum(-1)
    r = np.sqrt(r2)
    np.fill_diagonal(r, np.inf)
    r2[np.eye(n, dtype=bool)] = 1.0  # avoid division by zero
    x = np.exp(rho * (1.0 - r))
    # dE/dr = 2 * rho * exp(rho(1-r)) * (exp(rho(1-r)) - 1)
    # dE/dpos_i = sum_j dE/dr_ij * (pos_i - pos_j) / r_ij
    coeff = 2.0 * rho * x * (x - 1.0) / r
    coeff[np.eye(n, dtype=bool)] = 0.0
    grad = (coeff[:, :, np.newaxis] * diff).sum(1)
    return grad


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
                r2 = i * i + j * j + k * k
                if r2 > maxcoord * maxcoord:
                    continue
                if (i + j + k) % 2 == 0:
                    pts.append([i * a / 2.0, j * a / 2.0, k * a / 2.0])
                else:
                    if i % 2 != 0 and j % 2 != 0 and k % 2 != 0:
                        pts.append([(i - 1) * a / 2.0, (j - 1) * a / 2.0, (k - 1) * a / 2.0])
    pts = np.array(pts, dtype=float)
    d2 = (pts ** 2).sum(1)
    order = np.argsort(d2)
    return pts[order[:n]].copy()


def _hcp_cluster(n, a=1.0):
    """Build a compact HCP cluster of exactly n atoms."""
    pts = []
    maxcoord = int(3.0 * n ** (1.0 / 3.0) + 6.0)
    # HCP layers: ABAB...
    # Simplified generation: generate FCC-like and BCC-like and mix? 
    # Actually, for small n, HCP is often similar to FCC. 
    # Let's use a simple hexagonal packing logic for layers.
    # Layer 1: Hexagonal grid
    # Layer 2: Offset hexagonal grid
    # This is complex to generate exactly n atoms in a compact HCP shape.
    # Given time constraints, we might skip explicit HCP generation and rely on FCC/BCC/Random.
    # Or just return FCC as a fallback for HCP request in this context.
    return _fcc_cluster(n, a)


def _random_cluster(n, rng, spacing=1.0):
    """Random compact cluster with minimum separation."""
    radius = spacing * n ** (1.0 / 3.0) * 0.9
    pts = []
    minsep = 0.55
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
        base = base + rng.normal(0, 0.05, base.shape)
        return base
    return np.array(pts, dtype=float)


def _lbfgs_refine(pos, rho, max_iter=2000):
    """Direct L-BFGS-B refinement with time guard."""
    pos = np.asarray(pos, dtype=float).copy()
    t0 = time.time()
    timeout = 15.0  # max seconds per refinement

    def fe(x):
        if time.time() - t0 > timeout:
            return float("inf")
        e = _morse_energy(x, rho)
        if not np.isfinite(e):
            return 1e9
        return e

    def ge(x):
        if time.time() - t0 > timeout:
            return np.zeros_like(x)
        return _morse_grad(x, rho)

    try:
        res = minimize(fe, pos, jac=ge,
                       method="L-BFGS-B",
                       options={"maxiter": max_iter, "maxls": 40, "ftol": 1e-10})
        if res.x is not None and np.isfinite(_morse_energy(res.x, rho)):
            return np.asarray(res.x, dtype=float)
    except Exception:
        pass
    return pos


def _check_valid(pos, minsep=0.5):
    """Check that all pairs are at least minsep apart and positions are finite."""
    n = pos.shape[0]
    if n < 2:
        return True
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    d2 = (diff * diff).sum(-1)
    np.fill_diagonal(d2, np.inf)
    return np.sqrt(d2.min()) >= minsep - 1e-6 and np.all(np.isfinite(pos))


def _basin_hop(pos, rho, n_steps=20, sigma=0.2, rng=None):
    """
    Perform Basin Hopping starting from pos.
    Returns the best position found.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    pos = np.asarray(pos, dtype=float).copy()
    e = _morse_energy(pos, rho)
    if not np.isfinite(e):
        return pos, e
        
    pos = _lbfgs_refine(pos, rho)
    e = _morse_energy(pos, rho)
    
    best_pos = pos.copy()
    best_e = e
    
    for i in range(n_steps):
        if _remaining() < 10.0:
            break
            
        # Perturb
        noise = rng.normal(0, sigma, pos.shape)
        new_pos = pos + noise
        
        # Relax
        new_pos = _lbfgs_refine(new_pos, rho)
        new_e = _morse_energy(new_pos, rho)
        
        if not np.isfinite(new_e):
            continue
            
        # Accept if better (or with Metropolis, but we just want global min)
        # For global min search, we can just keep the best found so far.
        # However, BH usually uses Metropolis to escape local minima.
        # Since we want the *global* minimum, we should track the best found.
        # But to escape, we might need to accept worse moves.
        # Given the short time, a simple "keep best" might get stuck.
        # Let's use a simple Metropolis acceptance with a decaying temperature.
        
        if new_e < best_e:
            best_pos = new_pos.copy()
            best_e = new_e
            
        # Metropolis step for current position
        T = 0.1 * (1 - i / float(n_steps)) + 0.01
        if new_e < e or np.random.rand() < np.exp(-(new_e - e) / T):
            pos = new_pos
            e = new_e
        else:
            # Stay at current pos, but maybe try a larger step?
            pass
            
    return best_pos, best_e


def construct(instance):
    """Build compact clusters and use Basin Hopping to find global minima."""
    n = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 10.0))

    rng = np.random.default_rng(n)
    
    # Generate diverse seeds
    seeds = []
    seeds.append(_fcc_cluster(n))
    seeds.append(_bcc_cluster(n))
    seeds.append(_random_cluster(n, rng))
    seeds.append(_random_cluster(n, rng))
    
    best_pos = None
    best_e = float("inf")
    
    # Time budget per seed
    # Total time ~50s. 4 seeds. 10s each for BH?
    # BH involves multiple L-BFGS calls. Each L-BFGS might take 1-5s.
    # So 10s per seed allows for ~2-5 BH steps.
    
    for seed in seeds:
        if _remaining() < 20.0:
            break
            
        try:
            # Run Basin Hopping
            # Adjust steps based on remaining time
            steps = 10
            pos, e = _basin_hop(seed, rho, n_steps=steps, sigma=0.2, rng=rng)
            
            if np.isfinite(e) and _check_valid(pos):
                if e < best_e:
                    best_e = e
                    best_pos = pos
        except Exception:
            pass

    if best_pos is not None and _check_valid(best_pos):
        return best_pos

    # Fallback
    fcc = _fcc_cluster(n)
    return fcc
# EVOLVE-BLOCK-END
