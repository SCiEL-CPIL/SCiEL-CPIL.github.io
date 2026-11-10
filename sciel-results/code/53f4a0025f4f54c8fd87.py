"""Morse cluster global minima via basin hopping with a wall-clock deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START = time.perf_counter()
_DEADLINE = _START + 295.0  # 300s limit, 5s safety margin


def _time_left():
    return _DEADLINE - time.perf_counter()


def _energy(pos, rho):
    """Vectorized Morse energy; returns +inf if any pair is closer than 0.5."""
    n = len(pos)
    if n < 2:
        return 0.0
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=2))
    iu = np.triu_indices(n, 1)
    d = dist[iu]
    if d.min() < 0.5:
        return float('inf')
    x = rho * (1.0 - d)
    ex = np.exp(x)
    return float(np.sum(ex * (ex - 2.0)))


def _local_relax(pos, rho):
    """Gradient-based local minimization with a minimum-separation guard."""
    try:
        res = minimize(_energy, pos, jac=morse_energy_gradient, args=(rho,),
                       method='L-BFGS-B',
                       options={'maxiter': 600, 'maxls': 50, 'ftol': 1e-10,
                                'gtol': 1e-6, 'maxfun': 20000})
        p = res.x
    except Exception:
        p = pos
    if _energy(p, rho) == float('inf'):
        p = pos
    return p, _energy(p, rho)


def _separate(pos, n, spacing, rng, max_iter=50000):
    """Push apart any pair closer than spacing until all are separated."""
    pos = pos.copy()
    for _ in range(max_iter):
        tree = cKDTree(pos)
        pairs = tree.query_pairs(spacing, output_type='ndarray')
        if len(pairs) == 0:
            return pos
        moved = False
        for i, j in pairs:
            v = pos[i] - pos[j]
            d = np.linalg.norm(v)
            if d < 1e-8:
                v = rng.normal(size=3)
                d = np.linalg.norm(v)
            push = (spacing - d) * 0.5
            pos[i] += (v / d) * push
            pos[j] -= (v / d) * push
            moved = True
        if not moved:
            break
    return pos


def _gen_seed(n, rho, rng):
    """Random packing of n atoms with a minimum separation, then separated."""
    spacing = 0.55
    R = spacing * (n / (np.pi / 6.0)) ** (1.0 / 3.0) * 1.15
    pts = []
    tries = 0
    max_tries = 400 * n
    while len(pts) < n and tries < max_tries:
        tries += 1
        p = rng.uniform(-R, R, 3)
        if np.linalg.norm(p) > R:
            continue
        if not pts:
            pts.append(p)
            continue
        arr = np.array(pts)
        d = np.min(np.linalg.norm(arr - p, axis=1))
        if d >= spacing:
            pts.append(p)
    if len(pts) < n:
        # Fallback: place remaining in a spiral
        while len(pts) < n:
            k = len(pts)
            r = spacing * np.sqrt(k) * 1.1
            a = k * 2.399963
            pts.append([r * np.cos(a), r * np.sin(a), (k % 5 - 2) * spacing * 0.6])
    pos = np.array(pts)
    pos = _separate(pos, n, spacing, rng)
    return pos


def construct(instance):
    """Basin hopping over random seeds, best-of-many, with a wall-clock guard."""
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))
    potential = instance["potential"]
    rng = np.random.default_rng(n * 1000003 + int(rho) * 101)

    best_pos = None
    best_e = float('inf')

    def consider(pos, e):
        nonlocal best_pos, best_e
        if e < best_e:
            best_e = e
            best_pos = pos

    # Seed #0: random packing, relaxed
    if _time_left() > 0.5:
        s = _gen_seed(n, rho, rng)
        p, e = _local_relax(s, rho)
        consider(p, e)

    # Basin hopping from the best seed
    # Perturbation amplitude tuned to n (reduced units, spacing ~1)
    T = 0.15
    step = 0.6
    it = 0
    while _time_left() > 1.0:
        # perturb the current best
        noise = rng.normal(0.0, step, best_pos.shape)
        cand = best_pos + noise
        cand = _separate(cand, n, 0.5, rng, max_iter=4000)
        p, e = _local_relax(cand, rho)
        it += 1
        if e < best_e:
            consider(p, e)
            # accept downhill; occasionally restart perturbation size
            step = max(0.3, step * 0.9)
        else:
            # Metropolis acceptance to escape local minima
            dE = e - best_e
            if dE < 0 or rng.random() < np.exp(-dE / T):
                # move the basin-hopping walker to this (higher) minimum
                best_pos = p
                step = min(1.5, step * 1.05)
            else:
                step = min(1.5, step * 1.03)

        # Periodically inject a fresh random seed to diversify
        if it % 8 == 0 and _time_left() > 2.0:
            s = _gen_seed(n, rho, rng)
            p2, e2 = _local_relax(s, rho)
            if e2 < best_e:
                consider(p2, e2)

    # Final polish of the best with the provided relax helper (exact recompute path)
    if best_pos is not None and _time_left() > 0.5:
        try:
            e_final, pos_final = relax(best_pos, potential, rho)
            if e_final < best_e:
                consider(pos_final, e_final)
        except Exception:
            pass

    return best_pos
# EVOLVE-BLOCK-END
