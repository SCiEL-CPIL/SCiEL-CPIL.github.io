"""Basin-hopping global search over compact structural seeds (FCC/BCC/HCP/random).

Each instance gets an equal slice of the 300 s wall-clock budget. The inner
minimizer is L-BFGS-B on the exact Morse energy; the outer loop is a
simulated-annealing basin-hopping schedule that perturbs the current minimum
and accepts worse basins with a Boltzmann probability.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START_TIME = time.time()
_DEADLINE = 310.0  # safety margin before 330 s kill
_INSTANCES = 6
_PER_INSTANCE = (_DEADLINE - _START_TIME - 10.0) / _INSTANCES
_instance_idx = 0


def _remaining():
    return _DEADLINE - time.time()


def _instance_deadline():
    """Return the wall-clock deadline for the current instance (equal slices)."""
    global _instance_idx
    d = _START_TIME + (_instance_idx + 1) * _PER_INSTANCE
    # never exceed the global deadline, and leave a 2 s margin
    return min(d, _DEADLINE) - 2.0


def _advance_instance():
    global _instance_idx
    _instance_idx += 1


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
    """Build a compact FCC cluster of exactly n atoms (lattice constant a)."""
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


def _hcp_cluster(n, a=1.0):
    """Build a compact HCP cluster of exactly n atoms (lattice constant a)."""
    # HCP basis: A at (0,0,0), B at (a/2, a*sqrt(3)/2, 0)
    # Layer spacing c = a * sqrt(8/3)
    c = a * np.sqrt(8.0 / 3.0)
    a2 = a * a
    s3 = np.sqrt(3.0)
    pts = []
    # number of layers to generate
    n_layers = int(2.0 * n ** (1.0 / 3.0) + 4)
    # in-plane radius to cover n atoms
    r_in = int(np.sqrt(n / (2.0 * n_layers)) + 3)
    for layer in range(n_layers):
        z = layer * c
        # AB stacking: even layers use A sublattice, odd use B
        offset = 0.0 if layer % 2 == 0 else a / 2.0
        offset_y = 0.0 if layer % 2 == 0 else a * s3 / 2.0
        for i in range(-r_in, r_in + 1):
            for j in range(-r_in, r_in + 1):
                # triangular lattice in-plane
                x = i * a + (0.5 * a if j % 2 != 0 else 0.0) + offset
                y = j * a * s3 / 2.0 + offset_y
                pts.append([x, y, z])
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
                # BCC: corners and body centers
                if (i + j + k) % 2 == 0:
                    pts.append([i * a / 2.0, j * a / 2.0, k * a / 2.0])
                else:
                    # body center of cube [i-1,i]x[j-1,j]x[k-1,k] if i,j,k all odd
                    if i % 2 != 0 and j % 2 != 0 and k % 2 != 0:
                        pts.append([(i - 1) * a / 2.0, (j - 1) * a / 2.0, (k - 1) * a / 2.0])
    pts = np.array(pts, dtype=float)
    d2 = (pts ** 2).sum(1)
    order = np.argsort(d2)
    return pts[order[:n]].copy()


def _random_cluster(n, rng, spacing=1.0):
    """Random compact cluster with minimum separation."""
    radius = spacing * n ** (1.0 / 3.0) * 0.9
    pts = []
    minsep = 0.55
    maxtries = 200000
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


def _lbfgs_refine(pos, rho, deadline, max_iter=2000):
    """L-BFGS-B refinement bounded by an absolute wall-clock deadline."""
    pos = np.asarray(pos, dtype=float).copy()
    t0 = time.time()

    def fe(x):
        if time.time() > deadline:
            return float("inf")
        e = _morse_energy(x, rho)
        if not np.isfinite(e):
            return 1e9
        return e

    def ge(x):
        if time.time() > deadline:
            return np.zeros_like(x)
        return _morse_grad(x, rho)

    try:
        res = minimize(fe, pos, jac=ge,
                       method="L-BFGS-B",
                       options={"maxiter": max_iter, "maxls": 30,
                                "ftol": 1e-9, "gtol": 1e-7})
        if res.x is not None:
            e = _morse_energy(res.x, rho)
            if np.isfinite(e):
                return np.asarray(res.x, dtype=float)
    except Exception:
        pass
    return pos


def _basin_hop(pos, rho, deadline, rng):
    """Simulated-annealing basin-hopping starting from pos.

    Returns the best (energy, position) found before the deadline.
    """
    pos = np.asarray(pos, dtype=float).copy()
    e = _morse_energy(pos, rho)
    if not np.isfinite(e):
        e = 1e9

    best_e = e
    best_pos = pos.copy()

    # annealing schedule
    T0 = 2.0
    T_min = 1e-3
    # number of steps we can afford: estimate per-step cost ~ 0.05 s for n<=80
    # but we just loop until deadline
    step = 0
    # perturbation amplitude: scale with cluster size so we move ~1 bond length
    # total displacement per step ~ 0.3-0.5 in reduced units
    amp0 = 0.3
    amp_min = 0.05

    while time.time() < deadline - 2.0:
        step += 1
        # current temperature
        frac = min(1.0, step / 200.0)
        T = T0 * (T_min / T0) ** frac
        # current perturbation amplitude
        amp = amp0 * (amp_min / amp0) ** frac

        # perturb: random Gaussian displacement
        pert = pos + rng.normal(0.0, amp, size=pos.shape)
        # ensure finite
        if not np.all(np.isfinite(pert)):
            continue

        # inner L-BFGS minimization (short budget)
        inner_deadline = min(deadline - 1.0, time.time() + 3.0)
        cand = _lbfgs_refine(pert, rho, inner_deadline, max_iter=1500)
        cand_e = _morse_energy(cand, rho)
        if not np.isfinite(cand_e):
            continue

        # acceptance criterion
        if cand_e < e:
            accept = True
        else:
            dE = cand_e - e
            if dE < 1e-12:
                accept = True
            else:
                # Boltzmann acceptance
                log_p = -dE / T
                if log_p > 50.0:
                    accept = True
                else:
                    accept = (rng.random() < np.exp(log_p))

        if accept:
            pos = cand
            e = cand_e
            if e < best_e:
                best_e = e
                best_pos = pos.copy()
        else:
            # rejected: keep current pos/e, but still count the step for annealing
            pass

    return best_e, best_pos


def _check_valid(pos, minsep=0.5):
    """Check that all pairs are at least minsep apart and positions are finite."""
    n = pos.shape[0]
    if n < 2:
        return True
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    d2 = (diff * diff).sum(-1)
    np.fill_diagonal(d2, np.inf)
    return np.sqrt(d2.min()) >= minsep - 1e-6 and np.all(np.isfinite(pos))


def construct(instance):
    """Basin-hopping global search over compact structural seeds.

    Each instance gets an equal slice of the 300 s wall-clock budget.
    """
    _advance_instance()
    n = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 10.0))

    deadline = min(_instance_deadline(), _remaining())

    rng = np.random.default_rng(n * 7919 + int(rho))

    # Build diverse structural seeds
    seeds = []
    for builder in (_fcc_cluster, _bcc_cluster, _hcp_cluster):
        try:
            s = builder(n, a=1.0)
            if _check_valid(s):
                seeds.append(s)
        except Exception:
            pass
    # Add a couple of random seeds for diversity
    for k in range(2):
        try:
            s = _random_cluster(n, rng)
            if _check_valid(s):
                seeds.append(s)
        except Exception:
            pass

    if not seeds:
        seeds = [_fcc_cluster(n, a=1.0)]

    best_e = float("inf")
    best_pos = None

    # Phase 1: quick L-BFGS relaxation of each seed to establish baselines
    for s in seeds:
        if time.time() > deadline - 5.0:
            break
        try:
            ref = _lbfgs_refine(s, rho, deadline, max_iter=1500)
            e = _morse_energy(ref, rho)
            if np.isfinite(e) and _check_valid(ref):
                if e < best_e:
                    best_e = e
                    best_pos = ref.copy()
        except Exception:
            pass

    # Phase 2: basin-hopping from the best seed(s)
    # Use the best seed as the starting point; also try one random seed if time permits
    if best_pos is not None:
        bh_e, bh_pos = _basin_hop(best_pos, rho, deadline, rng)
        if np.isfinite(bh_e) and _check_valid(bh_pos):
            if bh_e < best_e:
                best_e = bh_e
                best_pos = bh_pos

    # If we still have time, basin-hop from a different seed for diversity
    if len(seeds) > 1 and time.time() < deadline - 5.0:
        alt_seed = seeds[-1]  # a random seed
        bh_e2, bh_pos2 = _basin_hop(alt_seed, rho, deadline, rng)
        if np.isfinite(bh_e2) and _check_valid(bh_pos2):
            if bh_e2 < best_e:
                best_e = bh_e2
                best_pos = bh_pos2

    if best_pos is not None and _check_valid(best_pos):
        return best_pos

    # Fallback: best relaxed seed or raw FCC
    for s in seeds:
        if _check_valid(s):
            return s
    return _fcc_cluster(n, a=1.0)
# EVOLVE-BLOCK-END
