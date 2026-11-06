"""Basin-hopping for Morse cluster global minima with diverse structural seeds.

Strategy:
  1. Generate diverse structural seeds (FCC, BCC, HCP, icosahedral shells, random).
  2. Run multiple parallel basin-hopping chains with adaptive perturbation.
  3. Periodic re-seeding from the global best to escape local minima.
  4. Return the best configuration found within the time budget.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np
import concurrent.futures as _cf
from scipy.optimize import minimize
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_T0 = None
_HARD_DEADLINE = None


def _init_timer():
    global _T0, _HARD_DEADLINE
    if _T0 is None:
        _T0 = time.time()
        _HARD_DEADLINE = _T0 + 285.0
    return _T0


def _morse_E(P, rho):
    """Exact energy E = sum_{i<j} exp(rho(1-r))(exp(rho(1-r))-2)."""
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    D = P[:, None, :] - P[None, :, :]
    r = np.linalg.norm(D, axis=2)
    iu = np.triu_indices(n, k=1)
    r = r[iu]
    t = np.exp(rho * (1.0 - r))
    return float((t * (t - 2.0)).sum())


def _morse_grad(P, rho):
    """Gradient of exact Morse energy w.r.t. coordinates (n,3)."""
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    D = P[:, None, :] - P[None, :, :]
    r = np.linalg.norm(D, axis=2)
    np.fill_diagonal(r, np.inf)
    t = np.exp(rho * (1.0 - r))
    c = rho * t * (2.0 * t - 2.0)
    inv_r = np.where(r > 1e-12, 1.0 / r, 0.0)
    W = c * inv_r
    rowsum = W.sum(axis=1)
    grad = -(rowsum[:, None] * P - W @ P)
    return grad


def _enforce_min_dist(P, min_d=0.5):
    """Robustly nudge so every pair is >= min_d apart."""
    P = np.array(P, dtype=float, copy=True)
    n = P.shape[0]
    for _ in range(500):
        D = P[:, None, :] - P[None, :, :]
        d = np.linalg.norm(D, axis=2)
        bad = (d < min_d)
        np.fill_diagonal(bad, False)
        if not bad.any():
            break
        idx = np.argwhere(bad)
        if len(idx) == 0:
            break
        i, j = idx[0]
        gap = min_d - d[i, j] + 0.01
        v = P[i] - P[j]
        nv = np.linalg.norm(v)
        if nv < 1e-9:
            v = np.array([1.0, 0.0, 0.0])
            nv = 1.0
        v = v / nv
        P[i] += 0.5 * gap * v
        P[j] -= 0.5 * gap * v
    return P


def _local_opt(P0, rho, max_iter=300, ftol=1e-10):
    """L-BFGS-B on exact Morse energy; returns (E, P)."""
    P0 = np.asarray(P0, dtype=float)
    n = P0.shape[0]

    def fun(x):
        P = x.reshape(n, 3)
        return _morse_E(P, rho)

    def gfun(x):
        P = x.reshape(n, 3)
        return _morse_grad(P, rho).ravel()

    try:
        res = minimize(fun, P0.ravel(), jac=gfun, method="L-BFGS-B",
                       options={"maxiter": max_iter, "maxfun": 10000,
                                "ftol": ftol, "gtol": 1e-7, "iprint": 0})
        P = res.x.reshape(n, 3)
        E = _morse_E(P, rho)
        return E, P
    except Exception:
        return None, None


def _fcc_points(n, a=1.0):
    """Generate FCC lattice points near origin."""
    R = n ** (1.0 / 3.0) * 1.5
    basis = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0]
    ])
    pts = []
    extent = int(np.ceil(R / a)) + 2
    for i in range(-extent, extent + 1):
        for j in range(-extent, extent + 1):
            for k in range(-extent, extent + 1):
                for b in basis:
                    p = np.array([i, j, k]) * a + b * a
                    if np.linalg.norm(p) <= R:
                        pts.append(p)
    if len(pts) < n:
        return np.array([])
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order][:n]
    return pts


def _bcc_points(n, a=1.0):
    """Generate BCC lattice points near origin."""
    R = n ** (1.0 / 3.0) * 1.5
    basis = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5]
    ])
    pts = []
    extent = int(np.ceil(R / a)) + 2
    for i in range(-extent, extent + 1):
        for j in range(-extent, extent + 1):
            for k in range(-extent, extent + 1):
                for b in basis:
                    p = np.array([i, j, k]) * a + b * a
                    if np.linalg.norm(p) <= R:
                        pts.append(p)
    if len(pts) < n:
        return np.array([])
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order][:n]
    return pts


def _hcp_points(n, a=1.0):
    """Generate HCP lattice points near origin."""
    R = n ** (1.0 / 3.0) * 1.5
    c = a * np.sqrt(8.0 / 3.0)
    a1 = np.array([a, 0, 0])
    a2 = np.array(-a / 2, a * np.sqrt(3) / 2, 0)
    a3 = np.array(0, 0, c)
    basis = np.array([
        [0.0, 0.0, 0.0],
        [1.0 / 3.0, 2.0 / 3.0, 0.5]
    ])
    pts = []
    extent = int(np.ceil(R / min(a, c))) + 2
    for i in range(-extent, extent + 1):
        for j in range(-extent, extent + 1):
            for k in range(-extent, extent + 1):
                for b in basis:
                    p = i * a1 + j * a2 + k * a3 + np.array(b) * np.array([a, a, c])
                    if np.linalg.norm(p) <= R:
                        pts.append(p)
    if len(pts) < n:
        return np.array([])
    pts = np.array(pts)
    dists = np.linalg.norm(pts, axis=1)
    order = np.argsort(dists)
    pts = pts[order][:n]
    return pts


def _icosahedral_shells(n):
    """Generate icosahedral shell points."""
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts = verts / np.linalg.norm(verts, axis=1, keepdims=True)

    # Build shells
    shells = [np.array([[0.0, 0.0, 0.0]])]
    remaining = n - 1

    # First shell: 12 vertices
    if remaining >= 12:
        shells.append(verts.copy())
        remaining -= 12

    # Second shell: 30 vertices (midpoints of edges)
    if remaining >= 30:
        edges = []
        # Generate all edges of icosahedron (pairs at distance sqrt(2) in unit form)
        for i in range(12):
            for j in range(i + 1, 12):
                d = np.linalg.norm(verts[i] - verts[j])
                if 1.0 < d < 1.5:
                    edges.append((verts[i] + verts[j]) / 2)
        shell2 = np.array(edges)
        shell2 = shell2 / np.linalg.norm(shell2, axis=1, keepdims=True) * 1.7
        if len(shell2) >= 30:
            shells.append(shell2[:30])
            remaining -= 30

    # Fill remaining with scaled vertices
    while remaining > 0:
        scale = 2.5 + 0.5 * len(shells)
        extra = verts * scale
        take = min(remaining, 12)
        shells.append(extra[:take])
        remaining -= take

    all_pts = np.vstack(shells)
    if len(all_pts) > n:
        all_pts = all_pts[:n]
    elif len(all_pts) < n:
        # Fill with random points
        rng = np.random.RandomState(42)
        extra = rng.normal(0, 2.0, size=(n - len(all_pts), 3))
        all_pts = np.vstack([all_pts, extra])
    return all_pts


def _random_compact(n, seed=None):
    """Random compact cluster."""
    rng = np.random.RandomState(seed)
    R = n ** (1.0 / 3.0) * 1.5
    P = rng.uniform(-R, R, size=(n, 3))
    return P


def _generate_seeds(n, rho):
    """Generate diverse structural seeds."""
    seeds = []

    # FCC with various lattice constants
    for a in [0.85, 0.95, 1.05, 1.15]:
        p = _fcc_points(n, a)
        if len(p) == n:
            seeds.append(p)

    # BCC with various lattice constants
    for a in [0.85, 0.95, 1.05]:
        p = _bcc_points(n, a)
        if len(p) == n:
            seeds.append(p)

    # HCP
    for a in [0.9, 1.0]:
        p = _hcp_points(n, a)
        if len(p) == n:
            seeds.append(p)

    # Icosahedral shells
    p = _icosahedral_shells(n)
    seeds.append(p)

    # Random compact clusters
    for s in [42, 123, 456, 789, 1011, 2022]:
        seeds.append(_random_compact(n, seed=s))

    return seeds


def _basin_hopping_chain(P0, rho, n, deadline, rng_seed, global_best_tracker):
    """
    Run a basin-hopping chain with adaptive step size and periodic re-seeding.
    """
    rng = np.random.RandomState(rng_seed)

    # Initial local optimization
    P0 = _enforce_min_dist(P0)
    E, P = _local_opt(P0, rho, max_iter=200)
    if E is None:
        E, P = 1e10, P0

    best_E = E
    best_P = P.copy()

    # Adaptive perturbation scale
    step = 0.35
    stagnant_iters = 0
    max_stagnant = 25

    it = 0
    max_iters = 150

    while it < max_iters:
        if time.time() > deadline - 2.0:
            break

        # Periodic re-seeding from global best
        if stagnant_iters > max_stagnant and global_best_tracker.get('best_P') is not None:
            gp = global_best_tracker['best_P']
            noise = rng.normal(0, 0.35, size=gp.shape)
            P = _enforce_min_dist(gp + noise)
            E, P = _local_opt(P, rho, max_iter=120)
            if E is not None:
                if E < best_E:
                    best_E = E
                    best_P = P.copy()
                stagnant_iters = 0
            else:
                E, P = 1e10, P

        # Perturb
        noise = rng.normal(0, step, size=P.shape)
        P_new = P + noise
        P_new = _enforce_min_dist(P_new)

        # Local optimization
        E_new, P_new = _local_opt(P_new, rho, max_iter=120)
        if E_new is None:
            E_new, P_new = 1e10, P

        # Track best
        if E_new < best_E:
            best_E = E_new
            best_P = P_new.copy()
            stagnant_iters = 0
        else:
            stagnant_iters += 1

        # Update global best tracker
        if E_new < global_best_tracker.get('best_E', float('inf')):
            global_best_tracker['best_E'] = E_new
            global_best_tracker['best_P'] = P_new.copy()

        # Metropolis acceptance with adaptive temperature
        T = 0.25 * (0.003 / 0.25) ** (it / max(1, max_iters - 1))
        if E_new <= E or rng.random() < np.exp(-(E_new - E) / max(T, 1e-12)):
            E, P = E_new, P_new
            step = min(step * 1.03, 0.7)
        else:
            step = max(step * 0.94, 0.08)

        it += 1

    return best_E, best_P


def _run_chain_wrapper(args):
    """Wrapper for parallel execution."""
    P0, rho, n, deadline, rng_seed, global_best_tracker = args
    try:
        return _basin_hopping_chain(P0, rho, n, deadline, rng_seed, global_best_tracker)
    except Exception:
        return None, None


def construct(instance):
    _init_timer()
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind"))

    call_start = time.time()
    deadline = min(_HARD_DEADLINE, call_start + 42.0)

    best_E = float('inf')
    best_P = None

    # Generate diverse seeds
    seeds = _generate_seeds(n, rho)

    # Quick local optimization on all seeds to get baseline
    seed_results = []
    for i, s in enumerate(seeds):
        if time.time() > deadline - 12.0:
            break
        P = _enforce_min_dist(s)
        E, P = _local_opt(P, rho, max_iter=80)
        if E is not None:
            seed_results.append((E, P))
            if E < best_E:
                best_E = E
                best_P = P

    # Sort seeds by energy
    seed_results.sort(key=lambda x: x[0])
    top_seeds = seed_results[:8] if len(seed_results) >= 8 else seed_results

    # Shared tracker for global best across chains
    global_best_tracker = {
        'best_E': best_E,
        'best_P': best_P.copy() if best_P is not None else None
    }

    # Run basin hopping chains in parallel (up to 4 threads)
    n_chains = min(4, len(top_seeds))
    if n_chains > 0 and time.time() < deadline - 10.0:
        chain_args = []
        for i in range(n_chains):
            P0 = top_seeds[i][1]
            chain_args.append((P0, rho, n, deadline, 5000 + i * 7, global_best_tracker))

        with _cf.ThreadPoolExecutor(max_workers=n_chains) as pool:
            futures = [pool.submit(_run_chain_wrapper, args) for args in chain_args]

            for f in futures:
                try:
                    E, P = f.result(timeout=2.0)
                    if E is not None and E < best_E:
                        best_E = E
                        best_P = P
                except Exception:
                    pass

        # Update from global tracker
        if global_best_tracker['best_E'] < best_E:
            best_E = global_best_tracker['best_E']
            best_P = global_best_tracker['best_P']

    # If we still have time, run additional chains from next-best seeds
    if len(top_seeds) > n_chains and time.time() < deadline - 8.0:
        extra_seeds = top_seeds[n_chains:n_chains + 4]
        if extra_seeds:
            extra_args = []
            for i, (E_s, P_s) in enumerate(extra_seeds):
                extra_args.append((P_s, rho, n, deadline, 9000 + i * 13, global_best_tracker))

            n_extra = min(4, len(extra_args))
            if n_extra > 0:
                with _cf.ThreadPoolExecutor(max_workers=n_extra) as pool:
                    futures = [pool.submit(_run_chain_wrapper, args) for args in extra_args[:n_extra]]
                    for f in futures:
                        try:
                            E, P = f.result(timeout=2.0)
                            if E is not None and E < best_E:
                                best_E = E
                                best_P = P
                        except Exception:
                            pass

                if global_best_tracker['best_E'] < best_E:
                    best_E = global_best_tracker['best_E']
                    best_P = global_best_tracker['best_P']

    # Final polish on the best configuration
    if best_P is not None and time.time() < deadline - 3.0:
        E, P = _local_opt(_enforce_min_dist(best_P), rho, max_iter=400, ftol=1e-12)
        if E is not None and E < best_E:
            best_E = E
            best_P = P

    # Fallback if nothing worked
    if best_P is None:
        best_P = _random_compact(n, seed=42)
        best_E = _morse_E(best_P, rho)

    # Final validity check
    best_P = _enforce_min_dist(best_P)

    # One last quick relaxation
    if time.time() < deadline - 1.0:
        E, P = _local_opt(best_P, rho, max_iter=50)
        if E is not None:
            best_P = P

    best_P = _enforce_min_dist(best_P)

    return best_P

# EVOLVE-BLOCK-END
