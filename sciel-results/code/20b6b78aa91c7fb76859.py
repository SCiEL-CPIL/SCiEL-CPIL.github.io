"""Lennard-Jones cluster global minimum, 38 atoms.

Strategy:
  1. Build a high-quality initial guess: FCC truncated-octahedron fragment
     (the known global minimum geometry) plus a few alternative seeds.
  2. For each seed, run basin-hopping with analytic gradients and L-BFGS.
  3. Enforce minimum separation of 0.5 throughout.
  4. Final high-precision polish on the best configuration.
  5. Deadline guard: return best found before 150 s wall-clock.
"""

import time
import numpy as np
from scipy.optimize import minimize

N = 38
DEADLINE_SECONDS = 145.0  # safety margin below 150 s


def _lj_energy(pos):
    """Total LJ energy 4*sum(r^-12 - r^-6) over all pairs."""
    n = len(pos)
    e = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            r2 = np.dot(pos[i] - pos[j], pos[i] - pos[j])
            if r2 < 1e-12:
                return 1e30
            inv_r2 = 1.0 / r2
            inv_r6 = inv_r2 ** 3
            e += 4.0 * (inv_r6 * inv_r6 - inv_r6)
    return e


def _lj_grad(pos):
    """Analytic gradient of LJ energy w.r.t. each atom position."""
    n = len(pos)
    grad = np.zeros_like(pos)
    for i in range(n):
        for j in range(i + 1, n):
            d = pos[i] - pos[j]
            r2 = np.dot(d, d)
            if r2 < 1e-12:
                return np.full_like(grad, 1e30)
            inv_r2 = 1.0 / r2
            inv_r6 = inv_r2 ** 3
            # dE/dr = 4*(-12*r^-13 + 6*r^-7) = 24*r^-7 - 48*r^-13
            # dE/dpos_i = (dE/dr)*(r_vec/r) where r_vec = pos_i - pos_j
            # = 24*r^-7 * d/r - 48*r^-13 * d/r
            # = (24*inv_r6 - 48*inv_r6*inv_r2) * d
            coeff = 24.0 * inv_r6 * (1.0 - 2.0 * inv_r2)
            g = coeff * d
            grad[i] += g
            grad[j] -= g
    return grad


def _enforce_min_separation(pos, min_dist=0.5):
    """Push atoms apart if any pair is closer than min_dist."""
    pos = pos.copy()
    n = len(pos)
    for _ in range(100):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = pos[i] - pos[j]
                r = np.linalg.norm(d)
                if r < min_dist and r > 1e-12:
                    # push apart along d
                    push = (min_dist - r) / r * 0.5
                    pos[i] += push * d
                    pos[j] -= push * d
                    moved = True
                elif r <= 1e-12:
                    # random small displacement
                    delta = np.random.randn(3) * 0.1
                    pos[i] += delta
                    pos[j] -= delta
                    moved = True
        if not moved:
            break
    return pos


def _relax(pos):
    """L-BFGS local relaxation with analytic gradient."""
    res = minimize(
        _lj_energy,
        pos,
        jac=_lj_grad,
        method='L-BFGS-B',
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50},
    )
    return res.x


def _basin_hop(pos, n_iters=200, T=0.1, max_T=0.5, min_T=0.01):
    """Basin-hopping with analytic gradients."""
    rng = np.random.RandomState(42)
    best_pos = pos.copy()
    best_e = _lj_energy(best_pos)

    for it in range(n_iters):
        # Perturb
        T = max(min_T, max_T * np.exp(-it / (n_iters / 3.0)))
        perturbation = rng.randn(*pos.shape) * T
        new_pos = _enforce_min_separation(pos + perturbation)

        # Relax
        new_pos = _relax(new_pos)
        new_pos = _enforce_min_separation(new_pos)
        new_e = _lj_energy(new_pos)

        # Metropolis accept
        if new_e < best_e or np.log(rng.random()) < (best_e - new_e):
            pos = new_pos
            if new_e < best_e:
                best_pos = new_pos.copy()
                best_e = new_e

    return best_pos, best_e


def _fcc_truncated_octahedron():
    """
    Generate FCC lattice points and take the 38 closest to origin.
    FCC lattice: points at (i,j,k) where i+j+k is even, scaled by a/2.
    The nearest-neighbor distance in FCC is a/sqrt(2).
    For LJ minimum at r=2^(1/6), we want nearest neighbors at that distance.
    So a/sqrt(2) = 2^(1/6) => a = sqrt(2) * 2^(1/6) = 2^(2/3).
    """
    a = 2.0 ** (2.0 / 3.0)  # FCC lattice constant for LJ
    # Generate FCC points in a box
    side = 4  # enough to get 38+ points
    pts = []
    for i in range(-side, side + 1):
        for j in range(-side, side + 1):
            for k in range(-side, side + 1):
                # FCC: include if i+j+k is even (this gives the FCC sublattice of simple cubic)
                # Actually FCC = simple cubic with basis, or equivalently:
                # FCC points are at (i,j,k) in units of a/2 where i,j,k all same parity
                # Let's use the standard: FCC = {(i,j,k) : i+j+k even} in units of a/2
                if (i + j + k) % 2 == 0:
                    pts.append((i * a / 2.0, j * a / 2.0, k * a / 2.0))

    pts = np.array(pts)
    # Center of mass
    pts -= pts.mean(axis=0)
    # Sort by distance from origin, take 38
    dists = np.linalg.norm(pts, axis=1)
    idx = np.argsort(dists)[:N]
    return pts[idx]


def _icosahedral_seed():
    """Icosahedral arrangement (a common local minimum trap)."""
    # 1 center + 12 icosahedron vertices + 24 edge midpoints (or similar)
    # Icosahedron vertices (radius 1):
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1],
    ])
    ico /= np.linalg.norm(ico[0])  # normalize to unit sphere
    # Scale so nearest neighbor ~ 2^(1/6)
    # In icosahedron, nearest neighbor distance = 2*sin(pi/5) ≈ 1.1756
    scale = 2.0 ** (1.0 / 6.0) / (2.0 * np.sin(np.pi / 5.0))
    ico *= scale

    # 1 center + 12 icosahedron + 24 more (face centers or edge midpoints)
    center = np.array([[0.0, 0.0, 0.0]])

    # Edge midpoints of icosahedron (20 faces -> 30 edges, but let's use face centers)
    # Actually for 38 atoms: 1 + 12 + 24 = 37, need one more.
    # Let's use: 1 center + 12 ico + 24 edge midpoints = 37, add 1 more
    # Or: 1 + 12 + 12 (inner) + 13? Let's just use 1+12+24+1
    # Edge midpoints:
    edges = [(0,4),(0,5),(0,8),(0,9),(1,4),(1,5),(1,12),(1,13),
             (2,6),(2,7),(2,8),(2,9),(3,6),(3,7),(3,12),(3,13),
             (4,7),(4,8),(4,10),(4,11),(5,6),(5,9),(5,10),(5,11),
             (6,11),(6,12),(7,10),(7,13),(8,10),(8,11),(9,10),(9,11),
             (12,10),(12,11),(13,10),(13,11)]
    # That's 30 edges, we need 24. Let's take first 24.
    edge_mids = []
    for i, j in edges[:24]:
        edge_mids.append((ico[i] + ico[j]) / 2.0)
    edge_mids = np.array(edge_mids)

    # Scale edge midpoints to be at proper distance
    # They should be at distance ~ 2^(1/6) from their neighboring ico vertices
    # Current distance from center: |edge_mid| = |ico|*cos(pi/5) ≈ scale * 0.809
    # We want them at a radius such that they're ~2^(1/6) from ico vertices
    # Let's just place them and relax
    extra = np.array([[0.0, 0.0, 2.0 ** (1.0/6.0) * 1.5]])  # 38th atom

    pos = np.vstack([center, ico, edge_mids, extra])
    return pos[:N]


def _random_seed():
    """Random compact seed."""
    rng = np.random.RandomState(7)
    pos = rng.randn(N, 3) * 1.0
    pos -= pos.mean(axis=0)
    return _enforce_min_separation(pos)


def construct():
    start_time = time.time()

    # Generate candidate seeds
    seeds = []

    # 1. FCC truncated octahedron (best guess for global min)
    try:
        fcc = _fcc_truncated_octahedron()
        fcc = _enforce_min_separation(fcc)
        seeds.append(fcc)
    except Exception:
        pass

    # 2. Icosahedral seed
    try:
        ico = _icosahedral_seed()
        ico = _enforce_min_separation(ico)
        seeds.append(ico)
    except Exception:
        pass

    # 3. Random seeds
    for seed_id in range(5):
        rng = np.random.RandomState(100 + seed_id)
        pos = rng.randn(N, 3) * 1.2
        pos -= pos.mean(axis=0)
        pos = _enforce_min_separation(pos)
        seeds.append(pos)

    # 4. Cubic lattice fragment (original seed)
    side = int(np.ceil(N ** (1 / 3)))
    points = [(i, j, k) for i in range(side) for j in range(side) for k in range(side)]
    points.sort(key=lambda p: (p[0] - (side - 1) / 2) ** 2 + (p[1] - (side - 1) / 2) ** 2 + (p[2] - (side - 1) / 2) ** 2)
    cubic = np.array(points[:N], dtype=float) * 1.12
    cubic = _enforce_min_separation(cubic)
    seeds.append(cubic)

    best_pos = None
    best_energy = np.inf

    for si, seed in enumerate(seeds):
        if time.time() - start_time > DEADLINE_SECONDS - 10:
            break

        # Relax the seed first
        try:
            relaxed = _relax(seed)
            relaxed = _enforce_min_separation(relaxed)
            e = _lj_energy(relaxed)
            if e < best_energy:
                best_energy = e
                best_pos = relaxed.copy()
        except Exception:
            continue

        # Basin hopping from this relaxed seed
        # Adjust number of iterations based on time
        elapsed = time.time() - start_time
        remaining = DEADLINE_SECONDS - 10 - elapsed
        if remaining < 5:
            break

        # Estimate time per basin-hop iteration (~0.05-0.2 s)
        # Use a conservative estimate
        max_iters = min(100, int(remaining / 0.3))
        if max_iters < 10:
            max_iters = 10

        try:
            bh_pos, bh_e = _basin_hop(relaxed, n_iters=max_iters)
            if bh_e < best_energy:
                best_energy = bh_e
                best_pos = bh_pos.copy()
        except Exception:
            continue

    # Final high-precision polish
    if best_pos is not None and time.time() - start_time < DEADLINE_SECONDS - 3:
        try:
            polished = minimize(
                _lj_energy,
                best_pos,
                jac=_lj_grad,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100},
            ).x
            polished = _enforce_min_separation(polished)
            pe = _lj_energy(polished)
            if pe < best_energy:
                best_energy = pe
                best_pos = polished.copy()
        except Exception:
            pass

    # Final check: ensure minimum separation
    best_pos = _enforce_min_separation(best_pos)

    return best_pos
