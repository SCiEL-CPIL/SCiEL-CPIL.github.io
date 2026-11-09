"""Morse cluster global minima: structured init + L-BFGS-B + basin hopping."""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

_START = time.time()
_DEADLINE = _START + 285.0  # 15s safety margin under 300s


def _remaining():
    return _DEADLINE - time.time()


def _morse_energy(rho, coords):
    n = len(coords)
    if n < 2:
        return 0.0
    d = cdist(coords, coords)
    np.fill_diagonal(d, np.inf)
    # collision penalty
    min_d = d.min()
    if min_d < 0.5:
        # heavy penalty to push apart
        return 1e6 * (0.5 - min_d) ** 2 + _raw_morse(rho, coords)
    return _raw_morse(rho, coords)


def _raw_morse(rho, coords):
    d = cdist(coords, coords)
    np.fill_diagonal(d, np.inf)
    x = rho * (1.0 - d)
    e = np.exp(x)
    return np.sum(e * (e - 2.0)) / 2.0  # sum over i<j, each pair counted twice in full matrix


def _morse_energy_grad(rho, coords):
    n = len(coords)
    d = cdist(coords, coords)
    np.fill_diagonal(d, np.inf)
    x = rho * (1.0 - d)
    e = np.exp(x)
    # dE/dr_ij = rho * e * (2e - 2)
    # grad_i = sum_{j != i} rho * e_ij * (2 e_ij - 2) * (coords_i - coords_j) / r_ij
    grad = np.zeros_like(coords)
    # vectorized
    # f_ij = rho * e_ij * (2 e_ij - 2) / r_ij  (scalar per pair)
    f = rho * e * (2.0 * e - 2.0) / d
    np.fill_diagonal(f, 0.0)
    # grad_i = sum_j f_ij * (coords_i - coords_j)
    # = (sum_j f_ij) * coords_i - sum_j f_ij * coords_j
    row_sum = f.sum(axis=1, keepdims=True)
    grad = row_sum * coords - f @ coords
    return grad


def _local_minimize(rho, coords):
    """L-BFGS-B with collision penalty. Returns (energy, coords)."""
    n = len(coords)
    if n < 2:
        return 0.0, coords

    def obj_and_grad(x):
        c = x.reshape(n, 3)
        # compute energy and gradient with penalty
        d = cdist(c, c)
        np.fill_diagonal(d, np.inf)
        min_d = d.min()
        x_m = rho * (1.0 - d)
        e = np.exp(x_m)
        energy = np.sum(e * (e - 2.0)) / 2.0
        f = rho * e * (2.0 * e - 2.0) / d
        np.fill_diagonal(f, 0.0)
        row_sum = f.sum(axis=1, keepdims=True)
        grad = row_sum * c - f @ c

        if min_d < 0.5:
            # penalty
            penalty = 1e4 * (0.5 - min_d) ** 2
            # gradient of penalty: for the closest pair
            idx = np.unravel_index(np.argmin(d), d.shape)
            i, j = idx
            diff = c[i] - c[j]
            norm = np.linalg.norm(diff)
            if norm > 1e-12:
                gpen = -2e4 * (0.5 - min_d) * diff / norm
                grad[i] += gpen
                grad[j] -= gpen
            energy += penalty

        return energy, grad.flatten()

    x0 = coords.flatten()
    try:
        res = minimize(
            obj_and_grad, x0, jac=True,
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 500}
        )
        coords_opt = res.x.reshape(n, 3)
        energy_opt = res.fun
        return energy_opt, coords_opt
    except Exception:
        return _morse_energy(rho, coords), coords


def _gen_icosahedral(n, rng):
    """Icosahedral shells: 1 center + 12 + 30 + 60 + ..."""
    # Icosahedral shell sizes: 1, 12, 30, 60, 120, 180, ...
    # Generate icosahedral vertices
    def ico_shell(radius):
        phi = (1 + np.sqrt(5)) / 2
        pts = []
        # 12 vertices of icosahedron
        base = [
            [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
            [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
            [phi, 0, 1], [-phi, 0, 1], [phi, 0, -1], [-phi, 0, -1],
        ]
        for p in base:
            v = np.array(p, dtype=float)
            v = v / np.linalg.norm(v) * radius
            pts.append(v)
        return np.array(pts)

    # Build shells
    shell_sizes = [1, 12, 30, 60, 120, 180, 300]
    radii = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    points = []
    for size, rad in zip(shell_sizes, radii):
        if len(points) >= n:
            break
        if size == 1:
            points.append(np.zeros(3))
        else:
            shell = ico_shell(rad)
            # For shells larger than 12, subdivide: use icosahedron vertices scaled + jitter
            if size > 12:
                # approximate: place points on shell with some structure
                for k in range(size):
                    # use icosahedral vertices with small random perturbation
                    base_idx = k % 12
                    p = shell[base_idx].copy()
                    # add small random offset
                    p += rng.normal(0, 0.15, 3)
                    p = p / np.linalg.norm(p) * rad
                    points.append(p)
            else:
                for p in shell:
                    if len(points) < n:
                        points.append(p)
    # If still short, add random points on a sphere
    while len(points) < n:
        p = rng.normal(0, 1, 3)
        p = p / np.linalg.norm(p) * (2.0 + rng.uniform(0, 1.5))
        points.append(p)
    return np.array(points[:n])


def _gen_fcc(n, rng):
    """FCC lattice chunk."""
    a = 1.0  # lattice constant ~ equilibrium distance
    pts = []
    # Generate FCC points in a cube
    extent = int(np.ceil(n ** (1/3))) + 2
    for i in range(-extent, extent + 1):
        for j in range(-extent, extent + 1):
            for k in range(-extent, extent + 1):
                # FCC basis
                for dx, dy, dz in [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    if np.linalg.norm(p) < 6.0:
                        pts.append(p)
    pts = np.array(pts)
    # Center
    pts -= pts.mean(axis=0)
    # Shuffle and take n
    idx = rng.permutation(len(pts))[:n]
    return pts[idx]


def _gen_bcc(n, rng):
    """BCC lattice chunk."""
    a = 1.0
    pts = []
    extent = int(np.ceil(n ** (1/3))) + 2
    for i in range(-extent, extent + 1):
        for j in range(-extent, extent + 1):
            for k in range(-extent, extent + 1):
                for dx, dy, dz in [(0, 0, 0), (0.5, 0.5, 0.5)]:
                    p = np.array([i + dx, j + dy, k + dz]) * a
                    if np.linalg.norm(p) < 6.0:
                        pts.append(p)
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    idx = rng.permutation(len(pts))[:n]
    return pts[idx]


def _gen_random_shell(n, rng):
    """Random points in a sphere."""
    radius = 1.0 * n ** (1/3) * 0.8
    pts = []
    while len(pts) < n:
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) <= radius:
            pts.append(p)
    return np.array(pts[:n])


def _generate_structured(n, rng, seed_idx):
    """Generate a structured initial configuration based on seed_idx."""
    if seed_idx == 0:
        return _gen_icosahedral(n, rng)
    elif seed_idx == 1:
        return _gen_fcc(n, rng)
    elif seed_idx == 2:
        return _gen_bcc(n, rng)
    elif seed_idx == 3:
        return _gen_random_shell(n, rng)
    else:
        # perturbed icosahedral
        pts = _gen_icosahedral(n, rng)
        pts += rng.normal(0, 0.3, pts.shape)
        return pts


def construct(instance):
    n = instance["n"]
    rho = instance.get("rho", instance.get("kind", 6))
    potential = instance.get("potential", "morse")

    rng = np.random.default_rng(42)

    best_energy = np.inf
    best_coords = None

    # Generate multiple structured starting points
    n_struct = 6  # number of structured inits
    for s in range(n_struct):
        if _remaining() < 15.0:
            break
        coords = _generate_structured(n, rng, s)
        # Ensure minimum distance constraint by small perturbation if needed
        # Local minimize
        energy, coords_opt = _local_minimize(rho, coords)
        if energy < best_energy:
            best_energy = energy
            best_coords = coords_opt

    if best_coords is None:
        # Fallback
        coords = _gen_random_shell(n, rng)
        best_energy, best_coords = _local_minimize(rho, coords)

    # Basin hopping
    kick_scale = 0.5
    best_bh = best_energy
    best_bh_coords = best_coords.copy()
    stagnation = 0

    while _remaining() > 20.0:
        # Perturb
        perturbed = best_bh_coords + rng.normal(0, kick_scale, best_bh_coords.shape)
        energy, coords_opt = _local_minimize(rho, perturbed)

        if energy < best_bh:
            best_bh = energy
            best_bh_coords = coords_opt.copy()
            stagnation = 0
            # Slightly increase kick on success (explore more)
            kick_scale = min(kick_scale * 1.1, 1.5)
        else:
            stagnation += 1
            if stagnation > 5:
                # Anneal: reduce kick
                kick_scale *= 0.85
                stagnation = 0
                if kick_scale < 0.1:
                    # Try a fresh structured init
                    s = rng.integers(0, 6)
                    coords = _generate_structured(n, rng, s)
                    energy, coords_opt = _local_minimize(rho, coords)
                    if energy < best_bh:
                        best_bh = energy
                        best_bh_coords = coords_opt.copy()
                    kick_scale = 0.5

    return best_bh_coords
