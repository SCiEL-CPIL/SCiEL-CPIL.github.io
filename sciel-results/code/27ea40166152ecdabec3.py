"""Morse cluster global minima search with multi-start optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, Bounds
from autoresearch_suite.chem_clusters import relax, morse_energy_gradient

_START_TIME = time.time()
_TIME_LIMIT = 290.0  # safety margin before 330s kill


def _remaining_time():
    return _TIME_LIMIT - (time.time() - _START_TIME)


def _morse_energy(pos, rho):
    """Compute Morse potential energy."""
    n = len(pos)
    d = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    r = np.sqrt(np.sum(d ** 2, axis=2))
    np.fill_diagonal(r, 1e10)
    u = rho * (1.0 - r)
    e = np.exp(u)
    energy = np.sum(e * (e - 2.0))
    return 0.5 * energy


def _fix_min_dist(pos, min_d=0.501):
    """Separate any atoms closer than min_d."""
    n = len(pos)
    for _ in range(200):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                d = pos[i] - pos[j]
                dist = np.linalg.norm(d)
                if dist < min_d:
                    if dist < 1e-8:
                        d = np.array([1.0, 0.0, 0.0])
                        dist = 1.0
                    push = (min_d - dist) / dist
                    pos[i] += 0.5 * push * d
                    pos[j] -= 0.5 * push * d
                    moved = True
        if not moved:
            break
    return pos


def _optimize(pos, potential, rho):
    """Run multi-method local optimization."""
    n = len(pos)
    pos = _fix_min_dist(pos)
    best_energy = _morse_energy(pos, rho)
    best_pos = pos.copy()
    
    # Try scipy L-BFGS-B
    if _remaining_time() > 2.0:
        try:
            res = minimize(
                lambda x: _morse_energy(x.reshape(n, 3), rho),
                pos.ravel(),
                jac=lambda x: morse_energy_gradient(x.reshape(n, 3), rho)[1].ravel(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.fun < best_energy:
                best_energy = res.fun
                best_pos = res.x.reshape(n, 3)
        except Exception:
            pass
    
    # Try chem_clusters.relax
    if _remaining_time() > 1.0:
        try:
            energy, positions = relax(best_pos.copy(), potential, rho)
            if energy < best_energy:
                best_energy = energy
                best_pos = positions
        except Exception:
            pass
    
    return best_energy, _fix_min_dist(best_pos)


def _make_icosahedron(n, rho):
    """Create icosahedral shell configuration."""
    # Icosahedron vertices
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [0, 1, phi], [0, -1, phi], [0, 1, -phi], [0, -1, -phi],
        [1, phi, 0], [-1, phi, 0], [1, -phi, 0], [-1, -phi, 0],
        [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    
    # Fibonacci sphere for additional shells
    def fib_sphere(m, radius):
        if m == 0:
            return np.empty((0, 3))
        if m == 1:
            return np.array([[0.0, 0.0, radius]])
        points = []
        golden = (1 + np.sqrt(5)) / 2
        for i in range(m):
            z = 1 - 2 * i / (m - 1) if m > 1 else 0
            r = np.sqrt(max(0, 1 - z * z))
            theta = 2 * np.pi * i / golden
            points.append([r * np.cos(theta) * radius, r * np.sin(theta) * radius, z * radius])
        return np.array(points)
    
    points = []
    # Core
    points.append(np.zeros(3))
    # First shell: 12 icosahedral vertices
    r1 = 1.0
    for v in verts:
        points.append(v * r1)
    count = 1 + 12
    
    # Additional shells
    shell_count = 0
    while count < n and shell_count < 10:
        shell_count += 1
        r = 1.0 + 0.9 * shell_count
        shell_size = min(n - count, 20 + 10 * shell_count)
        pts = fib_sphere(shell_size, r)
        points.extend(pts)
        count += shell_size
    
    pos = np.array(points[:n])
    pos = _fix_min_dist(pos)
    return pos


def _make_fcc(n, rho):
    """Create FCC-like cluster."""
    a = 1.0  # lattice parameter (equilibrium distance)
    points = []
    # Generate FCC lattice points in a sphere
    for ix in range(-6, 7):
        for iy in range(-6, 7):
            for iz in range(-6, 7):
                # FCC: corners + face centers
                for fx, fy, fz in [(0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5)]:
                    p = np.array([ix + fx, iy + fy, iz + fz]) * a
                    if np.linalg.norm(p) < 4.0:
                        points.append(p)
    
    points = np.array(points)
    # Sort by distance from center and take n closest
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    pos = points[idx]
    pos = _fix_min_dist(pos)
    return pos


def _make_bcc(n, rho):
    """Create BCC-like cluster."""
    a = 1.0
    points = []
    for ix in range(-6, 7):
        for iy in range(-6, 7):
            for iz in range(-6, 7):
                for fx, fy, fz in [(0,0,0), (0.5,0.5,0.5)]:
                    p = np.array([ix + fx, iy + fy, iz + fz]) * a
                    if np.linalg.norm(p) < 4.0:
                        points.append(p)
    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    pos = points[idx]
    pos = _fix_min_dist(pos)
    return pos


def _make_hcp(n, rho):
    """Create HCP-like cluster."""
    a = 1.0
    c = a * np.sqrt(8.0 / 3.0)
    points = []
    for layer in range(-4, 5):
        z = layer * c / 2
        for ix in range(-6, 7):
            for iy in range(-6, 7):
                # AB stacking
                if layer % 2 == 0:
                    p = np.array([ix * a, iy * a * np.sqrt(3) / 2, z])
                else:
                    p = np.array([(ix + 0.5) * a, (iy * np.sqrt(3) / 2 + a * np.sqrt(3) / 6), z])
                if np.linalg.norm(p) < 4.0:
                    points.append(p)
    
    points = np.array(points)
    dists = np.linalg.norm(points, axis=1)
    idx = np.argsort(dists)[:n]
    pos = points[idx]
    pos = _fix_min_dist(pos)
    return pos


def _make_random_packed(n, seed=42):
    """Random packed configuration."""
    rng = np.random.default_rng(seed)
    points = []
    radius = n ** (1.0 / 3.0) * 0.8
    max_attempts = n * 200
    attempts = 0
    while len(points) < n and attempts < max_attempts:
        attempts += 1
        p = rng.uniform(-radius, radius, 3)
        if np.linalg.norm(p) > radius:
            continue
        if all(np.linalg.norm(p - q) >= 0.52 for q in points):
            points.append(p)
    
    # Fill remaining with random points
    while len(points) < n:
        p = rng.uniform(-radius, radius, 3)
        points.append(p)
    
    pos = np.array(points)
    pos = _fix_min_dist(pos)
    return pos


def _make_perturbed(base, n, seed=0):
    """Perturb a base configuration."""
    rng = np.random.default_rng(seed)
    pos = base[:n].copy()
    pos += rng.normal(0, 0.1, pos.shape)
    pos = _fix_min_dist(pos)
    return pos


def construct(instance):
    """Find global minimum of Morse potential cluster."""
    n = instance["n"]
    potential = instance["potential"]
    rho = instance.get("rho", instance.get("kind", 3))
    
    t_start = time.time()
    
    # Generate diverse starting configurations
    generators = [
        lambda: _make_icosahedron(n, rho),
        lambda: _make_fcc(n, rho),
        lambda: _make_bcc(n, rho),
        lambda: _make_hcp(n, rho),
        lambda: _make_random_packed(n, seed=n),
        lambda: _make_random_packed(n, seed=n + 100),
        lambda: _make_perturbed(_make_icosahedron(n, rho), n, seed=1),
        lambda: _make_perturbed(_make_fcc(n, rho), n, seed=2),
    ]
    
    best_energy = np.inf
    best_pos = None
    
    # Try each generator
    for i, gen in enumerate(generators):
        if _remaining_time() < 5.0:
            break
        try:
            pos = gen()
            if pos is None or len(pos) != n:
                continue
            energy, pos = _optimize(pos, potential, rho)
            if energy < best_energy:
                best_energy = energy
                best_pos = pos.copy()
        except Exception:
            continue
    
    # If time remains, do additional random restarts
    rng = np.random.default_rng(n + 7)
    while _remaining_time() > 10.0:
        try:
            # Random start in a sphere
            radius = n ** (1.0 / 3.0) * 0.8
            pos = rng.uniform(-radius, radius, (n, 3))
            # Project to sphere
            norms = np.linalg.norm(pos, axis=1, keepdims=True)
            pos = pos / norms * np.minimum(norms, radius)
            pos = _fix_min_dist(pos)
            energy, pos = _optimize(pos, potential, rho)
            if energy < best_energy:
                best_energy = energy
                best_pos = pos.copy()
        except Exception:
            break
    
    if best_pos is None:
        # Fallback
        best_pos = _make_icosahedron(n, rho)
        try:
            _, best_pos = _optimize(best_pos, potential, rho)
        except Exception:
            pass
    
    return best_pos
# EVOLVE-BLOCK-END
