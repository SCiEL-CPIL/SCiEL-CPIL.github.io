"""Circle packing via repulsive-force MD + Nelder-Mead polish."""
import numpy as np
from scipy.optimize import minimize
import time

N = 50
DEADLINE = 140.0  # safety margin before 150s kill


def _compute_score(pts):
    """Compute the true packing radius (nonsmooth)."""
    n = len(pts)
    diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    radii = np.sqrt(np.sum(pts ** 2, axis=-1))
    min_margin = np.min(1.0 - radii)
    return min(min_pair, min_margin)


def _force_relax(pts, time_limit):
    """
    Repulsive-force MD: particles repel each other and are pushed inward by container.
    Uses velocity Verlet with adaptive repulsion strength.
    """
    pts = pts.copy().astype(np.float64)
    v = np.zeros_like(pts)
    n = len(pts)
    t0 = time.time()

    # Adaptive parameters
    repulse_strength = 10.0
    container_strength = 50.0
    damping = 0.98
    dt = 0.005

    # Initial overlap check to set dt
    def _forces(pos, rep_str, cont_str):
        """Compute forces on all particles (vectorized)."""
        # Pairwise repulsion
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dist, 1.0)  # avoid div by zero
        # Force magnitude: repulsive, stronger when closer
        # F = rep_str * diff / dist^3 (inverse square-ish)
        inv_dist3 = np.zeros_like(dist)
        mask = dist < 1.0  # only repel when close (within 1.0)
        # Use a smooth repulsion: F = rep_str * (1/dist - 1) * diff/dist for dist < 1
        # Actually let's use: F = rep_str * diff / dist^2 for all pairs (Coulomb-like)
        # But that's too long-range. Let's use: F = rep_str * (diff / dist) * max(0, 1/dist - 0.5)
        # Simpler: F = rep_str * diff / dist^3 for dist < threshold
        # Let's use a clean repulsion: F = rep_str * diff / (dist^2 + eps)
        eps = 1e-6
        inv_d2 = np.zeros_like(dist)
        inv_d2[mask] = 1.0 / (dist[mask] ** 2 + eps)
        # Force on i from j: rep_str * (pos_i - pos_j) / dist^2
        force = np.sum(rep_str * diff * inv_d2[:, :, np.newaxis], axis=1)

        # Container force: push inward if outside radius 0.99
        radii = np.sqrt(np.sum(pos ** 2, axis=1))
        container_force = np.zeros_like(pos)
        outside = radii > 0.99
        if np.any(outside):
            # Push toward center proportional to how far outside
            container_force[outside] = -cont_str * pos[outside] / radii[outside, np.newaxis]

        return force + container_force

    n_iters = 0
    max_iters = 20000
    best_score = _compute_score(pts)
    best_pts = pts.copy()
    patience = 0
    max_patience = 200

    for it in range(max_iters):
        if time.time() - t0 > time_limit:
            break

        # Compute forces
        f = _forces(pts, repulse_strength, container_strength)

        # Velocity Verlet: v += f*dt, x += v*dt
        v = v * damping + f * dt
        pts = pts + v * dt

        # Project into unit disk (hard constraint)
        radii = np.linalg.norm(pts, axis=1)
        mask = radii > 1.0
        if np.any(mask):
            pts[mask] = pts[mask] / radii[mask, np.newaxis] * 0.999

        # Check score periodically
        if it % 50 == 0:
            score = _compute_score(pts)
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
                patience = 0
            else:
                patience += 1
                # Adaptive: reduce repulsion if stuck
                if patience > max_patience:
                    repulse_strength *= 0.9
                    dt *= 0.9
                    patience = 0

        n_iters += 1

    return best_pts


def _polish(pts, time_limit):
    """Nelder-Mead polish on the true nonsmooth objective."""
    t0 = time.time()
    current_score = _compute_score(pts)

    def neg_score(x):
        pts_local = x.reshape(-1, 2)
        s = _compute_score(pts_local)
        return -s

    x0 = pts.flatten()
    # Use smaller initial simplex for better convergence
    res = minimize(neg_score, x0, method='Nelder-Mead',
                   options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-15,
                            'adaptive': True, 'initial_simplex': None})
    pts_opt = res.x.reshape(-1, 2)
    score_opt = _compute_score(pts_opt)
    if score_opt > current_score:
        return pts_opt
    return pts


def _hex_init(spacing):
    """Hexagonal lattice initialization."""
    k = int(1.0 / spacing) + 2
    rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
            for i in range(-k, k + 1) for j in range(-k, k + 1)]
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    points = points[radius <= 1.0 - spacing / 2.0]
    if len(points) >= N:
        radius = np.hypot(points[:, 0], points[:, 1])
        points = points[np.argsort(radius, kind='stable')[:N]]
    return points


def _ring_init(n_rings=8):
    """Concentric rings initialization."""
    pts = [np.array([0.0, 0.0])]
    for ring in range(1, n_rings):
        r = ring * 0.12
        n_in_ring = min(6 * ring, N - len(pts))
        if n_in_ring <= 0:
            break
        angles = np.linspace(0, 2 * np.pi, n_in_ring, endpoint=False)
        for a in angles:
            pts.append(np.array([r * np.cos(a), r * np.sin(a)]))
    pts = np.array(pts[:N])
    return pts


def _random_init(seed):
    """Random initialization within unit disk."""
    rng = np.random.RandomState(seed)
    # Use rejection sampling for uniform in disk
    pts = []
    while len(pts) < N:
        x = rng.uniform(-1, 1, size=(N * 2, 2))
        r = np.hypot(x[:, 0], x[:, 1])
        valid = r <= 0.95
        pts.extend(x[valid].tolist())
    pts = np.array(pts[:N])
    return pts


def construct():
    t_start = time.time()

    best_pts = None
    best_score = -1.0

    # Multiple initializations
    inits = [
        _hex_init(0.25),
        _hex_init(0.22),
        _hex_init(0.28),
        _ring_init(8),
        _ring_init(7),
        _random_init(42),
        _random_init(123),
        _random_init(7),
    ]

    for init_idx, init_pts in enumerate(inits):
        if time.time() - t_start > DEADLINE - 10:
            break
        if len(init_pts) < N:
            continue

        # Force relaxation
        time_remaining = DEADLINE - (time.time() - t_start)
        time_for_relax = min(time_remaining * 0.55, 15.0)
        if time_for_relax < 2.0:
            break

        pts_relaxed = _force_relax(init_pts, time_for_relax)

        # Polish
        time_remaining = DEADLINE - (time.time() - t_start)
        time_for_polish = min(time_remaining * 0.35, 12.0)
        if time_for_polish > 1.0:
            pts_polished = _polish(pts_relaxed, time_for_polish)
        else:
            pts_polished = pts_relaxed

        score = _compute_score(pts_polished)
        if score > best_score:
            best_score = score
            best_pts = pts_polished.copy()

    if best_pts is None:
        best_pts = _hex_init(0.25)

    return best_pts
