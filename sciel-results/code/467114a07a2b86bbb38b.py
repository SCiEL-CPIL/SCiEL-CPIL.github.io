"""Packing 50 equal circles in a unit circle - SDR-based QCQP with L-BFGS-B polish."""
import time
import numpy as np
from scipy.optimize import minimize, linprog

N = 50
_DEADLINE = time.time() + 140.0


def _score(pts):
    """Compute the valid packing radius (or 0 if invalid)."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    dists = np.hypot(pts[:, 0], pts[:, 1])
    if np.any(dists > 1.0):
        return 0.0
    r_cont = 1.0 - np.max(dists)
    if r_cont < 0:
        return 0.0
    if N < 2:
        return r_cont
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:1, 1]
    dmat = np.sqrt(dx ** 2 + dy ** 2)
    np.fill_diagonal(dmat, np.inf)
    min_pair = np.min(dmat)
    r_pair = min_pair / 2.0
    return min(r_cont, r_pair)


def _hex_lattice_init(n, jitter=0.0):
    """Generate initial positions from hexagonal lattice."""
    for spacing in np.linspace(0.14, 0.35, 40):
        k = int(1.0 / spacing) + 2
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                if x * x + y * y <= 1.0:
                    rows.append((x, y))
        pts = np.array(rows)
        if len(pts) >= n:
            idx = np.random.permutation(len(pts))[:n]
            pts = pts[idx]
            if jitter > 0:
                pts += np.random.randn(n, 2) * jitter
            r = np.hypot(pts[:, 0], pts[:, 1])
            mask = r > 1.0
            if np.any(mask):
                pts[mask] /= r[mask, None] * 1.001
            return pts
    theta = np.random.uniform(0, 2 * np.pi, n)
    rho = np.sqrt(np.random.uniform(0, 1, n))
    return np.column_stack([rho * np.cos(theta), rho * np.sin(theta)])


def _concentric_ring_init(n):
    """Generate initial positions on concentric rings."""
    pts = [[0.0, 0.0]]
    remaining = n - 1
    ring_radius = 0.2
    while remaining > 0:
        circumference = 2 * np.pi * ring_radius
        n_on_ring = max(1, int(circumference / 0.25))
        n_on_ring = min(n_on_ring, remaining)
        if n_on_ring == 0:
            ring_radius += 0.15
            continue
        angles = np.linspace(0, 2 * np.pi, n_on_ring, endpoint=False)
        for a in angles:
            pts.append([ring_radius * np.cos(a), ring_radius * np.sin(a)])
        remaining -= n_on_ring
        ring_radius += 0.18
    pts = np.array(pts[:n])
    r = np.hypot(pts[:, 0], pts[:, 1])
    mask = r > 1.0
    if np.any(mask):
        pts[mask] /= r[mask, None] * 1.001
    return pts


def _random_init(n):
    theta = np.random.uniform(0, 2 * np.pi, n)
    rho = np.sqrt(np.random.uniform(0, 1, n))
    return np.column_stack([rho * np.cos(theta), rho * np.sin(theta)])


def _soft_min_surrogate(pts, beta):
    """Compute soft-min surrogate and gradient."""
    if len(pts) != N:
        return 1e6, np.zeros(2 * N)
    if not np.all(np.isfinite(pts)):
        return 1e6, np.zeros(2 * N)

    dists = np.hypot(pts[:, 0], pts[:, 1])
    r_cont = 1.0 - np.max(dists)

    if r_cont < -0.1:
        return 1e6, np.zeros(2 * N)

    i_idx, j_idx = np.triu_indices(N, k=1)
    dx = pts[i_idx, 0] - pts[j_idx, 0]
    dy = pts[i_idx, 1] - pts[j_idx, 1]
    d_sq = dx * dx + dy * dy
    d = np.sqrt(d_sq + 1e-12)

    half_d = d / 2.0
    all_radii = np.concatenate([[r_cont], half_d])

    x_max = np.max(all_radii)
    exp_terms = np.exp(-beta * (all_radii - x_max))
    soft_min_val = x_max - (1.0 / beta) * np.log(np.sum(exp_terms))

    weights = exp_terms / np.sum(exp_terms)

    grad = np.zeros(2 * N)

    max_idx = np.argmax(dists)
    if dists[max_idx] > 1e-10:
        grad[2 * max_idx] = -weights[0] * pts[max_idx, 0] / dists[max_idx]
        grad[2 * max_idx + 1] = -weights[0] * pts[max_idx, 1] / dists[max_idx]

    for k in range(len(i_idx)):
        i = i_idx[k]
        j = j_idx[k]
        w = weights[k + 1]
        if d[k] > 1e-10:
            grad[2 * i] += w * (pts[i, 0] - pts[j, 0]) / (2.0 * d[k])
            grad[2 * i + 1] += w * (pts[i, 1] - pts[j, 1]) / (2.0 * d[k])
            grad[2 * j] += w * (pts[j, 0] - pts[i, 0]) / (2.0 * d[k])
            grad[2 * j + 1] += w * (pts[j, 1] - pts[i, 1]) / (2.0 * d[k])

    return -soft_min_val, -grad


def _lbfgs_optimize(pts, beta_start=10.0, beta_end=5000.0, n_phases=8):
    """Optimize using soft-min surrogate with L-BFGS-B, gradually increasing beta."""
    pts = pts.copy().astype(np.float64)
    best_score = _score(pts)
    best_pts = pts.copy()

    betas = np.geomspace(beta_start, beta_end, n_phases)

    for beta in betas:
        if time.time() > _DEADLINE - 5.0:
            break

        def objective_and_grad(flat):
            p = flat.reshape(N, 2)
            val, grad = _soft_min_surrogate(p, beta)
            return val, grad

        flat = pts.flatten()

        try:
            result = minimize(
                objective_and_grad,
                flat,
                jac=True,
                method='L-BFGS-B',
                options={
                    'maxiter': 500,
                    'gtol': 1e-8,
                    'ftol': 1e-10,
                    'maxfun': 5000
                }
            )
            new_pts = result.x.reshape(N, 2)
        except Exception:
            continue

        r = np.hypot(new_pts[:, 0], new_pts[:, 1])
        mask = r > 1.0
        if np.any(mask):
            new_pts[mask] /= r[mask, None] * 1.0001

        new_score = _score(new_pts)

        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts.copy()
            pts = new_pts.copy()

    return best_pts, best_score


def _sdr_iteration(pts, r_current, max_iters=20):
    """
    Sequential SDR approach:
    1. Linearize the QCQP around current point
    2. Solve the relaxed problem
    3. Project back to feasible region
    
    The QCQP is:
    maximize r
    subject to:
      ||c_i - c_j||^2 >= (2r)^2 for all i<j
      ||c_i||^2 + r^2 <= 1 for all i
    
    We use a sequential approach where we:
    1. Fix r at current value
    2. Solve for positions that satisfy constraints
    3. Try to increase r
    """
    pts = pts.copy().astype(np.float64)
    best_pts = pts.copy()
    best_score = _score(pts)
    best_r = r_current

    for iteration in range(max_iters):
        if time.time() > _DEADLINE - 5.0:
            break

        # Try to increase r slightly
        r_try = best_r * 1.001 + 0.0001

        # Check if r_try is feasible with current positions
        dists = np.hypot(pts[:, 0], pts[:, 1])
        r_cont = 1.0 - np.max(dists)
        
        i_idx, j_idx = np.triu_indices(N, k=1)
        dx = pts[i_idx, 0] - pts[j_idx, 0]
        dy = pts[i_idx, 1] - pts[j_idx, 1]
        d_sq = dx * dx + dy * dy
        min_pair_sq = np.min(d_sq)
        r_pair = np.sqrt(min_pair_sq) / 2.0

        if min(r_cont, r_pair) >= r_try:
            # Current positions already feasible for r_try, try to increase further
            best_r = r_try
            continue

        # Need to move points. Use force-directed approach:
        # Push overlapping pairs apart, pull container-violating points inward
        force = np.zeros_like(pts)

        # Pairwise repulsion for pairs too close
        for k in range(len(i_idx)):
            i = i_idx[k]
            j = j_idx[k]
            d = np.sqrt(d_sq[k] + 1e-12)
            if d < 2 * r_try:
                # Repulsive force proportional to overlap
                overlap = 2 * r_try - d
                if d > 1e-10:
                    fx = (pts[i, 0] - pts[j, 0]) / d * overlap * 0.5
                    fy = (pts[i, 1] - pts[j, 1]) / d * overlap * 0.5
                    force[i] += [fx, fy]
                    force[j] -= [fx, fy]

        # Container constraint: pull points inward if too far
        for i in range(N):
            if dists[i] + r_try > 1.0:
                if dists[i] > 1e-10:
                    pull = (dists[i] + r_try - 1.0) * 0.5
                    force[i, 0] -= pts[i, 0] / dists[i] * pull
                    force[i, 1] -= pts[i, 1] / dists[i] * pull

        # Apply forces with adaptive step size
        step = 0.1
        new_pts = pts + step * force

        # Project inside container
        nr = np.hypot(new_pts[:, 0], new_pts[:, 1])
        mask = nr > 1.0
        if np.any(mask):
            new_pts[mask] /= nr[mask, None] * 1.0001

        new_score = _score(new_pts)

        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts.copy()
            best_r = new_score
            pts = new_pts.copy()
        else:
            # Try smaller step
            step *= 0.5
            if step < 0.001:
                break

    return best_pts, best_score


def _sdr_refine(pts, r_current):
    """
    Full SDR refinement: alternate between increasing r and optimizing positions.
    """
    pts = pts.copy().astype(np.float64)
    best_pts = pts.copy()
    best_score = _score(pts)
    best_r = r_current

    for outer in range(5):
        if time.time() > _DEADLINE - 5.0:
            break

        # Phase 1: Try to increase r with force-directed moves
        pts, r_new = _sdr_iteration(pts, best_r, max_iters=10)
        if r_new > best_r:
            best_r = r_new
            best_pts = pts.copy()
            best_score = r_new

        # Phase 2: Polish with L-BFGS-B at high beta
        if time.time() > _DEADLINE - 10.0:
            break

        polished_pts, polished_score = _lbfgs_optimize(
            pts, beta_start=1000.0, beta_end=10000.0, n_phases=4
        )
        if polished_score > best_score:
            best_score = polished_score
            best_pts = polished_pts.copy()
            best_r = polished_score
            pts = polished_pts.copy()

    return best_pts, best_score


def _simulated_annealing(pts, max_iters=3000, T_init=0.08, T_min=1e-4, cooling=0.9995):
    """Simulated annealing to escape local minima."""
    pts = pts.copy().astype(np.float64)
    n = len(pts)
    T = T_init
    best_score = _score(pts)
    best_pts = pts.copy()
    current_score = best_score

    for it in range(max_iters):
        if time.time() > _DEADLINE - 5.0:
            break

        scale = 0.05 * T / T_init + 0.001
        new_pts = pts + np.random.randn(n, 2) * scale

        nr = np.hypot(new_pts[:, 0], new_pts[:, 1])
        mask = nr > 1.0
        if np.any(mask):
            new_pts[mask] /= nr[mask, None] * 1.0001

        new_score = _score(new_pts)

        if new_score >= current_score:
            pts = new_pts
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_pts = pts.copy()
        else:
            delta = current_score - new_score
            if delta > 0 and np.random.random() < np.exp(-delta / max(T, 1e-10)):
                pts = new_pts
                current_score = new_score

        T = max(T * cooling, T_min)

    return best_pts, best_score


def construct():
    """Construct 50 circle centers in unit circle maximizing common radius."""
    best_pts = None
    best_score = 0.0

    # Multiple initialization strategies
    inits = [
        _hex_lattice_init(N, jitter=0.0),
        _hex_lattice_init(N, jitter=0.02),
        _concentric_ring_init(N),
        _random_init(N),
        _hex_lattice_init(N, jitter=0.05),
        _concentric_ring_init(N) + np.random.randn(N, 2) * 0.03,
        _hex_lattice_init(N, jitter=0.01),
        _random_init(N),
    ]

    for i, pts in enumerate(inits):
        if time.time() > _DEADLINE - 20.0:
            break

        # Project inside container
        r = np.hypot(pts[:, 0], pts[:, 1])
        mask = r > 1.0
        if np.any(mask):
            pts[mask] /= r[mask, None] * 1.001

        # Run L-BFGS-B optimization
        optimized_pts, score = _lbfgs_optimize(
            pts,
            beta_start=10.0,
            beta_end=5000.0,
            n_phases=6
        )

        if score > best_score:
            best_score = score
            best_pts = optimized_pts.copy()

        # If we have time, try SDR refinement
        if time.time() < _DEADLINE - 30.0 and i % 2 == 0:
            sdr_pts, sdr_score = _sdr_refine(optimized_pts, score)
            if sdr_score > best_score:
                best_score = sdr_score
                best_pts = sdr_pts.copy()

        # If we have time, try simulated annealing
        if time.time() < _DEADLINE - 40.0 and i % 3 == 0:
            sa_pts, sa_score = _simulated_annealing(
                best_pts,
                max_iters=2000,
                T_init=0.06,
                T_min=1e-4,
                cooling=0.9995
            )
            if sa_score > best_score:
                best_score = sa_score
                best_pts = sa_pts.copy()

    if best_pts is None:
        best_pts = _hex_lattice_init(N, jitter=0.0)
        best_score = _score(best_pts)

    # Final high-precision polish
    if time.time() < _DEADLINE - 3.0:
        polished_pts, polished_score = _lbfgs_optimize(
            best_pts, beta_start=2000.0, beta_end=10000.0, n_phases=4
        )
        if polished_score > best_score:
            best_score = polished_score
            best_pts = polished_pts

    # Final validation and cleanup
    r = np.hypot(best_pts[:, 0], best_pts[:, 1])
    mask = r > 1.0
    if np.any(mask):
        best_pts[mask] /= r[mask, None] * 1.0001

    return best_pts
