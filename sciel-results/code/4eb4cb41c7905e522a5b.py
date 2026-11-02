import numpy as np
from scipy.optimize import minimize
import time

N = 300
SEED = 12345
T_START = time.time()
DEADLINE = 145.0  # Safety margin before 150s limit

# Precompute upper triangle indices
_I_IDX, _J_IDX = np.triu_indices(N, k=1)
_N_PAIRS = len(_I_IDX)


def _score(points):
    """Compute the true packing radius."""
    if points.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0

    # Distance to container boundary
    norm = np.linalg.norm(points, axis=1)
    r_bound = 1.0 - np.max(norm)
    if r_bound < 0.0:
        return 0.0

    # Minimum pairwise half-distance (only upper triangle)
    diff = points[_I_IDX] - points[_J_IDX]
    dists = np.sqrt(np.sum(diff * diff, axis=1))
    r_pair = 0.5 * np.min(dists)

    r = min(r_bound, r_pair)
    return float(r) if r > 0.0 else 0.0


def _soft_min_obj_and_grad(x, beta):
    """
    Compute the negative soft-min value and its gradient.
    x: shape (2N,)
    beta: softness parameter
    """
    C = x.reshape(N, 2)

    # Pairwise distances
    diff = C[_I_IDX] - C[_J_IDX]
    L_flat = np.sqrt(np.sum(diff * diff, axis=1))

    # Handle zero distances
    L_flat_safe = np.where(L_flat > 1e-12, L_flat, 1.0)

    # Margins
    x_pairs = L_flat_safe / 2.0
    L_wall = np.linalg.norm(C, axis=1)
    x_walls = 1.0 - L_wall

    # Combine all margins
    x_all = np.concatenate([x_pairs, x_walls])

    # Numerically stable soft-min
    min_x = np.min(x_all)
    shifted_x = x_all - min_x
    exp_terms = np.exp(-beta * shifted_x)
    Z = np.sum(exp_terms)

    # Value: -soft_min
    val = -(min_x - (1.0 / beta) * np.log(Z))

    # Gradient
    P = exp_terms / Z  # Probabilities
    P_pairs = P[:_N_PAIRS]
    P_walls = P[_N_PAIRS:]

    grad = np.zeros_like(C)

    # Pairwise part
    U = diff / L_flat_safe[:, np.newaxis]  # Unit vectors from i to j
    W = P_pairs * 0.5

    # d(val)/dp_i = -W * U (for each pair i,j)
    # d(val)/dp_j = +W * U (for each pair i,j)
    np.add.at(grad, _I_IDX, -W[:, np.newaxis] * U)
    np.add.at(grad, _J_IDX, W[:, np.newaxis] * U)

    # Boundary part
    L_wall_safe = np.where(L_wall > 1e-12, L_wall, 1.0)
    U_walls = -C / L_wall_safe[:, np.newaxis]  # Outward normal is C/|C|, so inward is -C/|C|
    # x_i = 1 - ||p_i||, dx_i/dp_i = -p_i/||p_i|| = U_walls
    # d(val)/dp_i = -P_i * U_walls
    grad += -P_walls[:, np.newaxis] * U_walls

    return val, grad.flatten()


def _lattice_seed(spacing):
    """Generate a hexagonal lattice and pick N points inside the disk."""
    while True:
        k = int(1.0 / spacing) + 3
        rows = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                rows.append((x, y))
        pts = np.array(rows)
        rad = np.hypot(pts[:, 0], pts[:, 1])
        keep = rad <= 1.0 - spacing / 2.0
        pts = pts[keep]
        if len(pts) >= N:
            order = np.argsort(np.hypot(pts[:, 0], pts[:, 1]), kind="stable")
            return pts[order[:N]].copy()
        spacing *= 0.97


def _radial_ring_init(rng):
    """Generate initial points using concentric rings."""
    # Estimate ring spacing
    s = np.sqrt(2 * np.pi / (N * np.sqrt(3)))
    n_rings = int(1.0 / s) + 1
    points = []
    for r_idx in range(n_rings):
        r = (r_idx + 0.5) * s
        if r >= 1.0:
            break
        n_on_ring = max(1, int(2 * np.pi * r / s))
        angles = np.linspace(0, 2 * np.pi, n_on_ring, endpoint=False) + rng.random() * 0.1
        for a in angles:
            x = r * np.cos(a)
            y = r * np.sin(a)
            if np.hypot(x, y) < 1.0:
                points.append([x, y])
    pts_ring = np.array(points)
    if len(pts_ring) < N:
        r_rand = np.sqrt(rng.random(N - len(pts_ring)))
        a_rand = rng.random(N - len(pts_ring)) * 2 * np.pi
        extra = np.column_stack([r_rand * np.cos(a_rand), r_rand * np.sin(a_rand)])
        pts_ring = np.vstack([pts_ring, extra])
    elif len(pts_ring) > N:
        dists = np.linalg.norm(pts_ring, axis=1)
        indices = np.argsort(dists)
        pts_ring = pts_ring[indices[:N]]
    return pts_ring


def _optimize_lbfgs(points, beta, max_iter=200, time_limit=5.0):
    """Run L-BFGS-B optimization with a time limit."""
    start_time = time.time()

    def obj_and_grad(x):
        if time.time() - start_time > time_limit:
            raise TimeoutError()
        return _soft_min_obj_and_grad(x, beta)

    try:
        res = minimize(obj_and_grad, points.flatten(), method='L-BFGS-B', jac=True,
                       options={'maxiter': max_iter, 'gtol': 1e-9, 'ftol': 1e-15,
                                'maxls': 50, 'maxfun': 500})
        if res.success or res.status in [1, 2, 4]:
            return res.x.reshape(N, 2)
    except Exception:
        pass
    return points


def _gd_polish(points, beta, n_steps=1000, time_limit=3.0):
    """
    Gradient descent polish with adaptive step size.
    This helps escape shallow local minima that L-BFGS might get stuck in.
    """
    start_time = time.time()
    current = points.copy()
    best = points.copy()
    best_score = _score(points)
    lr = 1e-4
    patience = 0
    max_patience = 50

    for step in range(n_steps):
        if time.time() - start_time > time_limit:
            break

        val, grad = _soft_min_obj_and_grad(current.flatten(), beta)
        grad = grad.reshape(N, 2)

        # Adaptive step size
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break

        # Try the step
        candidate = current - lr * grad

        # Project back inside if needed
        norms = np.linalg.norm(candidate, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            candidate[mask] = candidate[mask] / norms[mask, None] * 0.999

        # Check if it improves
        cand_score = _score(candidate)
        if cand_score > best_score:
            current = candidate
            best = candidate.copy()
            best_score = cand_score
            patience = 0
        else:
            patience += 1
            lr *= 0.9
            if patience > max_patience:
                break

    return best


def _anneal_beta(points, betas, time_budget):
    """
    Anneal through a sequence of beta values, optimizing at each.
    Returns the best points found.
    """
    start_time = time.time()
    current = points.copy()
    best = points.copy()
    best_score = _score(points)

    for beta in betas:
        elapsed = time.time() - start_time
        if elapsed > time_budget:
            break

        remaining = time_budget - elapsed
        # Allocate more time to higher beta values
        frac = 1.0 / len(betas)
        t_alloc = remaining * frac
        if t_alloc < 0.5:
            t_alloc = 0.5

        current = _optimize_lbfgs(current, beta, max_iter=100, time_limit=t_alloc * 0.6)

        # Project back inside
        norms = np.linalg.norm(current, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            current[mask] = current[mask] / norms[mask, None] * 0.999

        # GD polish
        if time.time() - start_time < time_budget - 1.0:
            current = _gd_polish(current, beta, n_steps=500, time_limit=min(2.0, time_budget - (time.time() - start_time) - 0.5))

        score = _score(current)
        if score > best_score:
            best_score = score
            best = current.copy()

    return best, best_score


def construct():
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    # Initializations
    inits = []

    # 1. Hex lattice
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    inits.append(_lattice_seed(spacing))

    # 2. Hex lattice rotated
    pts_rot = _lattice_seed(spacing)
    angle = np.pi / 6
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    inits.append(pts_rot @ R.T)

    # 3. Radial rings
    inits.append(_radial_ring_init(rng))

    # 4. Another hex lattice with slightly different spacing
    spacing2 = spacing * 0.95
    inits.append(_lattice_seed(spacing2))

    best_points = None
    best_score = -np.inf

    # Progressive Beta Schedule
    betas = [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0]

    # Allocate time per initialization
    n_inits = len(inits)
    time_per_init = (DEADLINE - (time.time() - t0) - 10.0) / n_inits  # 10s reserved for final polish

    for init in inits:
        if time.time() - t0 > DEADLINE - 10.0:
            break

        current = init.copy()
        t_alloc = min(time_per_init, DEADLINE - (time.time() - t0) - 10.0)

        current, score = _anneal_beta(current, betas, t_alloc)

        if score > best_score:
            best_score = score
            best_points = current.copy()

    # Final high-precision polish on the best found
    if best_points is not None and time.time() - t0 < DEADLINE - 5:
        current = best_points.copy()
        # One more pass with very high beta and tight tolerance
        current = _optimize_lbfgs(current, 50000.0, max_iter=1000, time_limit=8.0)
        norms = np.linalg.norm(current, axis=1)
        mask = norms > 1.0
        if np.any(mask):
            current[mask] = current[mask] / norms[mask, None] * 0.999

        # GD polish at high beta
        if time.time() - t0 < DEADLINE - 2:
            current = _gd_polish(current, 50000.0, n_steps=2000, time_limit=min(5.0, DEADLINE - (time.time() - t0) - 1.0))

        score = _score(current)
        if score > best_score:
            best_score = score
            best_points = current

    # Fallback
    if best_points is None:
        best_points = _lattice_seed(spacing)

    return best_points
