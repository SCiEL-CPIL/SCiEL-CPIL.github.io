"""Combined program: Simulated Annealing (global) + epigraph L-BFGS-B polish (local)."""
import numpy as np
import time
from scipy.optimize import minimize

N = 300
TOTAL_BUDGET = 145.0  # safety margin under the 150s wall-clock kill


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _score(pts):
    """Maximum feasible common radius for a configuration (double precision)."""
    n = len(pts)
    if n < 2:
        return 1.0 - np.max(np.hypot(pts[:, 0], pts[:, 1]))
    # Pairwise distances (vectorized)
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, np.inf)
    min_pair = np.sqrt(np.min(d2)) / 2.0
    # Boundary distance
    min_bound = 1.0 - np.max(np.hypot(pts[:, 0], pts[:, 1]))
    return float(min(min_pair, min_bound))


# ---------------------------------------------------------------------------
# Initialization: hexagonal lattice
# ---------------------------------------------------------------------------
def _hex_init(n=N):
    """Generate a hexagonal-lattice initialization and keep the n innermost points."""
    spacing = 0.105
    k = int(1.1 / spacing) + 2
    xs, ys = [], []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            xs.append(x)
            ys.append(y)
    pts = np.column_stack([xs, ys])
    r = np.hypot(pts[:, 0], pts[:, 1])
    pts = pts[r <= 1.0]
    r = r[r <= 1.0]
    if len(pts) < n:
        spacing = 0.09
        k = int(1.1 / spacing) + 2
        xs, ys = [], []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                xs.append(x)
                ys.append(y)
        pts = np.column_stack([xs, ys])
        r = np.hypot(pts[:, 0], pts[:, 1])
        pts = pts[r <= 1.0]
        r = r[r <= 1.0]
    order = np.argsort(r, kind="stable")
    return pts[order[:n]].copy()


# ---------------------------------------------------------------------------
# Simulated Annealing (global structure)
# ---------------------------------------------------------------------------
def _sa_optimize(pts, deadline):
    """Adaptive simulated annealing to build a good global structure."""
    n = len(pts)
    pts = pts.copy()
    rad = np.hypot(pts[:, 0], pts[:, 1])
    out = rad > 1.0
    if np.any(out):
        pts[out] = pts[out] / rad[out, None]

    best_pts = pts.copy()
    best_score = _score(pts)
    current_score = best_score

    T = 0.01
    cooling_rate = 0.999
    min_T = 1e-6
    reheat_T = 0.008
    reheat_every = 6000

    it = 0
    while time.time() < deadline:
        if T < min_T:
            break
        # Adaptive step: scale with current radius so moves are ~ a fraction of a circle
        step_size = max(0.0015, 0.18 * current_score)
        idx = np.random.randint(n)
        dx, dy = np.random.normal(0, step_size, 2)

        new_pts = pts
        # In-place update then undo on rejection (cheaper than a full copy)
        old_x, old_y = new_pts[idx, 0], new_pts[idx, 1]
        new_pts[idx, 0] = old_x + dx
        new_pts[idx, 1] = old_y + dy
        r = np.hypot(new_pts[idx, 0], new_pts[idx, 1])
        if r > 1.0:
            new_pts[idx, 0] = new_pts[idx, 0] / r
            new_pts[idx, 1] = new_pts[idx, 1] / r

        new_score = _score(new_pts)
        delta = new_score - current_score
        if delta > 0 or np.random.rand() < np.exp(delta / T):
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_pts = new_pts.copy()
        else:
            new_pts[idx, 0] = old_x
            new_pts[idx, 1] = old_y

        T *= cooling_rate
        it += 1
        if it % reheat_every == 0:
            T = max(T, reheat_T)  # periodic reheat to escape local minima

    return best_pts, best_score


# ---------------------------------------------------------------------------
# Epigraph L-BFGS-B polish (local precision)
# ---------------------------------------------------------------------------
def _polish(pts, deadline):
    """Polish with L-BFGS-B on the epigraph formulation (auxiliary variable t)."""
    n = len(pts)
    pts = pts.copy()
    rad = np.hypot(pts[:, 0], pts[:, 1])
    out = rad > 1.0
    if np.any(out):
        pts[out] = pts[out] / rad[out, None]

    best_pts = pts.copy()
    best_score = _score(pts)
    t0 = best_score * 0.99

    # Precompute unit-difference structure lazily inside the objective.
    def objective_and_grad(z):
        t = z[0]
        p = z[1:].reshape(n, 2)

        r = np.hypot(p[:, 0], p[:, 1])
        mask = r > 1.0
        if np.any(mask):
            p = p.copy()
            p[mask] = p[mask] / r[mask, None]
            r = r.copy()
            r[mask] = 1.0

        diff = p[:, None, :] - p[None, :, :]
        d2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(d2, 1e10)
        d = np.sqrt(d2)

        b = 1.0 - r

        viol_pair = np.maximum(0.0, 2.0 * t - d)
        viol_bound = np.maximum(0.0, t - b)

        obj = -t + 0.5 * (np.sum(viol_pair * viol_pair) + np.sum(viol_bound * viol_bound))

        grad = np.zeros_like(z)
        with np.errstate(divide='ignore', invalid='ignore'):
            unit_vec = diff / d[:, :, None]
        unit_vec = np.where(np.isfinite(unit_vec), unit_vec, 0.0)

        w_pair = 2.0 * viol_pair
        grad_p_pair = np.sum(w_pair[:, :, None] * unit_vec, axis=1)

        with np.errstate(divide='ignore', invalid='ignore'):
            inward_vec = p / r[:, None]
        inward_vec = np.where(np.isfinite(inward_vec), inward_vec, 0.0)

        grad_p_bound = viol_bound[:, None] * inward_vec
        grad_p = -grad_p_pair + grad_p_bound

        grad[0] = -1.0 + 4.0 * np.sum(viol_pair) + np.sum(viol_bound)
        grad[1:] = grad_p.reshape(-1)
        return obj, grad

    def objective_only(z):
        return objective_and_grad(z)[0]

    def grad_only(z):
        return objective_and_grad(z)[1]

    z0 = np.concatenate(([t0], pts.flatten()))
    bounds = [(t0 * 0.5, 1.0)] + [(-1.0, 1.0)] * (2 * n)
    current_z = z0.copy()

    while time.time() < deadline:
        res = minimize(
            objective_only,
            current_z,
            jac=grad_only,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 120, 'ftol': 1e-14, 'gtol': 1e-10},
        )
        if res.success:
            current_z = res.x
            new_pts = current_z[1:].reshape(n, 2)
            r = np.hypot(new_pts[:, 0], new_pts[:, 1])
            out = r > 1.0
            if np.any(out):
                new_pts = new_pts.copy()
                new_pts[out] = new_pts[out] / r[out, None]
            new_score = _score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
            else:
                break
        else:
            current_z = res.x

    return best_pts, best_score


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def construct():
    start_time = time.time()
    deadline = start_time + TOTAL_BUDGET

    # 1. Initialize
    pts = _hex_init(N)
    rad = np.hypot(pts[:, 0], pts[:, 1])
    out = rad > 1.0
    if np.any(out):
        pts[out] = pts[out] / rad[out, None]

    best_pts = pts.copy()
    best_score = _score(pts)

    # 2. Simulated Annealing phase (global structure) — use ~70s
    sa_deadline = start_time + 70.0
    try:
        sa_pts, sa_score = _sa_optimize(pts, sa_deadline)
        if sa_score > best_score:
            best_score = sa_score
            best_pts = sa_pts.copy()
    except Exception:
        pass

    # 3. Multi-restart epigraph polish (local precision) — use remaining time
    polish_deadline = deadline - 3.0
    # First, a direct polish of the SA result.
    try:
        p_pts, p_score = _polish(best_pts, polish_deadline)
        if p_score > best_score:
            best_score = p_score
            best_pts = p_pts.copy()
    except Exception:
        pass

    # Additional polish restarts from perturbed copies while time remains.
    np.random.seed(12345)
    for _ in range(6):
        if time.time() >= polish_deadline:
            break
        perturb = best_score * 0.01
        cand = best_pts.copy()
        cand += np.random.normal(0, perturb, size=cand.shape)
        r = np.hypot(cand[:, 0], cand[:, 1])
        out = r > 1.0
        if np.any(out):
            cand[out] = cand[out] / r[out, None]
        try:
            p_pts, p_score = _polish(cand, polish_deadline)
            if p_score > best_score:
                best_score = p_score
                best_pts = p_pts.copy()
        except Exception:
            pass

    # Final safety: ensure the returned configuration is valid.
    r = np.hypot(best_pts[:, 0], best_pts[:, 1])
    out = r > 1.0
    if np.any(out):
        best_pts = best_pts.copy()
        best_pts[out] = best_pts[out] / r[out, None]

    return best_pts
