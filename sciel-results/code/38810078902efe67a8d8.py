"""
Packing 300 equal circles in a unit circle.
Strategy:
  1. Fast hexagonal-lattice init.
  2. Hertzian-overlap dynamics (smooth, physically-motivated) for global structure.
  3. Epigraph L-BFGS-B polish (auxiliary variable t) for local precision.
  4. A few perturbed polish restarts if time remains.
"""
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
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(d2, np.inf)
    min_pair = np.sqrt(np.min(d2)) / 2.0
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
    mask = r <= 1.0
    pts = pts[mask]
    r = r[mask]
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
        mask = r <= 1.0
        pts = pts[mask]
        r = r[mask]
    order = np.argsort(r, kind="stable")
    return pts[order[:n]].copy()


# ---------------------------------------------------------------------------
# Hertzian dynamics (global structure)
# ---------------------------------------------------------------------------
def _hertzian_dynamics(pts, deadline):
    """
    Smooth Hertzian overlap^4 repulsion + boundary repulsion.
    Velocity-Verlet-ish integration with adaptive damping.
    """
    n = len(pts)
    pts = pts.copy().astype(np.float64)
    vel = np.zeros_like(pts)

    # Target radius for force computation (start slightly below current best)
    r_target = _score(pts) * 1.02
    # Force parameters
    k_repel = 10.0       # repulsion stiffness
    k_bound = 50.0       # boundary stiffness
    damping = 0.85       # velocity damping per step
    dt = 0.002           # time step
    max_dt = 0.005
    min_dt = 0.0005

    best_pts = pts.copy()
    best_score = _score(pts)

    # Preallocate for neighbor search (simple O(n^2) for clarity, n=300)
    it = 0
    while time.time() < deadline:
        # Compute forces
        diff = pts[:, None, :] - pts[None, :, :]
        d2 = np.einsum('ijk,ijk->ij', diff, diff)
        np.fill_diagonal(d2, np.inf)
        d = np.sqrt(d2)

        # Hertzian repulsion: F ~ (2*r_target - d)^4 when d < 2*r_target
        overlap = np.maximum(0.0, 2.0 * r_target - d)
        # Force magnitude per pair
        f_mag = k_repel * overlap ** 4
        # Direction: unit vector from j to i
        with np.errstate(divide='ignore', invalid='ignore'):
            unit = diff / d[:, :, None]
        unit = np.where(np.isfinite(unit), unit, 0.0)

        # Sum forces from pairs
        forces = np.sum(f_mag[:, :, None] * unit, axis=1)

        # Boundary repulsion: push points inward if too close to edge
        rad = np.hypot(pts[:, 0], pts[:, 1])
        boundary_overlap = np.maximum(0.0, rad - (1.0 - r_target))
        with np.errstate(divide='ignore', invalid='ignore'):
            inward = pts / rad[:, None]
        inward = np.where(np.isfinite(inward), inward, 0.0)
        forces -= k_bound * boundary_overlap[:, None] * inward

        # Update velocity with damping
        vel = damping * vel + dt * forces
        # Limit velocity
        v_mag = np.hypot(vel[:, 0], vel[:, 1])
        v_max = 0.05
        scale = np.where(v_mag > v_max, v_max / v_mag, 1.0)
        vel *= scale[:, None]

        # Update positions
        pts = pts + dt * vel

        # Clamp to unit circle
        rad = np.hypot(pts[:, 0], pts[:, 1])
        out = rad > 1.0
        if np.any(out):
            pts[out] = pts[out] / rad[out, None]

        # Track best
        it += 1
        if it % 50 == 0:
            sc = _score(pts)
            if sc > best_score:
                best_score = sc
                best_pts = pts.copy()
                # Slowly increase r_target as we improve
                r_target = max(r_target, sc * 1.01)

        # Adaptive dt: reduce if energy is high
        avg_overlap = np.mean(np.maximum(0.0, 2.0 * r_target - d))
        if avg_overlap > 0.02:
            dt = max(min_dt, dt * 0.9)
        elif avg_overlap < 0.005:
            dt = min(max_dt, dt * 1.05)

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
            options={'maxiter': 150, 'ftol': 1e-14, 'gtol': 1e-10},
        )
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
            # Continue polishing from improved point
            current_z = np.concatenate(([new_score], new_pts.flatten()))
        else:
            # No improvement; try one more round with tighter bounds
            if time.time() > deadline - 10:
                break
            # Perturb t slightly upward to encourage growth
            current_z[0] = min(1.0, current_z[0] * 1.001)

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

    # 2. Hertzian dynamics phase (global structure) — use ~50s
    dyn_deadline = start_time + 50.0
    try:
        dyn_pts, dyn_score = _hertzian_dynamics(pts, dyn_deadline)
        if dyn_score > best_score:
            best_score = dyn_score
            best_pts = dyn_pts.copy()
    except Exception:
        pass

    # 3. Epigraph polish (local precision) — use remaining time
    polish_deadline = deadline - 3.0
    try:
        p_pts, p_score = _polish(best_pts, polish_deadline)
        if p_score > best_score:
            best_score = p_score
            best_pts = p_pts.copy()
    except Exception:
        pass

    # 4. Additional polish restarts from perturbed copies while time remains.
    np.random.seed(12345)
    for _ in range(8):
        if time.time() >= polish_deadline:
            break
        perturb = best_score * 0.008
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
