"""Pack 200 equal circles in a unit square via KDTree dynamics + active-set polish."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import KDTree

N = 200
DEADLINE = 150.0
MARGIN = 15.0


def _score(pts):
    """Largest common radius for a set of centres (raw value)."""
    if pts is None or len(pts) != N or pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    wall = 0.5 - np.max(np.abs(pts), axis=1)
    wall_min = float(np.min(wall))
    if wall_min <= 0:
        return 0.0
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    pair_min = float(np.min(d)) / 2.0
    return min(wall_min, pair_min)


def _soft_min(pts, lam):
    """Compute soft-min of all distance terms (pairwise/2 and wall distances)."""
    wall = 0.5 - np.max(np.abs(pts), axis=1)
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    pair = d / 2.0
    terms = np.concatenate([wall, pair.ravel()])
    terms = np.clip(terms, -1e6, 1e6)
    m = np.min(terms)
    z = np.exp(-lam * (terms - m))
    r_soft = m + (1.0 / lam) * np.log(np.sum(z))
    return r_soft


def _soft_min_neg(pts, lam):
    """Objective to minimize: -soft_min."""
    return -_soft_min(pts, lam)


def _soft_min_grad(pts, lam):
    """Gradient of -soft_min wrt pts."""
    pts = np.asarray(pts)
    wall = 0.5 - np.max(np.abs(pts), axis=1)
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    pair = d / 2.0
    terms = np.concatenate([wall, pair.ravel()])
    terms = np.clip(terms, -1e6, 1e6)
    m = np.min(terms)
    e = np.exp(-lam * (terms - m))
    w = e / np.sum(e)

    # Wall gradient
    g_wall = np.zeros_like(pts)
    ax = np.abs(pts[:, 0])
    ay = np.abs(pts[:, 1])
    mx = ax > ay
    g_wall[mx, 0] = -np.sign(pts[mx, 0])
    my = ay > ax
    g_wall[my, 1] = -np.sign(pts[my, 1])
    tie = (ax == ay) & (ax > 0)
    g_wall[tie, 0] = -0.5 * np.sign(pts[tie, 0])
    g_wall[tie, 1] = -0.5 * np.sign(pts[tie, 1])

    # Pair gradient
    w_pair = w[N:].reshape(N, N)
    w_pair = w_pair * (1.0 - np.eye(N))
    diff = pts[:, None, :] - pts[None, :, :]
    dmat = np.sqrt((diff ** 2).sum(-1))
    np.fill_diagonal(dmat, np.inf)
    factor = w_pair / (2.0 * dmat)
    g_pair = (factor[..., None] * diff).sum(axis=1)

    g_terms = g_wall + g_pair
    return -g_terms


def _get_active_pairs(pts, r, tol=1e-6):
    """Get pairs with distance < 2r + tol and wall contacts with dist < r + tol."""
    tree = KDTree(pts)
    pairs = []
    for i in range(N):
        nearby = tree.query_ball_point(pts[i], 2.0 * r + tol)
        for j in nearby:
            if j > i:
                d = np.linalg.norm(pts[i] - pts[j])
                if d < 2.0 * r + tol:
                    pairs.append((i, j))
    wall_contacts = []
    for i in range(N):
        wd = 0.5 - np.max(np.abs(pts[i]))
        if wd < r + tol:
            wall_contacts.append(i)
    return pairs, wall_contacts


def _hex_init():
    """Hexagonal lattice initialization with varied spacing."""
    best = None
    best_score = 0
    for s in np.linspace(0.035, 0.085, 300):
        vstep = s * np.sqrt(3) / 2.0
        pts = []
        x = -0.5 + s / 2.0
        row = 0
        y = -0.5 + vstep / 2.0
        while y <= 0.5:
            offset = (s / 2.0) if (row % 2) else 0.0
            xx = x + offset
            while xx <= 0.5:
                pts.append((xx, y))
                xx += s
            y += vstep
            row += 1
        pts = np.array(pts)
        if len(pts) >= N:
            r2 = (pts ** 2).sum(1)
            idx = np.argsort(r2)
            pts = pts[idx[:N]]
            sc = _score(pts)
            if sc > 0 and sc > best_score:
                best = pts.copy()
                best_score = sc
    if best is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    return best


def _random_cluster_init(rng=None):
    """Initialize with random clusters to escape local minima."""
    if rng is None:
        rng = np.random.default_rng(42)
    n_clusters = 8
    pts = []
    for c in range(n_clusters):
        cx = rng.uniform(-0.4, 0.4)
        cy = rng.uniform(-0.4, 0.4)
        n_in_cluster = N // n_clusters + (1 if c < N % n_clusters else 0)
        for k in range(n_in_cluster):
            x = cx + rng.normal(0, 0.05)
            y = cy + rng.normal(0, 0.05)
            x = np.clip(x, -0.48, 0.48)
            y = np.clip(y, -0.48, 0.48)
            pts.append((x, y))
    pts = np.array(pts[:N])
    return pts


def _kd_dynamics(pts, t_budget=45.0):
    """
    Fast force-directed dynamics using KDTree for neighbor lookup.
    Only computes forces from nearby neighbors, enabling many more iterations.
    """
    pts = pts.astype(np.float64).copy()
    vel = np.zeros_like(pts)
    n = len(pts)

    r_target = _score(pts) * 1.002
    k_rep = 100.0
    p_rep = 3
    damping = 0.92
    dt = 0.001
    dt_min = 1e-7
    dt_max = 0.005

    t0 = time.time()
    max_time = t_budget
    search_radius = 2.5 * r_target

    for step in range(200000):
        if time.time() - t0 > max_time:
            break

        forces = np.zeros_like(pts)

        # Wall forces
        ax = np.abs(pts[:, 0])
        ay = np.abs(pts[:, 1])
        wall_dist = 0.5 - np.maximum(ax, ay)
        for i in range(n):
            if ax[i] >= ay[i]:
                if wall_dist[i] < r_target:
                    overlap = r_target - wall_dist[i]
                    f = k_rep * overlap ** p_rep
                    forces[i, 0] += -np.sign(pts[i, 0]) * f
            else:
                if wall_dist[i] < r_target:
                    overlap = r_target - wall_dist[i]
                    f = k_rep * overlap ** p_rep
                    forces[i, 1] += -np.sign(pts[i, 1]) * f

        # Pairwise repulsion via KDTree
        tree = KDTree(pts)
        for i in range(n):
            nearby = tree.query_ball_point(pts[i], search_radius)
            for j in nearby:
                if j <= i:
                    continue
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    continue
                if dist < 2.0 * r_target:
                    overlap = 2.0 * r_target - dist
                    f_mag = k_rep * overlap ** p_rep
                    unit = diff / dist
                    forces[i] += f_mag * unit
                    forces[j] -= f_mag * unit

        # Symplectic Euler
        vel = vel * damping + forces * dt
        pts = pts + vel * dt
        pts = np.clip(pts, -0.5, 0.5)

        v_max = np.max(np.abs(vel))
        if v_max > 0.05:
            dt = max(dt_min, dt * 0.5)
        elif v_max < 0.005:
            dt = min(dt_max, dt * 1.2)

        if v_max < 1e-6 and step > 200:
            break

    return pts


def _gradient_polish(pts, t_budget=35.0):
    """Polish with L-BFGS-B using soft-min objective with annealing."""
    pts = pts.astype(np.float64).copy()
    lo = np.full(N * 2, -0.5)
    hi = np.full(N * 2, 0.5)

    t0 = time.time()
    max_time = t_budget

    lam_schedule = [80.0, 160.0, 320.0, 640.0, 1280.0, 2560.0, 5120.0]

    cur = pts.copy()
    for lam in lam_schedule:
        if time.time() - t0 > max_time:
            break

        def f(x):
            return _soft_min_neg(x.reshape(N, 2), lam)

        def g(x):
            return _soft_min_grad(x.reshape(N, 2), lam).ravel()

        try:
            res = minimize(
                f, cur.ravel(), jac=g, method="L-BFGS-B",
                bounds=list(zip(lo, hi)),
                options={"maxiter": 400, "maxfun": 2500, "ftol": 1e-13, "gtol": 1e-11},
            )
            cand = res.x.reshape(N, 2)
            cand = np.clip(cand, -0.5, 0.5)
            if _score(cand) >= _score(cur):
                cur = cand
        except Exception:
            pass

    return cur


def _active_set_polish(pts, t_budget=25.0):
    """
    Focused polish: identify active constraints, then optimize only those
    with tight tolerance. Multiple rounds with perturbation to escape.
    """
    pts = pts.astype(np.float64).copy()
    t0 = time.time()
    max_time = t_budget

    cur = pts.copy()
    cur_score = _score(cur)
    if cur_score <= 0:
        return cur

    lo = np.full(N * 2, -0.5)
    hi = np.full(N * 2, 0.5)

    for round_num in range(5):
        if time.time() - t0 > max_time:
            break

        r = _score(cur)
        if r <= 0:
            break

        pairs, wall_contacts = _get_active_pairs(cur, r, tol=1e-5)
        if len(pairs) == 0 and len(wall_contacts) == 0:
            break

        # Use very sharp lambda for focused optimization
        for lam in [8000.0, 16000.0, 32000.0]:
            if time.time() - t0 > max_time:
                break

            def f(x):
                return _soft_min_neg(x.reshape(N, 2), lam)

            def g(x):
                return _soft_min_grad(x.reshape(N, 2), lam).ravel()

            try:
                res = minimize(
                    f, cur.ravel(), jac=g, method="L-BFGS-B",
                    bounds=list(zip(lo, hi)),
                    options={"maxiter": 600, "maxfun": 4000, "ftol": 1e-15, "gtol": 1e-13},
                )
                cand = res.x.reshape(N, 2)
                cand = np.clip(cand, -0.5, 0.5)
                if _score(cand) > cur_score:
                    cur = cand
                    cur_score = _score(cur)
            except Exception:
                pass

        # Perturbation: nudge active points slightly to escape local minima
        if round_num < 3 and time.time() - t0 < max_time - 5.0:
            rng = np.random.default_rng(42 + round_num)
            perturbed = cur.copy()
            for i, j in pairs[:20]:
                direction = rng.normal(size=2)
                direction /= (np.linalg.norm(direction) + 1e-10)
                perturbed[i] += 0.0005 * direction
                perturbed[j] -= 0.0005 * direction
            for i in wall_contacts[:10]:
                ax = np.abs(perturbed[i, 0])
                ay = np.abs(perturbed[i, 1])
                if ax >= ay:
                    perturbed[i, 0] -= 0.0003 * np.sign(perturbed[i, 0])
                else:
                    perturbed[i, 1] -= 0.0003 * np.sign(perturbed[i, 1])
            perturbed = np.clip(perturbed, -0.5, 0.5)
            if _score(perturbed) > cur_score:
                cur = perturbed
                cur_score = _score(cur)

    return cur


def construct():
    t0 = time.time()
    limit = DEADLINE - MARGIN

    # Phase 1: Try multiple initializations
    candidates = []
    
    # Hexagonal
    hex_pts = _hex_init().astype(np.float64)
    candidates.append(hex_pts)
    
    # Random cluster
    rng = np.random.default_rng(123)
    rc_pts = _random_cluster_init(rng)
    candidates.append(rc_pts)
    
    # Another random with different seed
    rng2 = np.random.default_rng(456)
    rc2_pts = _random_cluster_init(rng2)
    candidates.append(rc2_pts)

    best = None
    best_score = 0
    for cand in candidates:
        cand = np.clip(cand, -0.5, 0.5)
        sc = _score(cand)
        if sc > best_score:
            best = cand.copy()
            best_score = sc

    if best is None:
        best = hex_pts
        best_score = _score(best)

    # Phase 2: KDTree-based dynamics relaxation
    if time.time() - t0 < limit:
        t_dyn = min(45.0, limit - (time.time() - t0) - 5.0)
        if t_dyn > 5.0:
            dyn_pts = _kd_dynamics(best, t_budget=t_dyn)
            dyn_pts = np.clip(dyn_pts, -0.5, 0.5)
            dyn_score = _score(dyn_pts)
            if dyn_score > best_score:
                best = dyn_pts.copy()
                best_score = dyn_score

    # Phase 3: Gradient polish
    if time.time() - t0 < limit:
        t_pol = min(35.0, limit - (time.time() - t0) - 5.0)
        if t_pol > 5.0:
            pol_pts = _gradient_polish(best, t_budget=t_pol)
            pol_pts = np.clip(pol_pts, -0.5, 0.5)
            pol_score = _score(pol_pts)
            if pol_score > best_score:
                best = pol_pts.copy()
                best_score = pol_score

    # Phase 4: Active-set polish with perturbation
    if time.time() - t0 < limit:
        t_act = min(25.0, limit - (time.time() - t0) - 2.0)
        if t_act > 3.0:
            act_pts = _active_set_polish(best, t_budget=t_act)
            act_pts = np.clip(act_pts, -0.5, 0.5)
            act_score = _score(act_pts)
            if act_score > best_score:
                best = act_pts.copy()
                best_score = act_score

    # Final clamp
    best = np.clip(best, -0.5, 0.5)
    return best


if __name__ == "__main__":
    t = time.time()
    p = construct()
    print("elapsed", time.time() - t, "score", _score(p), "shape", p.shape)
