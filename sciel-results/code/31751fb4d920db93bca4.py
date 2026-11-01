import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import Voronoi

N = 50
START_TIME = time.time()
TIME_LIMIT = 145.0   # safety margin for 150 s kill


def compute_radius(pts):
    if pts is None or pts.shape != (N, 3):
        return -1e6
    if not np.all(np.isfinite(pts)):
        return -1e6
    if np.any(np.abs(pts) > 0.5 + 1e-9):
        return -1e6
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    wall_dists = 0.5 - np.abs(pts)
    min_wall_dist = np.min(wall_dists)
    return min(min_pair_dist / 2.0, min_wall_dist)


def neg_radius(pts):
    return -compute_radius(pts)


def soft_min_radius(pts, beta=50.0):
    if np.any(np.abs(pts) > 0.5 + 1e-9):
        return 1e6
    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    pair_flat = (dists / 2.0)[np.triu_indices(N, k=1)]
    wall_flat = (0.5 - np.abs(pts)).flatten()
    all_d = np.concatenate([pair_flat, wall_flat])
    d_min = np.min(all_d)
    shifted = -beta * (all_d - d_min)
    shifted = np.clip(shifted, -700, 700)
    log_sum = np.log(np.sum(np.exp(shifted)))
    return d_min - log_sum / beta


def smooth_obj(x, beta):
    p = x.reshape(N, 3)
    s = soft_min_radius(p, beta)
    pen = 10.0 * np.sum(np.maximum(0.0, np.abs(p) - 0.4999) ** 2)
    return -s + pen


# ---------------- initializations ----------------
def get_grid_init(nx=5, ny=5, nz=2):
    xs = (np.arange(nx) + 0.5) / nx - 0.5
    ys = (np.arange(ny) + 0.5) / ny - 0.5
    zs = (np.arange(nz) + 0.5) / nz - 0.5
    pts = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    pts += np.random.normal(0, 0.001, size=(N, 3))
    return np.clip(pts, -0.49, 0.49)


def _prune_to_n(arr, thr=0.12):
    if len(arr) > N:
        sel, rem = [], arr.copy()
        for _ in range(N):
            if rem.size == 0:
                break
            pt = rem[0]
            sel.append(pt)
            rem = rem[np.linalg.norm(rem - pt, axis=1) > thr]
        arr = np.array(sel)
    if len(arr) < N:
        arr = np.vstack([arr, np.random.uniform(-0.45, 0.45, size=(N - len(arr), 3))])
    return arr[:N]


def get_fcc_init():
    pts, cs = [], 1.0 / 3.0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                x0, y0, z0 = i * cs, j * cs, k * cs
                pts += [[x0, y0, z0], [x0 + .5*cs, y0 + .5*cs, z0],
                        [x0 + .5*cs, y0, z0 + .5*cs], [x0, y0 + .5*cs, z0 + .5*cs]]
    pts = np.array(pts) - 0.5
    pts = pts[np.all(np.abs(pts) <= 0.5, axis=1)]
    return _prune_to_n(pts)


def get_bcc_init():
    pts, cs = [], 1.0 / 3.0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                x0, y0, z0 = i * cs, j * cs, k * cs
                pts += [[x0, y0, z0], [x0 + .5*cs, y0 + .5*cs, z0 + .5*cs]]
    pts = np.array(pts) - 0.5
    pts = pts[np.all(np.abs(pts) <= 0.5, axis=1)]
    return _prune_to_n(pts)


# ---------------- refinement machinery ----------------
def smooth_polish(pts, deadline, betas=(20.0, 50.0, 100.0, 200.0)):
    cur = np.clip(pts, -0.499, 0.499).copy()
    best = cur.copy()
    best_r = compute_radius(cur)
    for beta in betas:
        if time.time() > deadline - 5.0:
            break
        try:
            res = minimize(smooth_obj, cur.ravel(), args=(beta,),
                           method='L-BFGS-B', bounds=[(-0.5, 0.5)] * (N * 3),
                           options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 100})
            nc = res.x.reshape(N, 3)
            r = compute_radius(nc)
            if r > best_r:
                cur, best, best_r = nc, nc.copy(), r
        except Exception:
            pass
    return best


def repulsion_anneal(pts, deadline, iters=3000, t0=0.05, t1=0.001, seed=42):
    rng = np.random.RandomState(seed)
    cur = np.clip(pts, -0.499, 0.499).copy()
    best, best_r = cur.copy(), compute_radius(cur)
    cur_r = best_r
    for it in range(iters):
        if time.time() > deadline - 6.0:
            break
        t = t0 * (t1 / t0) ** (it / iters)
        d = cur[:, None, :] - cur[None, :, :]
        dists = np.linalg.norm(d, axis=2)
        np.fill_diagonal(dists, np.inf)
        safe = np.where(dists < 0.3, dists, 1.0) + 1e-10
        inv = np.where(dists < 0.3, 1.0 / safe ** 2, 0.0)
        forces = np.sum(d * inv[:, :, None], axis=1)
        w = np.clip(np.abs(cur) - 0.3, 0.0, None)
        forces -= 10.0 * w * np.sign(cur)
        noise = rng.normal(0, t, size=cur.shape)
        nc = np.clip(cur + 0.1 * forces + noise, -0.4999, 0.4999)
        nr = compute_radius(nc)
        if nr >= cur_r or rng.random() < np.exp((nr - cur_r) / (t * 0.01)):
            cur, cur_r = nc, nr
            if cur_r > best_r:
                best, best_r = cur.copy(), cur_r
    return best


def lloyd_step(centers, alpha=0.8):
    c = centers.copy()
    try:
        vor = Voronoi(c)
    except Exception:
        return c
    for i in range(len(c)):
        region = vor.regions[vor.point_region[i]]
        if len(region) == 0:
            continue
        vidx = [v for v in region if v >= 0]
        if len(vidx) < 3:
            continue
        verts = np.clip(vor.vertices[vidx], -0.5, 0.5)
        centroid = np.clip(np.mean(verts, axis=0), -0.4999, 0.4999)
        c[i] = (1 - alpha) * c[i] + alpha * centroid
    return np.clip(c, -0.4999, 0.4999)


def lloyd_optimize(pts, deadline, max_iter=120, alpha=0.8):
    c = np.clip(pts, -0.499, 0.499).copy()
    best, best_r = c.copy(), compute_radius(c)
    for _ in range(max_iter):
        if time.time() > deadline - 8.0:
            break
        nc = lloyd_step(c, alpha)
        disp = np.max(np.linalg.norm(nc - c, axis=1))
        c = nc
        r = compute_radius(c)
        if r > best_r:
            best, best_r = c.copy(), r
        if disp < 1e-10:
            break
    return best


def local_search(pts, deadline, iters=4000):
    cur = np.clip(pts, -0.499, 0.499).copy()
    best, best_r = cur.copy(), compute_radius(cur)
    step, rng = 0.005, np.random.RandomState(123)
    for _ in range(iters):
        if time.time() > deadline - 3.0:
            break
        improved = False
        idx, coord = rng.randint(N), rng.randint(3)
        for delta in [step, -step, step * .5, -step * .5, step * .1, -step * .1]:
            nc = cur.copy()
            nc[idx, coord] += delta
            if np.all(np.abs(nc) <= 0.5):
                s = compute_radius(nc)
                if s > best_r + 1e-12:
                    cur, best_r, improved = nc, s, True
                    break
        if not improved and rng.random() < 0.02:
            step = max(step * 0.8, 1e-7)
    return best


def trajectory_polish(pts, deadline, T=60, lr=0.02, K=0.5):
    """Optimise a control sequence driving centers apart (continuous-control view)."""
    cur = np.clip(pts, -0.499, 0.499).copy()
    best, best_r = cur.copy(), compute_radius(cur)
    controls = np.zeros((T, N, 3))
    rng = np.random.RandomState(7)
    for step_i in range(T):
        if time.time() > deadline - 4.0:
            break
        d = cur[:, None, :] - cur[None, :, :]
        dists = np.linalg.norm(d, axis=2)
        np.fill_diagonal(dists, np.inf)
        safe = np.where(dists < 0.35, dists, 1.0) + 1e-10
        inv = np.where(dists < 0.35, 1.0 / safe ** 2, 0.0)
        force = np.sum(d * inv[:, :, None], axis=1)
        force -= K * np.clip(np.abs(cur) - 0.35, 0.0, None) * np.sign(cur)
        controls[step_i] = force
    # integrate
    for step_i in range(T):
        cur = np.clip(cur + lr * controls[step_i] + rng.normal(0, 0.0005, cur.shape), -0.4999, 0.4999)
        r = compute_radius(cur)
        if r > best_r:
            best, best_r = cur.copy(), r
    return best


def exact_polish(pts, deadline, maxiter=400):
    if time.time() > deadline - 2.0:
        return np.clip(pts, -0.499, 0.499)
    try:
        res = minimize(neg_radius, pts, method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-11, 'fatol': 1e-13, 'adaptive': True})
        return np.clip(res.x, -0.499, 0.499)
    except Exception:
        return np.clip(pts, -0.499, 0.499)


def refine(pts, deadline):
    cands = [pts]
    if time.time() < deadline - 25.0:
        cands.append(smooth_polish(pts, deadline))
    if time.time() < deadline - 20.0:
        cands.append(lloyd_optimize(pts, deadline))
    if time.time() < deadline - 12.0:
        cands.append(repulsion_anneal(pts, deadline))
    if time.time() < deadline - 8.0:
        cands.append(trajectory_polish(pts, deadline))
    best, best_r = None, -np.inf
    for c in cands:
        if c is None:
            continue
        c = np.clip(c, -0.499, 0.499)
        r = compute_radius(c)
        if r > best_r:
            best, best_r = c.copy(), r
    return local_search(best, deadline)


def construct():
    global START_TIME
    START_TIME = time.time()
    deadline = START_TIME + TIME_LIMIT
    inits = [get_grid_init(5, 5, 2), get_fcc_init(), get_bcc_init(),
             np.random.uniform(-0.45, 0.45, size=(N, 3))]
    while time.time() < deadline - 40.0:
        inits.append(np.random.uniform(-0.45, 0.45, size=(N, 3)))

    best_pts, best_r = None, -np.inf
    for i, init_pts in enumerate(inits):
        if time.time() > deadline - 18.0:
            break
        init_pts = np.clip(init_pts, -0.49, 0.49)
        refined = refine(init_pts, deadline)
        r = compute_radius(refined)
        if r > best_r:
            best_r, best_pts = r, refined.copy()

    if best_pts is None:
        return get_grid_init(5, 5, 2)

    if best_r > 0.135 and time.time() < deadline - 5.0:
        polished = exact_polish(best_pts, deadline, maxiter=600)
        r2 = compute_radius(polished)
        if r2 > best_r:
            best_r, best_pts = r2, polished.copy()

    return best_pts
