"""Pack 50 equal spheres in the unit cube [-1/2,1/2]^3 to maximise common radius."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
HALF = 0.5
# Best known reference radius (Pfoertner)
REF_R = 0.1359545126417


def _score(centers):
    """Raw radius: min over pair distances/2 and wall distances."""
    # wall term
    wall = HALF - np.abs(centers)
    wall_min = wall.min()
    # pair distances
    diff = centers[:, None, :] - centers[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, 1)
    pair_min = d[iu].min()
    return 0.5 * pair_min, wall_min


def _radius(centers):
    pm, wm = _score(centers)
    return min(pm, wm)


def _soft_min(values, beta):
    """Soft minimum: approx min(values) using log-sum-exp trick.
    soft_min(v) = -1/beta * log(sum(exp(-beta * v)))
    As beta -> inf, this approaches the true minimum.
    """
    v = np.asarray(values, dtype=float)
    v_min = v.min()
    # For numerical stability, subtract min before exponentiating
    exp_terms = np.exp(-beta * (v - v_min))
    log_sum = np.log(np.sum(exp_terms))
    return v_min - log_sum / beta


def _objective_flat(x, beta):
    """Smooth surrogate: minimise negative soft-min radius.
    We want to maximize r = min(pair_min/2, wall_min).
    Use soft-min to get a smooth approximation.
    """
    c = x.reshape(N, 3)
    diff = c[:, None, :] - c[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, 1)
    pd = d[iu] / 2.0
    wall = HALF - np.abs(c)
    w = wall.min(axis=1)
    
    # Soft-min over all pair distances/2 and all wall distances
    all_vals = np.concatenate([pd, w])
    r_smooth = _soft_min(all_vals, beta)
    return -r_smooth


def _init_grid():
    """Simple cubic grid initialization."""
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].astype(float)


def _init_bcc():
    """Body-centered cubic lattice initialization, scaled to fit N points."""
    # BCC has 2 points per conventional cell
    # For N=50, we need 25 conventional cells
    # Use a supercell approach
    n_cells = N // 2
    # Arrange in a roughly cubic supercell
    side = int(np.ceil(n_cells ** (1.0/3.0)))
    points = []
    for iz in range(side):
        for iy in range(side):
            for ix in range(side):
                if len(points) >= N:
                    break
                # Corner point
                points.append([ix, iy, iz])
                if len(points) < N:
                    # Body center
                    points.append([ix + 0.5, iy + 0.5, iz + 0.5])
    pts = np.array(points[:N], dtype=float)
    # Scale to fit in [-0.5, 0.5]
    max_coord = pts.max()
    scale = (HALF - 0.01) / max_coord if max_coord > 0 else 1.0
    pts = pts * scale - scale * max_coord / 2.0
    return pts


def _init_random(seed):
    rng = np.random.default_rng(seed)
    return (HALF * (1.0 - 2 * REF_R)) * (2.0 * rng.random((N, 3)) - 1.0)


def _optimize(x0, beta, deadline, maxiter):
    x0 = x0.reshape(-1).astype(float)
    best_x = x0.copy()
    best_val = _radius(x0.reshape(N, 3))
    t_end = deadline
    for it in range(maxiter):
        if time.time() > t_end - 2.0:
            break
        try:
            res = minimize(_objective_flat, x0, args=(beta,), method='L-BFGS-B',
                           options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50})
        except Exception:
            break
        x0 = res.x
        val = _radius(x0.reshape(N, 3))
        if val > best_val:
            best_val = val
            best_x = x0.copy()
        # stop if very close to reference and no room
        if best_val >= REF_R - 1e-10:
            break
    return best_x, best_val


def _polish(x0, deadline):
    """High-precision polish: use exact objective with very tight tolerances."""
    def _exact_neg_radius(x):
        c = x.reshape(N, 3)
        diff = c[:, None, :] - c[None, :, :]
        d = np.linalg.norm(diff, axis=2)
        iu = np.triu_indices(N, 1)
        pair_min = d[iu].min() / 2.0
        wall = HALF - np.abs(c)
        wall_min = wall.min()
        return -min(pair_min, wall_min)
    
    x0 = x0.reshape(-1).astype(float)
    best_x = x0.copy()
    best_val = _radius(x0.reshape(N, 3))
    
    # Use multiple restarts of the polish from slightly perturbed positions
    rng = np.random.default_rng(42)
    for trial in range(5):
        if time.time() > deadline - 1.0:
            break
        if trial > 0:
            perturb = 1e-6 * rng.standard_normal(x0.shape)
            x_cur = x0 + perturb
        else:
            x_cur = x0.copy()
        
        try:
            res = minimize(_exact_neg_radius, x_cur, method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-16, 'gtol': 1e-12})
        except Exception:
            continue
        x_cur = res.x
        val = _radius(x_cur.reshape(N, 3))
        if val > best_val:
            best_val = val
            best_x = x_cur.copy()
    
    return best_x, best_val


def construct():
    deadline = time.time() + 145.0  # safety margin vs 150s kill
    t_start = time.time()
    
    # Generate diverse initializations
    best_x = _init_grid()
    best_val = _radius(best_x)
    
    bcc_init = _init_bcc()
    bcc_val = _radius(bcc_init)
    if bcc_val > best_val:
        best_val = bcc_val
        best_x = bcc_init
    
    rng = np.random.default_rng(0)
    starts = [best_x.copy()]
    for s in range(8):
        starts.append(_init_random(s * 1000 + 7))
    starts.append(bcc_init.copy())
    
    # Beta schedule: start coarse, refine gradually
    beta_schedule = [15.0, 30.0, 60.0, 120.0, 250.0]
    
    for start in starts:
        if time.time() > deadline - 10.0:
            break
        cur = start.copy()
        for beta in beta_schedule:
            if time.time() > deadline - 10.0:
                break
            cur, val = _optimize(cur, beta, deadline, maxiter=1)
            if val > best_val:
                best_val = val
                best_x = cur.copy()
    
    # Final high-precision polish
    if time.time() < deadline - 3.0:
        best_x, val = _polish(best_x, deadline)
        if val > best_val:
            best_val = val
    
    # Safety: clip to be strictly inside
    out = best_x.reshape(N, 3)
    out = np.clip(out, -HALF + 1e-10, HALF - 1e-10)
    return out


if __name__ == "__main__":
    c = construct()
    print("radius:", _radius(c))
    print("shape:", c.shape)
