"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

DEADLINE_SECONDS = 115.0  # Safety margin before 150s limit

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def compute_radius(pts):
    """Max radius for centers pts (N,2)."""
    d_x = 0.5 - np.abs(pts[:, 0])
    d_y = 0.5 - np.abs(pts[:, 1])
    min_bound = float(np.min(np.minimum(d_x, d_y)))
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_inter = float(np.sqrt(np.min(dist_sq)))
    r = min(min_dist_inter / 2.0, min_bound)
    return r

# ---------------------------------------------------------------------------
# Parametric lattice representation (structural idea from inspiration 1)
# ---------------------------------------------------------------------------
def lattice_points(row_lengths, h_spacing, v_spacing, x_offsets, y_offset=0.0):
    """Build (N,2) centers from a hexagonal row pattern."""
    n_rows = len(row_lengths)
    total_height = (n_rows - 1) * v_spacing
    y_start = -total_height / 2.0 + y_offset
    pts = []
    for i, length in enumerate(row_lengths):
        y = y_start + i * v_spacing
        total_width = (length - 1) * h_spacing
        x_start = -total_width / 2.0 + x_offsets[i]
        for j in range(length):
            x = x_start + j * h_spacing
            pts.append([x, y])
    return np.array(pts[:N])

def lattice_objective(params, row_lengths):
    n_rows = len(row_lengths)
    h = params[0]
    v = params[1]
    x_offsets = params[2:2 + n_rows]
    y_offset = params[2 + n_rows]
    if h <= 1e-6 or v <= 1e-6:
        return 100.0
    pts = lattice_points(row_lengths, h, v, x_offsets, y_offset)
    d_x = 0.5 - np.abs(pts[:, 0])
    d_y = 0.5 - np.abs(pts[:, 1])
    min_bound = float(np.min(np.minimum(d_x, d_y)))
    if min_bound < -0.02:
        return 100.0 + abs(min_bound) * 10
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_inter = float(np.sqrt(np.min(dist_sq)))
    r = min(min_dist_inter / 2.0, min_bound)
    return -r

def optimize_lattice(row_lengths, start_time, deadline):
    """Global DE + Nelder-Mead refinement over lattice parameters."""
    n_rows = len(row_lengths)
    bounds = [(0.08, 0.22), (0.08, 0.22)] + [(-0.06, 0.06)] * n_rows + [(-0.06, 0.06)]
    h0 = 1.0 / (max(row_lengths) + 0.3)
    v0 = h0 * np.sqrt(3) / 2.0
    x_offs0 = [0.0] * n_rows
    for i in range(1, n_rows, 2):
        x_offs0[i] = h0 / 2.0
    x0 = np.array([h0, v0] + x_offs0 + [0.0])

    x_best = x0.copy()
    # Phase 1: differential evolution (global)
    if time.time() < deadline - 25:
        try:
            de = differential_evolution(
                lattice_objective, bounds, args=(row_lengths,),
                maxiter=120, popsize=12, tol=1e-8, seed=42,
                polish=False, init='latinhypercube')
            x_best = de.x
        except Exception:
            pass
    # Phase 2: Nelder-Mead (local refinement)
    if time.time() < deadline - 5:
        try:
            nm = minimize(lattice_objective, x_best, args=(row_lengths,),
                          method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-11, 'fatol': 1e-13})
            x_best = nm.x
        except Exception:
            pass
    h = x_best[0]; v = x_best[1]
    x_offs = x_best[2:2 + n_rows]; y_off = x_best[2 + n_rows]
    pts = lattice_points(row_lengths, h, v, x_offs, y_off)
    pts = np.clip(pts, -0.49999, 0.49999)
    return pts, compute_radius(pts)

# ---------------------------------------------------------------------------
# Smooth soft-min objective + free-point polish (refinement machinery)
# ---------------------------------------------------------------------------
def soft_min(vals, beta=50.0):
    vals = np.asarray(vals, dtype=np.float64)
    m = np.min(vals)
    if m < -1e6:
        return m
    shifted = vals - m
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return m - log_sum / beta

def smooth_neg_radius(x, beta=50.0):
    p = x.reshape(-1, 2)
    d_x = 0.5 - np.abs(p[:, 0])
    d_y = 0.5 - np.abs(p[:, 1])
    bound_vals = np.minimum(d_x, d_y)
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    dists = np.sqrt(np.maximum(dist_sq, 0.0))
    inter_vals = dists / 2.0
    all_vals = np.concatenate([bound_vals, inter_vals])
    return -soft_min(all_vals, beta)

def neg_r(x):
    p = x.reshape(-1, 2)
    d_x = 0.5 - np.abs(p[:, 0])
    d_y = 0.5 - np.abs(p[:, 1])
    min_bound = float(np.min(np.minimum(d_x, d_y)))
    if min_bound < 0:
        return -min_bound * 10 + 100.0
    diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_inter = float(np.sqrt(np.min(dist_sq)))
    r = min(min_dist_inter / 2.0, min_bound)
    return -r

def free_point_optimize(pts, start_time, deadline):
    """Powell + high-beta L-BFGS-B soft-min polish on free coordinates."""
    best_pts = pts.copy()
    best_r = compute_radius(best_pts)
    bounds = [(-0.5, 0.5)] * (2 * N)
    # Phase 1: Powell (robust, no gradient)
    if time.time() < deadline - 12:
        try:
            res = minimize(neg_r, best_pts.flatten(), method='Powell',
                           options={'maxiter': 150, 'xtol': 1e-13, 'ftol': 1e-15})
            cand = np.clip(res.x.reshape(-1, 2), -0.49999, 0.49999)
            cr = compute_radius(cand)
            if cr > best_r:
                best_r = cr; best_pts = cand.copy()
        except Exception:
            pass
    # Phase 2: L-BFGS-B with smooth soft-min, increasing beta for precision
    for beta in (50.0, 200.0, 1000.0):
        if time.time() > deadline - 6:
            break
        try:
            res = minimize(smooth_neg_radius, best_pts.flatten(),
                           args=(beta,), method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 400, 'maxfun': 40000, 'gtol': 1e-8})
            cand = np.clip(res.x.reshape(-1, 2), -0.49999, 0.49999)
            cr = compute_radius(cand)
            if cr > best_r:
                best_r = cr; best_pts = cand.copy()
        except Exception:
            pass
    return best_pts, best_r

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS

    # Candidate row-length patterns summing to N=50 (hexagonal-ish)
    patterns = [
        [7, 7, 7, 7, 7, 7, 6, 2],   # 8 rows
        [8, 8, 8, 8, 8, 8],          # 6 rows
        [6, 7, 7, 7, 7, 7, 7, 2],    # 8 rows alt
        [5, 7, 7, 7, 7, 7, 7, 3],    # 8 rows
        [6, 6, 6, 6, 6, 6, 6, 6, 2], # 9 rows
    ]

    best_centers = None
    best_val = -1.0

    for pat in patterns:
        if time.time() > deadline - 35:
            break
        if sum(pat) != N:
            continue
        try:
            pts, r = optimize_lattice(pat, start_time, deadline - 40)
        except Exception:
            continue
        if r > best_val:
            best_val = r
            best_centers = pts.copy()

    # Fallback: grid init if nothing found
    if best_centers is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = np.clip(grid[:N].copy(), -0.49999, 0.49999)
        best_val = compute_radius(best_centers)

    # Refinement: free-point optimize from the best lattice result
    if time.time() < deadline - 8:
        try:
            refined, r = free_point_optimize(best_centers, start_time, deadline - 8)
            if r > best_val:
                best_val = r
                best_centers = refined.copy()
        except Exception:
            pass

    # A couple of perturbation+refine cycles to escape local minima
    for seed in (777, 1234):
        if time.time() > deadline - 12:
            break
        try:
            np.random.seed(seed)
            pert = best_centers.copy() + np.random.normal(0, 0.01, size=best_centers.shape)
            pert = np.clip(pert, -0.49999, 0.49999)
            refined, r = free_point_optimize(pert, start_time, deadline - 12)
            if r > best_val:
                best_val = r
                best_centers = refined.copy()
        except Exception:
            pass

    # Final high-precision polish
    if time.time() < deadline - 5:
        try:
            refined, r = free_point_optimize(best_centers, start_time, deadline - 5)
            if r > best_val:
                best_val = r
                best_centers = refined.copy()
        except Exception:
            pass

    best_centers = np.clip(best_centers, -0.5 + 1e-9, 0.5 - 1e-9)
    return best_centers
# EVOLVE-BLOCK-END
