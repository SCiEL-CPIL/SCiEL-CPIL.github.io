import time
import numpy as np
from scipy.optimize import minimize

N = 50
# 150s kill limit; keep a hard safety margin
_DEADLINE = time.time() + 140.0

# Precompute unique pair indices (upper triangle)
_IU = np.triu_indices(N, k=1)
_I_IDX = _IU[0]
_M_IDX = _IU[1]


def _score(pts):
    """Compute the valid packing radius (or 0 if invalid)."""
    if pts is None or len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    if pts.shape != (N, 2):
        return 0.0

    dists_origin = np.linalg.norm(pts, axis=1)
    if np.any(dists_origin > 1.0 + 1e-9):
        return 0.0

    r_cont = 1.0 - np.max(dists_origin)
    if r_cont < 0:
        return 0.0

    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist_mat, np.inf)
    r_pair = 0.5 * np.min(dist_mat)

    return min(r_cont, r_pair)


def _force_directed_relax(pts, max_iter=600, dt=0.004, time_budget=6.0):
    """
    Smooth force-directed relaxation: pairwise repulsion + soft boundary push.
    This nudges a rough init into a good local basin before the L-BFGS-B pass.
    """
    start = time.time()
    pts = pts.astype(np.float64).copy()
    n = len(pts)
    for it in range(max_iter):
        if time.time() - start > time_budget:
            break

        # Pairwise repulsion: push apart when closer than a soft cutoff
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(d, np.inf)
        # Soft repulsion strength ~ 1/d^2, capped
        # Only act where d is small to keep it stable
        inv_d = np.where(d > 1e-8, 1.0 / d, 0.0)
        inv_d2 = inv_d * inv_d
        # Cap the force to avoid explosion
        force_mag = np.minimum(inv_d2, 50.0)
        # zero out diagonal
        np.fill_diagonal(force_mag, 0.0)
        force = force_mag[:, :, np.newaxis] * diff
        grad_pts = np.sum(force, axis=1)

        # Boundary: push inward when near/over the edge
        r_o = np.linalg.norm(pts, axis=1)
        safe_r = np.where(r_o < 1e-8, 1.0, r_o)
        # soft boundary force: grows as r_o approaches 1
        bnd = np.maximum(0.0, r_o - 0.85) ** 2
        bnd_vec = (bnd / safe_r)[:, np.newaxis] * pts
        grad_pts -= bnd_vec

        # Adaptive step: scale so max movement is bounded
        max_move = np.max(np.linalg.norm(grad_pts, axis=1))
        if max_move > 1e-12:
            step = dt * (1.0 / (1.0 + max_move * 0.5))
        else:
            step = dt
        pts += step * grad_pts

        # Project into unit disk (hard constraint)
        r_o = np.linalg.norm(pts, axis=1)
        mask = r_o > 1.0
        if np.any(mask):
            pts[mask] = pts[mask] / r_o[mask, np.newaxis]

    return pts


def _soft_min_obj_and_grad(flat, beta):
    """
    Soft-min objective + gradient for L-BFGS-B.
    f = (1/beta) * ln(sum(exp(-beta * x_k))) over critical distances x_k.
    Minimizing f == maximizing soft-min of x_k.
    """
    pts = flat.reshape(N, 2)

    dists_origin = np.linalg.norm(pts, axis=1)
    boundary_dists = 1.0 - dists_origin

    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_mat = np.linalg.norm(diff, axis=-1)

    pair_dists = dist_mat[_I_IDX, _M_IDX]

    all_dists = np.concatenate([boundary_dists, pair_dists / 2.0])

    y = -beta * all_dists
    max_y = np.max(y)
    y_shifted = y - max_y
    sum_exp = np.sum(np.exp(y_shifted))
    log_sum_exp = max_y + np.log(sum_exp)
    obj_val = log_sum_exp / beta

    weights = np.exp(y_shifted) / sum_exp
    grad_pts = np.zeros_like(pts)

    # Boundary terms
    w_boundary = weights[0:N]
    safe_dists = np.where(dists_origin < 1e-8, 1.0, dists_origin)
    grad_pts += (w_boundary / safe_dists)[:, np.newaxis] * pts

    # Pairwise terms
    w_pairs = weights[N:]
    d_pairs = dist_mat[_I_IDX, _M_IDX]
    safe_d_pairs = np.where(d_pairs < 1e-8, 1.0, d_pairs)
    vecs = pts[_I_IDX] - pts[_M_IDX]
    unit_vecs = vecs / safe_d_pairs[:, np.newaxis]

    grad_contrib_i = -0.5 * w_pairs[:, np.newaxis] * unit_vecs
    grad_contrib_m = 0.5 * w_pairs[:, np.newaxis] * unit_vecs

    np.add.at(grad_pts, _I_IDX, grad_contrib_i)
    np.add.at(grad_pts, _M_IDX, grad_contrib_m)

    return obj_val, grad_pts.flatten()


def _true_hex_init():
    """
    True hexagonal lattice: 1 + 6 + 12 + 18 + 13 = 50 points.
    Uses proper 60-degree geometry for the first full rings, truncates the last.
    """
    pts = [[0.0, 0.0]]

    # Full hex rings: ring k (k>=1) has 6k points at radius k * d
    # Use axial/hex geometry. For a hex lattice with nearest-neighbor spacing d,
    # ring k consists of 6k points. We place them on the hexagonal boundary.
    def hex_ring(k):
        """Return 6k points forming hexagonal ring k (radius k*d) with d=1."""
        # Hex ring: 6 sides, each of length k (in lattice steps).
        # Use the 6 direction vectors.
        pts_list = []
        # Directions in hex lattice (60 deg apart)
        # For a point at ring k, walk around.
        # Standard: ring k has points at angles; we generate by 6 edges.
        # Edge vectors (unit hex steps):
        # We'll generate the 6 corners and interpolate along edges.
        # Corners of hexagon ring k (radius k in axial coords):
        # (k,0),(0,k),(-k,k),(-k,0),(0,-k),(k,-k) in axial -> convert to cartesian
        # Axial to pixel (pointy-top): x = d*(q + r/2), y = d*(sqrt(3)/2 * r)
        s3 = np.sqrt(3.0) / 2.0
        corners_axial = [(k, 0), (0, k), (-k, k), (-k, 0), (0, -k), (k, -k)]
        # For each edge from corner i to corner i+1, place k points (excluding the end)
        for e in range(6):
            q1, r1 = corners_axial[e]
            q2, r2 = corners_axial[(e + 1) % 6]
            for t in range(k):
                q = q1 + (q2 - q1) * (t / k)
                r = r1 + (r2 - r1) * (t / k)
                x = (q + r / 2.0)
                y = s3 * r
                pts_list.append([x, y])
        return np.array(pts_list)

    # Full rings 1..3 give 6+12+18 = 36, plus center = 37. Need 13 more from ring 4 (which has 24).
    for k in [1, 2, 3]:
        pts.extend(hex_ring(k).tolist())

    # Ring 4 has 24 points; take 13 of them (evenly spaced)
    ring4 = hex_ring(4)
    idx = np.linspace(0, 23, 13, endpoint=False).astype(int)
    pts.extend(ring4[idx].tolist())

    pts = np.array(pts[:N])

    # Scale to fit nicely inside unit disk
    max_r = np.max(np.linalg.norm(pts, axis=1))
    if max_r > 0:
        pts = pts * (0.92 / max_r)
    return pts


def _ring_init(ring_counts, jitter=0.01):
    """Ring-based init: ring_counts[0] is center count (0 or 1)."""
    pts = []
    if ring_counts[0] > 0:
        pts.append([0.0, 0.0])
    num_rings = len(ring_counts) - 1
    if num_rings > 0:
        radii = np.linspace(0.25, 0.82, num_rings) if num_rings > 1 else [0.5]
        for k in range(num_rings):
            count = ring_counts[k + 1]
            if count <= 0:
                continue
            r = radii[k]
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
            angles += np.random.uniform(-0.05, 0.05, count)
            x = r * np.cos(angles)
            y = r * np.sin(angles)
            pts.extend(np.column_stack([x, y]))
    pts = np.array(pts)
    if jitter > 0:
        pts = pts + np.random.normal(0, jitter, pts.shape)
    # project to disk
    r_o = np.linalg.norm(pts, axis=1)
    mask = r_o > 1.0
    if np.any(mask):
        pts[mask] = pts[mask] / r_o[mask, np.newaxis]
    return pts[:N]


def _random_reject_init():
    """Random init with rejection to avoid gross overlaps."""
    pts = np.zeros((N, 2))
    pts[0] = [0.0, 0.0]
    placed = 1
    for i in range(1, N):
        for _ in range(200):
            ang = np.random.uniform(0, 2 * np.pi)
            rad = np.sqrt(np.random.uniform(0, 0.95))
            cand = np.array([rad * np.cos(ang), rad * np.sin(ang)])
            if np.all(np.linalg.norm(pts[:placed] - cand, axis=1) > 0.15):
                pts[i] = cand
                placed += 1
                break
        else:
            pts[i] = cand
    return pts


def _generate_inits():
    inits = []

    # 1. True hex lattice + small noise
    hex_pts = _true_hex_init()
    hex_pts = hex_pts + np.random.normal(0, 0.004, (N, 2))
    r_o = np.linalg.norm(hex_pts, axis=1)
    mask = r_o > 1.0
    if np.any(mask):
        hex_pts[mask] = hex_pts[mask] / r_o[mask, np.newaxis]
    inits.append(hex_pts)

    # 2. Ring configs
    ring_configs = [
        [1, 6, 12, 15, 16],
        [1, 6, 12, 12, 19],
        [1, 6, 12, 10, 21],
        [1, 6, 10, 14, 19],
        [1, 6, 12, 18, 13],
        [1, 6, 14, 14, 15],
        [1, 6, 12, 16, 15],
        [1, 6, 13, 13, 17],
        [1, 7, 12, 12, 18],
    ]
    for cfg in ring_configs:
        pts = _ring_init(cfg, jitter=0.008)
        inits.append(pts)

    # 3. Random-reject
    inits.append(_random_reject_init())

    return inits


def _optimize_with_lbfgsb(pts0, beta_schedule, max_time=10.0,
                          ftol=1e-12, gtol=1e-10, maxiter=800):
    start_time = time.time()
    current_pts = pts0.copy()
    for beta in beta_schedule:
        if time.time() - start_time > max_time:
            break
        try:
            res = minimize(
                _soft_min_obj_and_grad,
                current_pts.flatten(),
                args=(beta,),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol},
            )
            current_pts = res.x.reshape(N, 2)
        except Exception:
            break
        r_o = np.linalg.norm(current_pts, axis=1)
        mask = r_o > 1.0
        if np.any(mask):
            current_pts[mask] = current_pts[mask] / r_o[mask, np.newaxis]
    return current_pts


def construct():
    start_time = time.time()

    inits = _generate_inits()

    best_pts = inits[0].copy()
    best_score = _score(best_pts)

    # Main beta schedule (smooth ramp)
    beta_schedule = [4, 10, 25, 60, 150, 400]

    time_per_run = 11.0

    for i, init_pts in enumerate(inits):
        if time.time() > _DEADLINE - 18.0:
            break
        try:
            # Pre-relax with force-directed method to reach a good basin
            relaxed = _force_directed_relax(init_pts, max_iter=500, dt=0.004,
                                            time_budget=5.0)
            # Then soft-min L-BFGS-B polish
            opt_pts = _optimize_with_lbfgsb(relaxed, beta_schedule,
                                            max_time=time_per_run)
            opt_score = _score(opt_pts)
            if opt_score > best_score:
                best_score = opt_score
                best_pts = opt_pts.copy()
        except Exception:
            pass

    # Final high-precision polish
    if time.time() < _DEADLINE - 4.0:
        remaining = _DEADLINE - time.time() - 4.0
        polish_beta = [800, 2000, 6000, 15000]
        try:
            polished = _optimize_with_lbfgsb(best_pts, polish_beta,
                                             max_time=remaining,
                                             ftol=1e-14, gtol=1e-12, maxiter=1200)
            polished_score = _score(polished)
            if polished_score > best_score:
                best_pts = polished
                best_score = polished_score
        except Exception:
            pass

    final_score = _score(best_pts)
    if final_score <= 0:
        return inits[0]
    return best_pts


# EVOLVE-BLOCK-END
