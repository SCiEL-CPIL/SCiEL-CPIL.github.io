"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _score(c):
    """
    Compute the actual score (largest common radius) for a configuration.
    c: (N, 3) array of centers.
    Returns the radius r.
    """
    if c.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(c)):
        return 0.0
    # Check boundaries: min distance to each face
    face_dists = 0.5 - np.abs(c)
    min_face = np.min(face_dists)
    if min_face < -1e-12:
        return 0.0
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    r = min(min_pair / 2.0, min_face)
    return r


def _soft_min_obj_and_grad(c_flat, beta):
    """
    Compute -soft_min(all_constraints) and its gradient w.r.t. c_flat.
    
    The soft min is: -log(sum(exp(-beta * d_k))) / beta
    where d_k are pairwise half-distances and face distances.
    
    We want to maximize min, so we minimize -soft_min.
    
    Returns (value, gradient) where gradient is w.r.t. c_flat (shape 3N).
    """
    c = c_flat.reshape(N, 3)
    
    # --- Pairwise distances / 2 ---
    # diff[i,j] = c[i] - c[j], shape (N, N, 3)
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)  # (N, N)
    np.fill_diagonal(dists, np.inf)
    
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu] / 2.0  # shape (M_pair,)
    
    # --- Face distances ---
    face_dists = 0.5 - np.abs(c)  # shape (N, 3)
    face_flat = face_dists.flatten()  # shape (3N,)
    
    # --- Combine ---
    all_dists = np.concatenate([pair_dists, face_flat])
    
    # Soft min computation
    d_min = np.min(all_dists)
    # weights: exp(-beta * (d_k - d_min)) / sum(...)
    shifted = -beta * (all_dists - d_min)
    # Clip for numerical stability
    shifted = np.clip(shifted, -500, 0)
    exp_terms = np.exp(shifted)
    sum_exp = np.sum(exp_terms)
    weights = exp_terms / sum_exp  # shape (M_total,)
    
    r_smooth = d_min - np.log(sum_exp) / beta
    value = -r_smooth
    
    # --- Gradient ---
    # d(-r_smooth)/d(c) = -d(r_smooth)/d(c)
    # r_smooth = d_min - log(sum_exp)/beta
    # dr_smooth/d(d_k) = delta_{k,argmin} - (1/beta) * weights[k]
    # But more precisely, since we shifted:
    # r_smooth = d_min - log(sum exp(-beta*(d_k - d_min)))/beta
    # Let S = sum exp(-beta*(d_k - d_min))
    # dr_smooth/d(d_k) = [delta_{k,k*} * (1 - 1/S) ... ] 
    # Actually simpler: r_smooth = -log(sum exp(-beta*d_k))/beta
    # dr_smooth/d(d_k) = (1/beta) * exp(-beta*d_k) / sum(exp(-beta*d_j)) = weights[k]
    # So dr_smooth/d(c) = sum_k weights[k] * d(d_k)/dc
    # And d(-r_smooth)/dc = -sum_k weights[k] * d(d_k)/dc
    
    grad = np.zeros(3 * N)
    
    # Pairwise part: d_k = |c_i - c_j| / 2 for pair (i,j)
    # d(d_k)/dc_i = (c_i - c_j) / (2 * |c_i - c_j|) = (c_i - c_j) / (4 * d_k)
    # d(d_k)/dc_j = (c_j - c_i) / (2 * |c_i - c_j|) = -(c_i - c_j) / (4 * d_k)
    
    n_pairs = len(pair_dists)
    for idx in range(n_pairs):
        w = weights[idx]
        if w < 1e-15:
            continue
        i, j = iu[0][idx], iu[1][idx]
        d_ij = dists[i, j]
        if d_ij < 1e-15:
            continue
        # d_k = d_ij / 2
        # d(d_k)/dc_i = (c_i - c_j) / (2 * d_ij)
        # d(d_k)/dc_j = (c_j - c_i) / (2 * d_ij)
        diff_ij = c[i] - c[j]
        factor = w / (2.0 * d_ij)
        grad[3*i:3*i+3] -= factor * diff_ij  # negative because we minimize -r_smooth
        grad[3*j:3*j+3] += factor * diff_ij
    
    # Face part: d_k = 0.5 - |c_ik|
    # d(d_k)/dc_ik = -sign(c_ik)
    for idx in range(3 * N):
        w = weights[n_pairs + idx]
        if w < 1e-15:
            continue
        i = idx // 3
        k = idx % 3
        sign_val = np.sign(c[i, k])
        if sign_val == 0:
            sign_val = 1.0
        # d(d_k)/dc_ik = -sign(c_ik)
        # d(-r_smooth)/dc_ik = -w * (-sign) = w * sign
        grad[idx] += w * sign_val
    
    return value, grad


def _generate_bcc_init(n_points):
    """
    Generate initial points on a BCC (body-centered cubic) lattice.
    BCC has 2 points per unit cell: corners + center.
    For a cube of side 1, we tile with small BCC cells.
    """
    # We want approximately n_points points.
    # BCC: in a cubic cell of side a, we have 2 atoms.
    # Total points ~ (1/a)^3 * 2
    # So a ~ (2/n_points)^(1/3)
    
    # Try different scales and pick the one closest to n_points
    best_points = None
    best_diff = float('inf')
    
    for scale in np.arange(0.5, 1.5, 0.01):
        a = scale * (2.0 / n_points) ** (1.0/3.0)
        n_cells = int(np.floor(1.0 / a))
        if n_cells < 1:
            continue
        
        points = []
        for ix in range(n_cells):
            for iy in range(n_cells):
                for iz in range(n_cells):
                    # Corner
                    cx = ix * a + 0.5
                    cy = iy * a + 0.5
                    cz = iz * a + 0.5
                    points.append([cx, cy, cz])
                    # Body center
                    points.append([cx + a/2, cy + a/2, cz + a/2])
        
        pts = np.array(points)
        # Keep only points inside [-0.5, 0.5]^3
        mask = np.all(pts >= -0.5 + 1e-6, axis=1) & np.all(pts <= 0.5 - 1e-6, axis=1)
        pts = pts[mask]
        
        if len(pts) >= n_points:
            # Take first n_points
            pts = pts[:n_points]
            diff = abs(len(pts) - n_points)
            if diff < best_diff:
                best_diff = diff
                best_points = pts.copy()
        elif len(pts) > 0:
            # Pad with random points if needed
            remaining = n_points - len(pts)
            rng = np.random.default_rng(42)
            extra = rng.uniform(-0.45, 0.45, size=(remaining, 3))
            pts = np.vstack([pts, extra])
            if best_points is None:
                best_points = pts.copy()
                best_diff = remaining
    
    if best_points is None:
        # Fallback: simple grid
        m = int(np.ceil(n_points ** (1.0/3.0)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_points = grid[:n_points].copy()
    
    return best_points


def _optimize_from_initial(c_init, beta, maxiter=300, ftol=1e-12, gtol=1e-10):
    """Run L-BFGS-B optimization with analytic gradient."""
    try:
        res = minimize(
            _soft_min_obj_and_grad,
            c_init.flatten(),
            args=(beta,),
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol}
        )
        return res.x.reshape(N, 3)
    except Exception:
        return c_init


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (150s limit)
    
    # BCC lattice initialization
    c0 = _generate_bcc_init(N)
    c0 = np.clip(c0, -0.49, 0.49)
    
    best_c = c0.copy()
    best_score = _score(best_c)
    
    # Multi-start optimization with beta annealing
    rng = np.random.default_rng(42)
    
    # Beta schedule: tighter continuation
    beta_schedule = [5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 400.0, 800.0]
    
    for beta in beta_schedule:
        if time.time() > deadline - 5.0:
            break
        
        # Use current best as starting point
        c_opt = _optimize_from_initial(best_c, beta, maxiter=200, ftol=1e-12, gtol=1e-10)
        c_opt = np.clip(c_opt, -0.4999, 0.4999)
        score = _score(c_opt)
        if score > best_score:
            best_score = score
            best_c = c_opt.copy()
        
        # Also try a perturbed version
        if time.time() < deadline - 5.0:
            c_pert = best_c + rng.normal(0, 0.02, size=(N, 3))
            c_pert = np.clip(c_pert, -0.49, 0.49)
            c_opt2 = _optimize_from_initial(c_pert, beta, maxiter=150, ftol=1e-12, gtol=1e-10)
            c_opt2 = np.clip(c_opt2, -0.4999, 0.4999)
            score2 = _score(c_opt2)
            if score2 > best_score:
                best_score = score2
                best_c = c_opt2.copy()
    
    # Final polish passes with very high beta
    for polish_beta in [1000.0, 2000.0, 5000.0]:
        if time.time() > deadline - 3.0:
            break
        c_polished = _optimize_from_initial(best_c, polish_beta, maxiter=500, ftol=1e-14, gtol=1e-12)
        c_polished = np.clip(c_polished, -0.49999, 0.49999)
        score = _score(c_polished)
        if score > best_score:
            best_score = score
            best_c = c_polished.copy()
    
    return best_c
# EVOLVE-BLOCK-END
