"""Packing 50 equal circles in a unit circle using soft-min energy with ring structures."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 50
DEADLINE = 135.0  # safety margin before 150s kill


def _pairwise_indices(n):
    """Return array of (i, j) pairs for i < j."""
    iu = np.triu_indices(n, k=1)
    return np.column_stack((iu[0], iu[1]))


def _compute_score(P):
    """Calculate the radius score for a given set of points."""
    P = np.asarray(P)
    if P.shape != (N, 2) or not np.all(np.isfinite(P)):
        return 0.0
    
    # Distance to boundary
    d_bound = 1.0 - np.linalg.norm(P, axis=1)
    r_bound = np.min(d_bound)
    
    # Pairwise distances
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    r_pair = min_pair_dist / 2.0
    
    return max(0.0, min(r_pair, r_bound))


def _softmin_energy(P, r, beta=50.0):
    """
    Smooth energy using log-sum-exp soft-min of constraint violations.
    Lower energy = better packing. Returns 0 if all constraints satisfied.
    """
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    dist_origin = np.linalg.norm(P, axis=1)
    
    # Violations: how much each constraint is violated
    cc_viol = 2.0 * r - dist  # positive when overlapping
    cb_viol = dist_origin + r - 1.0  # positive when outside
    
    # Soft-min of violations using log-sum-exp
    # For each constraint, we want a smooth approximation of max(violation, 0)
    # Use: softplus(beta * violation) / beta as smooth max(violation, 0)
    def smooth_max(v, b):
        return np.logaddexp(0.0, b * v) / b
    
    penalty_cc = np.sum(smooth_max(cc_viol, beta))
    penalty_cb = np.sum(smooth_max(cb_viol, beta))
    
    return penalty_cc + penalty_cb


def _softmin_grad(P, r, beta=50.0):
    """
    Gradient of the soft-min energy function.
    """
    grad = np.zeros_like(P)
    
    diff = P[:, np.newaxis, :] - P[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    
    dist_origin = np.linalg.norm(P, axis=1)
    
    cc_viol = 2.0 * r - dist
    cb_viol = dist_origin + r - 1.0
    
    # Gradient of softplus(beta * v) / beta = sigmoid(beta * v)
    # For cc_viol: d/dP_i [sigmoid(beta*(2r - dist_ij))] * (P_i - P_j)/dist_ij
    sig_cc = 1.0 / (1.0 + np.exp(-beta * cc_viol))
    sig_cc[~np.isfinite(sig_cc)] = 0.0
    
    inv_dist = np.zeros_like(dist)
    np.divide(1.0, dist, out=inv_dist, where=dist > 1e-12)
    
    W_cc = sig_cc * inv_dist
    sum_W_cc = np.sum(W_cc, axis=1)
    term1_cc = (P.T @ W_cc).T
    term2_cc = P * sum_W_cc[:, np.newaxis]
    grad += term1_cc - term2_cc
    
    # For cb_viol: d/dP_i [sigmoid(beta*(|P_i| + r - 1))] * P_i/|P_i|
    sig_cb = 1.0 / (1.0 + np.exp(-beta * cb_viol))
    sig_cb[~np.isfinite(sig_cb)] = 0.0
    
    inv_dist_origin = np.zeros_like(dist_origin)
    np.divide(1.0, dist_origin, out=inv_dist_origin, where=dist_origin > 1e-12)
    
    grad += (sig_cb * inv_dist_origin)[:, np.newaxis] * P
    
    return grad


def _get_ring_initialization(params, ring_counts):
    """
    Generate points based on concentric rings.
    params: [R1, R2, ..., Rk, offset1, offset2, ..., offsetk]
    ring_counts: list of number of points in each ring.
    """
    P = np.zeros((N, 2))
    k = len(ring_counts)
    
    # Radii are the first k params
    radii = params[:k]
    # Offsets are the next k params
    offsets = params[k:2*k]
    
    count = 0
    for i, n_pts in enumerate(ring_counts):
        if n_pts == 1:
            # Center point
            P[count] = [0.0, 0.0]
            count += 1
        else:
            R = max(0.0, radii[i])
            offset = offsets[i]
            angles = np.linspace(0, 2 * np.pi, n_pts, endpoint=False) + offset
            P[count:count+n_pts, 0] = R * np.cos(angles)
            P[count:count+n_pts, 1] = R * np.sin(angles)
            count += 1 if n_pts == 1 else n_pts
            
    return P


def _optimize_positions_soft(P_init, r_target, time_limit=5.0, beta=50.0):
    """
    Optimize positions P using soft-min energy for a fixed r_target.
    Returns (P_opt, energy).
    """
    start_time = time.time()
    
    def objective(P_flat):
        P = P_flat.reshape(N, 2)
        return _softmin_energy(P, r_target, beta)
    
    def grad(P_flat):
        P = P_flat.reshape(N, 2)
        return _softmin_grad(P, r_target, beta).flatten()
        
    try:
        res = minimize(objective, P_init.flatten(), jac=grad, method='L-BFGS-B', 
                       options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12})
        P_opt = res.x.reshape(N, 2)
        min_energy = objective(P_opt)
        return P_opt, min_energy
    except Exception:
        return P_init, objective(P_init.flatten())


def _optimize_positions(P_init, r_target, time_limit=5.0):
    """
    Optimize positions P to minimize soft-min energy for a fixed r_target.
    Returns (P_opt, min_energy).
    """
    return _optimize_positions_soft(P_init, r_target, time_limit, beta=50.0)


def _get_hex_initial_points():
    """Fallback initialization using hexagonal lattice."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        margin = spacing / 2.0 
        points, radius = points[radius <= 1.0 - margin], radius[radius <= 1.0 - margin]
        if len(points) >= N:
            indices = np.argsort(radius, kind="stable")
            return points[indices[:N]]
        spacing *= 0.98


def _optimize_ring_structure(radii0, counts, angles_offset0=None, time_limit=30.0):
    """Optimize ring radii and angles for fixed counts using differential evolution."""
    if angles_offset0 is None:
        angles_offset0 = [0.0] * len(counts)
    
    n_rings = len(counts)
    
    def build_and_score(params):
        radii = np.maximum(params[:n_rings], 0.0)
        angles = params[n_rings:]
        P = _get_ring_initialization(np.concatenate([radii, angles]), counts)
        return -_compute_score(P)  # minimize negative score
    
    # Bounds for differential evolution
    bounds = []
    for i in range(n_rings):
        if counts[i] == 1:
            bounds.append((0.0, 0.0))  # center point, radius fixed at 0
        else:
            bounds.append((0.0, 0.9))  # ring radius
    for i in range(n_rings):
        bounds.append((0.0, 2.0 * np.pi))  # angle offset
    
    t_start = time.time()
    try:
        res = differential_evolution(
            build_and_score,
            bounds,
            seed=42,
            maxiter=100,
            popsize=15,
            tol=1e-8,
            callback=lambda x, convergence: None if time.time() - t_start < time_limit else (_ for _ in ()).throw(Exception("timeout"))
        )
        if res.success or res.fun < build_and_score(np.concatenate([radii0, angles_offset0])):
            radii_opt = np.maximum(res.x[:n_rings], 0.0)
            angles_opt = res.x[n_rings:]
            return radii_opt, angles_opt
    except Exception:
        pass
    
    return radii0, angles_offset0


def _polish_l_bfgs(P, r, time_limit=10.0):
    """High-precision polish pass using L-BFGS-B with tight tolerances."""
    t_start = time.time()
    
    def objective(P_flat):
        P = P_flat.reshape(N, 2)
        return _softmin_energy(P, r, beta=100.0)
    
    def grad(P_flat):
        P = P_flat.reshape(N, 2)
        return _softmin_grad(P, r, beta=100.0).flatten()
    
    try:
        res = minimize(objective, P.flatten(), jac=grad, method='L-BFGS-B',
                       options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-13})
        P_opt = res.x.reshape(N, 2)
        return P_opt
    except Exception:
        return P


def _anneal_points(pts, deadline):
    """Multi-stage annealing optimization on full point coordinates."""
    current = pts.copy()
    current_score = _compute_score(current)
    
    # Progressive beta values for annealing - more stages for better convergence
    betas = [3.0, 8.0, 20.0, 50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0, 6400.0]
    
    for beta in betas:
        if time.time() > deadline - 4:
            break
        try:
            def objective(P_flat, b=beta):
                P = P_flat.reshape(N, 2)
                return _softmin_energy(P, current_score * 0.999, b)
            
            def grad(P_flat, b=beta):
                P = P_flat.reshape(N, 2)
                return _softmin_grad(P, current_score * 0.999, b).flatten()
            
            res = minimize(
                objective,
                current.ravel(),
                jac=grad,
                method='L-BFGS-B',
                options={
                    'maxiter': 1500, 
                    'ftol': 1e-16, 
                    'gtol': 1e-14,
                    'maxfun': 30000
                }
            )
            new_pts = res.x.reshape(N, 2)
            if not np.all(np.isfinite(new_pts)):
                continue
            s = _compute_score(new_pts)
            if s > current_score:
                current = new_pts
                current_score = s
        except Exception:
            pass
            
    return current, current_score


def _constraint_push(pts, r_target, deadline, max_iters=400):
    """Polish solution by iteratively pushing violating constraints with relaxation."""
    pts = pts.copy()
    relaxation = 1.0  # Start with full push, decrease if oscillating
    prev_score = _compute_score(pts)
    
    for _ in range(max_iters):
        if time.time() > deadline - 3:
            break
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(d, np.inf)
        pair_margins = d / 2.0
        boundary_margins = 1.0 - np.sqrt(np.sum(pts ** 2, axis=1))
        
        min_pair = np.min(pair_margins)
        min_bound = np.min(boundary_margins)
        
        if min_pair < min_bound - 1e-14:
            # Pair constraint is binding
            i, j = np.unravel_index(np.argmin(pair_margins), pair_margins.shape)
            if i == j:
                break
            di = pts[i] - pts[j]
            dist = np.linalg.norm(di)
            if dist < 1e-15:
                break
            di /= dist
            push = (2.0 * r_target - dist) / 2.0 * relaxation
            if push > 1e-15:
                pts[i] += push * di
                pts[j] -= push * di
        else:
            # Boundary constraint is binding
            i = int(np.argmin(boundary_margins))
            pi = pts[i]
            dist = np.linalg.norm(pi)
            if dist < 1e-15:
                pts[i] += 1e-4 * np.array([1.0, 0.0])
            else:
                di = pi / dist
                push = (1.0 - r_target - dist) * relaxation
                if push > 1e-15:
                    pts[i] += push * di
        
        # Check convergence and adjust relaxation
        current_score = _compute_score(pts)
        if current_score < prev_score - 1e-10:
            # Score decreased, reduce relaxation to avoid oscillation
            relaxation *= 0.7
        elif current_score > prev_score + 1e-12:
            # Score improved, maintain or slightly increase relaxation
            relaxation = min(relaxation * 1.1, 1.0)
        prev_score = current_score
        
        # Check if target achieved
        if current_score >= r_target - 1e-12:
            break
    
    return pts


def construct():
    t_start = time.time()
    
    # Define ring structures for N=50
    ring_structures = [
        [1, 6, 12, 18, 13],
        [1, 6, 12, 14, 17],
        [1, 6, 12, 16, 15],
        [1, 6, 12, 15, 16],
        [1, 6, 10, 14, 19],
        [1, 6, 10, 12, 21],
        [1, 6, 12, 17, 14],
        [1, 6, 13, 13, 17],
    ]
    
    # Start with hexagonal initialization
    P_best = _get_hex_initial_points()
    score_best = _compute_score(P_best)
    
    # For each ring structure, optimize and refine
    for counts in ring_structures:
        elapsed = time.time() - t_start
        if elapsed > DEADLINE - 30:
            break
        
        n_rings = len(counts)
        
        # Initial radii: evenly spaced
        if n_rings == 1:
            radii0 = np.array([0.0])
        else:
            radii0 = np.linspace(0, 0.85, n_rings)
            radii0[0] = 0.0
        
        angles0 = [0.0] * n_rings
        
        # Optimize ring structure parameters using DE
        remaining = DEADLINE - (time.time() - t_start) - 35
        if remaining < 5.0:
            continue
        radii_opt, angles_opt = _optimize_ring_structure(
            radii0, counts, angles0,
            time_limit=min(15.0, remaining)
        )
        
        # Build centers from optimized ring structure
        P_candidate = _get_ring_initialization(
            np.concatenate([radii_opt, angles_opt]), counts
        )
        
        score_candidate = _compute_score(P_candidate)
        
        if score_candidate > score_best:
            P_best = P_candidate.copy()
            score_best = score_candidate
        
        # Binary search on r with soft-min optimization
        r_low = score_best * 0.98
        r_high = min(0.135, score_best * 1.05)
        
        # Ensure r_low is feasible
        remaining = DEADLINE - (time.time() - t_start) - 20
        if remaining < 3.0:
            continue
        P_opt, E = _optimize_positions(P_candidate, r_low, time_limit=min(4.0, remaining * 0.3))
        if E < 1e-8:
            score = _compute_score(P_opt)
            if score > score_best:
                P_best = P_opt.copy()
                score_best = score
            r_low = score
        
        # Binary search for better r
        for _ in range(10):
            remaining = DEADLINE - (time.time() - t_start) - 15
            if remaining < 3.0:
                break
            
            r_mid = (r_low + r_high) / 2.0
            P_init = P_best.copy()
            
            P_opt, E = _optimize_positions(P_init, r_mid, time_limit=min(3.5, remaining * 0.25))
            
            if E < 1e-8:
                score = _compute_score(P_opt)
                if score > score_best:
                    P_best = P_opt.copy()
                    score_best = score
                r_low = score
            else:
                r_high = r_mid
        
        # Final high-precision polish with L-BFGS-B
        remaining = DEADLINE - (time.time() - t_start) - 5
        if remaining > 5.0:
            r_current = _compute_score(P_best)
            P_polished = _polish_l_bfgs(P_best, r_current * 0.999, time_limit=remaining)
            score_polished = _compute_score(P_polished)
            if score_polished > score_best:
                P_best = P_polished.copy()
                score_best = score_polished
    
    # Multi-stage annealing polish
    remaining = DEADLINE - (time.time() - t_start)
    if remaining > 10.0:
        P_annealed, score_annealed = _anneal_points(P_best, DEADLINE - 5)
        if score_annealed > score_best:
            P_best = P_annealed
            score_best = score_annealed
    
    # Constraint push polish
    remaining = DEADLINE - (time.time() - t_start)
    if remaining > 5.0:
        r_target = score_best * 1.001
        P_pushed = _constraint_push(P_best, r_target, DEADLINE - 2)
        score_pushed = _compute_score(P_pushed)
        if score_pushed > score_best:
            P_best = P_pushed
            score_best = score_pushed
    
    # Ensure validity: clamp points that might be slightly outside
    r_final = _compute_score(P_best)
    if r_final > 0:
        r2 = np.hypot(P_best[:, 0], P_best[:, 1])
        for i in range(N):
            if r2[i] > 1 - r_final + 1e-10:
                scale = (1 - r_final) / r2[i]
                P_best[i] *= scale
    
    return P_best.reshape(N, 2)
