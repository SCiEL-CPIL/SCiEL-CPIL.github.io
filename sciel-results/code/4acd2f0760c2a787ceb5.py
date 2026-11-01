"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
_DEADLINE = time.time() + 140.0  # 10s safety margin before 150s kill


def _score(pts):
    """Compute the valid packing radius (or 0 if invalid)."""
    if len(pts) != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    dists = np.hypot(pts[:, 0], pts[:, 1])
    if np.any(dists > 1.0):
        return 0.0
    r_cont = 1.0 - np.max(dists)
    if r_cont < 0:
        return 0.0
    if N < 2:
        return r_cont
    # Efficient pairwise distance computation
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:1, 1]
    dist_mat = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist_mat, np.inf)
    r_pair = 0.5 * np.min(dist_mat)
    return min(r_cont, r_pair)


def _soft_min_obj_and_grad(x, beta=100.0):
    """
    Compute soft-min objective and gradient for maximizing the minimum distance.
    Objective: -soft_min(d_ij/2, 1-r_i) where soft_min uses log-sum-exp.
    We want to maximize min(d_ij/2, 1-r_i), so we minimize its negative.
    """
    pts = x.reshape(-1, 2)
    
    # Compute all pairwise distances
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:1, 1]
    dist_mat = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist_mat, np.inf)
    
    # Compute container distances
    r_cont = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
    
    # Collect all constraints: pairwise distances/2 and container distances
    # For soft min, we use: soft_min(a) = -1/beta * log(sum(exp(-beta*a)))
    # We want to maximize min(constraints), so minimize -soft_min(constraints)
    
    # Pairwise constraints: d_ij / 2
    pair_vals = dist_mat[dist_mat < np.inf] / 2.0
    
    # Container constraints
    cont_vals = r_cont
    
    # All constraints
    all_vals = np.concatenate([pair_vals, cont_vals])
    
    # Soft min via log-sum-exp (numerically stable)
    m = np.min(all_vals)
    log_sum_exp = m + np.log(np.sum(np.exp(-beta * (all_vals - m))))
    soft_min_val = -log_sum_exp / beta
    
    # Objective: minimize -soft_min_val
    obj = -soft_min_val
    
    # Gradient
    # d(soft_min)/dx = (1/beta) * sum(exp(-beta*(v_i - m))) * (-beta) * dv_i/dx / sum(exp(-beta*(v_i - m)))
    # = -sum(w_i * dv_i/dx) where w_i = exp(-beta*(v_i-m)) / sum(exp(-beta*(v_i-m)))
    
    exp_vals = np.exp(-beta * (all_vals - m))
    weights = exp_vals / np.sum(exp_vals)
    
    # Gradient with respect to positions
    grad = np.zeros_like(x)
    
    # Pairwise part
    w_pair = weights[:len(pair_vals)]
    w_cont = weights[len(pair_vals):]
    
    # Reconstruct which pairs correspond to which weights
    # This is complex, so use a simpler approach: compute gradient numerically or use autograd-like approach
    # For efficiency, let's compute the gradient properly
    
    # Actually, let's use a different approach: compute the weighted sum of gradients
    # For pair (i,j) with value d_ij/2, gradient wrt pts[i] is (pts[i]-pts[j])/d_ij * w_ij
    # For container i with value 1-r_i, gradient wrt pts[i] is -pts[i]/r_i * w_cont_i
    
    # Build pair index mapping
    idx_i, idx_j = np.where(np.triu(np.ones((N, N), dtype=bool), k=1))
    pair_dists = dist_mat[idx_i, idx_j]
    
    # Map weights back to pairs
    # The order in pair_vals matches the order in idx_i, idx_j (row-major of upper triangle)
    for k in range(len(idx_i)):
        i, j = idx_i[k], idx_j[k]
        w = w_pair[k]
        d = pair_dists[k]
        if d > 1e-10:
            dx_vec = pts[i] - pts[j]
            grad[2*i:2*i+2] += w * dx_vec / d * 0.5  # d(d_ij/2)/d(pts[i]) = 0.5 * (pts[i]-pts[j])/d_ij
            grad[2*j:2*j+2] -= w * dx_vec / d * 0.5
    
    # Container part
    for i in range(N):
        w = w_cont[i]
        r = np.hypot(pts[i, 0], pts[i, 1])
        if r > 1e-10:
            # d(1-r)/d(pts[i]) = -pts[i]/r
            grad[2*i:2*i+2] += w * (-pts[i] / r)
    
    # Since we minimize -soft_min, gradient of objective is -gradient of soft_min
    # soft_min = -log_sum_exp / beta
    # d(soft_min)/dx = (1/beta) * sum(w_i * dv_i/dx) where w_i are normalized exp weights
    # Actually: soft_min = -1/beta * log(sum(exp(-beta*v)))
    # d(soft_min)/dx = -1/beta * (1/sum(exp(-beta*v))) * sum(exp(-beta*v) * (-beta) * dv/dx)
    # = sum(w_i * dv_i/dx) where w_i = exp(-beta*v_i)/sum(exp(-beta*v))
    # So d(obj)/dx = -d(soft_min)/dx = -sum(w_i * dv_i/dx)
    
    grad = -grad  # Because obj = -soft_min
    
    return obj, grad


def _hex_lattice_init():
    """Generate a hexagonal lattice packing as initial guess."""
    best_pts = None
    best_score = 0.0
    for spacing in np.linspace(0.15, 0.35, 80):
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points = points[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            r = np.hypot(points[:, 0], points[:, 1])
            idx = np.argsort(r, kind="stable")[:N]
            pts = points[idx]
            s = _score(pts)
            if s > best_score:
                best_score = s
                best_pts = pts
    if best_pts is not None:
        return best_pts
    # Fallback: random points in unit disk
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 0.81, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _ring_init():
    """Generate a ring-based initial configuration."""
    pts = np.zeros((N, 2))
    # Place some at center, rest in rings
    center_count = min(7, N)
    # Hexagonal arrangement for center
    center_pts = np.array([[0, 0]])
    for ring in range(1, 4):
        count = 6 * ring
        if len(center_pts) + count > center_count:
            count = center_count - len(center_pts)
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
        r = ring * 0.25
        center_pts = np.vstack([center_pts, r * np.column_stack([np.cos(angles), np.sin(angles)])])
    
    pts[:center_count] = center_pts[:center_count]
    
    # Rest in outer ring
    remaining = N - center_count
    if remaining > 0:
        angles = np.linspace(0, 2 * np.pi, remaining, endpoint=False)
        r = 0.7
        pts[center_count:] = r * np.column_stack([np.cos(angles), np.sin(angles)])
    
    return pts


def _soft_min_optimize(pts0, beta=100.0, max_time=30.0):
    """Optimize using soft-min objective with L-BFGS-B."""
    x0 = pts0.flatten()
    
    # Define objective and gradient
    def obj_and_grad(x):
        return _soft_min_obj_and_grad(x, beta)
    
    start_time = time.time()
    
    try:
        result = minimize(
            obj_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 5000}
        )
        
        # Project back to disk
        pts = result.x.reshape(-1, 2)
        radii = np.hypot(pts[:, 0], pts[:, 1])
        mask = radii > 1.0
        if np.any(mask):
            pts[mask] = pts[mask] / radii[mask, None]
        
        return pts
    
    except Exception:
        return pts0


def _full_optimize(pts0, budget=120.0):
    """Full optimization with multiple restarts and increasing beta."""
    best_pts = pts0.copy()
    best_score = _score(pts0)
    start_time = time.time()
    
    # Try different betas (higher beta = sharper min approximation)
    betas = [50.0, 100.0, 200.0, 400.0]
    
    for beta in betas:
        if time.time() - start_time > budget - 10:
            break
        
        # Optimize from current best
        opt_pts = _soft_min_optimize(best_pts, beta=beta, max_time=budget - (time.time() - start_time) - 5)
        opt_score = _score(opt_pts)
        
        if opt_score > best_score:
            best_score = opt_score
            best_pts = opt_pts.copy()
        
        # Also try from hex lattice init
        hex_pts = _hex_lattice_init()
        opt_pts2 = _soft_min_optimize(hex_pts, beta=beta, max_time=budget - (time.time() - start_time) - 5)
        opt_score2 = _score(opt_pts2)
        
        if opt_score2 > best_score:
            best_score = opt_score2
            best_pts = opt_pts2.copy()
        
        # Try ring init
        ring_pts = _ring_init()
        opt_pts3 = _soft_min_optimize(ring_pts, beta=beta, max_time=budget - (time.time() - start_time) - 5)
        opt_score3 = _score(opt_pts3)
        
        if opt_score3 > best_score:
            best_score = opt_score3
            best_pts = opt_pts3.copy()
    
    # Final polish with high beta
    if time.time() - start_time < budget - 5:
        final_pts = _soft_min_optimize(best_pts, beta=500.0, max_time=budget - (time.time() - start_time))
        final_score = _score(final_pts)
        if final_score > best_score:
            best_score = final_score
            best_pts = final_pts.copy()
    
    return best_pts


def construct():
    start_time = time.time()
    
    # Get initial hexagonal lattice configuration
    pts0 = _hex_lattice_init()
    best_pts = pts0
    best_score = _score(pts0)
    
    # Run full optimization
    try:
        if time.time() < _DEADLINE:
            opt_pts = _full_optimize(pts0, budget=130.0)
            opt_score = _score(opt_pts)
            if opt_score > best_score:
                best_pts = opt_pts
                best_score = opt_score
    except Exception:
        pass
    
    # Final safety check: ensure validity
    final_score = _score(best_pts)
    if final_score <= 0:
        # Return the initial hex lattice as fallback
        return pts0
    
    return best_pts

# EVOLVE-BLOCK-END
