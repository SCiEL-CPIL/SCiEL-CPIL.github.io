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
    
    dists_origin = np.hypot(pts[:, 0], pts[:, 1])
    if np.any(dists_origin > 1.0):
        return 0.0
    r_cont = 1.0 - np.max(dists_origin)
    if r_cont < 0:
        return 0.0
        
    if N < 2:
        return r_cont
        
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    r_pair = 0.5 * np.min(dists)
    
    return min(r_cont, r_pair)


def _ring_init():
    """
    Generate ring-based initialization matching known optimal structure:
    1 (center) + 6 + 12 + 18 + 13 = 50
    """
    rings = [
        (0, 1),    # center
        (1, 6),    # 6 circles
        (2, 12),   # 12 circles
        (3, 18),   # 18 circles
        (4, 13),   # 13 circles (outermost, partial ring)
    ]
    
    # Target radii for each ring (tuned for r ~ 0.1258)
    ring_radii = [0.0, 0.25, 0.50, 0.75, 0.95]
    
    pts = []
    for ring_idx, (r_frac, count) in enumerate(rings):
        radius = ring_radii[ring_idx]
        if count == 1:
            pts.append([0.0, 0.0])
        else:
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
            # Add slight angular offset per ring to avoid alignment
            offset = ring_idx * 0.1
            for a in angles:
                a += offset
                pts.append([radius * np.cos(a), radius * np.sin(a)])
    
    pts = np.array(pts[:N])
    return pts


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
    angles = np.random.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(np.random.uniform(0, 0.81, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _random_init(seed=0):
    """Random initialization in unit disk."""
    rng = np.random.RandomState(seed)
    angles = rng.uniform(0, 2 * np.pi, N)
    radii = np.sqrt(rng.uniform(0, 0.9, N))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _soft_min_obj_and_grad(x, beta):
    """
    Soft-min objective with gradient for L-BFGS-B.
    Minimizes -soft_min(min_pair_dist/2, 1 - max_dist_to_origin)
    """
    pts = x.reshape(N, 2)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, np.inf)
    
    finite_mask = np.isfinite(dists)
    finite_dists = dists[finite_mask]
    
    if len(finite_dists) == 0:
        return 0.0, np.zeros_like(x)
    
    # soft_min of pairwise distances
    val = -finite_dists / beta
    max_val = np.max(val)
    log_sum_exp = max_val + np.log(np.sum(np.exp(val - max_val)))
    soft_min_pair = -beta * log_sum_exp
    
    # Container distance
    dists_origin = np.hypot(pts[:, 0], pts[:, 1])
    r_cont = 1.0 - np.max(dists_origin)
    
    # Combine: we want to maximize min(r_pair, r_cont)
    # Use soft min of the two
    r_pair = 0.5 * soft_min_pair
    combined = min(r_pair, r_cont)
    
    # For gradient, compute gradient of -r_pair and -r_cont
    # Gradient of soft_min: d(soft_min)/d(d_ij) = exp(-d_ij/beta) / sum(exp(-d_kj/beta))
    # But we need gradient w.r.t. positions
    
    # Compute gradient of soft_min_pair w.r.t. pts
    # soft_min = -beta * log(sum_{i<j} exp(-d_ij/beta))
    # d(soft_min)/d(pts_i) = beta * sum_{j != i} exp(-d_ij/beta) * (pts_i - pts_j) / (d_ij * sum_{k<l} exp(-d_kl/beta))
    
    # Build weight matrix
    w = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j and finite_mask[i, j]:
                w[i, j] = np.exp(-dists[i, j] / beta)
    
    w_sum = np.sum(w)
    if w_sum > 0:
        # grad_soft_min[i] = beta * sum_j w[i,j] * (pts[i] - pts[j]) / d_ij / w_sum
        grad_soft = np.zeros((N, 2))
        for i in range(N):
            for j in range(N):
                if i != j and w[i, j] > 0:
                    d = dists[i, j]
                    if d > 1e-10:
                        grad_soft[i] += beta * w[i, j] * (pts[i] - pts[j]) / (d * w_sum)
        
        # Objective is -min(r_pair, r_cont), gradient of -r_pair = -0.5 * grad_soft_min
        grad_obj = -0.5 * grad_soft
    else:
        grad_obj = np.zeros((N, 2))
    
    # Gradient of -r_cont = -(1 - max_dist) = max_dist - 1
    # d(max_dist)/d(pts_i) = pts_i / |pts_i| for the point with max distance
    idx_max = np.argmax(dists_origin)
    if dists_origin[idx_max] > 1e-10:
        grad_obj[idx_max] -= pts[idx_max] / dists_origin[idx_max]
    
    # Boundary penalty gradient
    boundary_pen = 0.0
    for i in range(N):
        if dists_origin[i] > 1.0:
            boundary_pen += 10.0 * 2.0 * (dists_origin[i] - 1.0) * pts[i] / dists_origin[i]
            grad_obj[i] += 20.0 * (dists_origin[i] - 1.0) * pts[i] / dists_origin[i]
    
    return -combined + boundary_pen, grad_obj.flatten()


def _force_directed(pts0, time_limit, iters=2000):
    """
    Force-directed layout: repulsive forces between circles,
    attractive force toward center for boundary constraint.
    """
    pts = pts0.copy()
    best_score = _score(pts)
    best_pts = pts.copy()
    
    # Adaptive learning rate
    lr = 0.01
    decay = 0.999
    
    for it in range(iters):
        if time.time() > time_limit:
            break
        
        # Compute forces
        forces = np.zeros_like(pts)
        
        # Pairwise repulsion: F = 1/d^4 for d < 2*r_est
        # Estimate current radius
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dists, np.inf)
        
        r_est = 0.5 * np.min(dists[dists < np.inf])
        if r_est < 1e-6:
            r_est = 0.05
        
        # Repulsion between all pairs
        for i in range(N):
            for j in range(i + 1, N):
                d = dists[i, j]
                if d < 1e-10:
                    # Random nudge
                    forces[i] += 0.01 * np.random.randn(2)
                    forces[j] -= 0.01 * np.random.randn(2)
                    continue
                if d < 4.0 * r_est:
                    # Strong repulsion when close
                    f_mag = 1.0 / (d ** 4)
                    direction = (pts[i] - pts[j]) / d
                    forces[i] += f_mag * direction
                    forces[j] -= f_mag * direction
        
        # Boundary force: push inward if too close to edge
        dists_origin = np.hypot(pts[:, 0], pts[:, 1])
        margin = 1.0 - dists_origin
        for i in range(N):
            if margin[i] < r_est:
                # Push toward center
                if dists_origin[i] > 1e-10:
                    forces[i] -= 2.0 * (r_est - margin[i]) * pts[i] / dists_origin[i]
        
        # Centering force (weak) to keep packing compact
        forces -= 0.001 * pts
        
        # Update positions
        pts += lr * forces
        
        # Project to unit disk
        radii = np.hypot(pts[:, 0], pts[:, 1])
        mask = radii > 1.0
        if np.any(mask):
            pts[mask] = pts[mask] / radii[mask, np.newaxis]
        
        # Clamp to avoid NaN
        pts = np.nan_to_num(pts, nan=0.0, posinf=0.9, neginf=-0.9)
        
        # Track best
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
        
        # Decay learning rate
        lr *= decay
        
        # Early stopping if no improvement for a while
        if it > 500 and s <= best_score * 0.999:
            continue
        
    return best_pts


def _polish(pts0, time_limit):
    """
    Progressive polish using L-BFGS-B with decreasing beta.
    """
    pts = pts0.copy()
    best_score = _score(pts)
    best_pts = pts.copy()
    
    # Progressive betas: start coarse, end fine
    betas = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    for beta in betas:
        if time.time() > time_limit:
            break
        
        def obj(x):
            val, _ = _soft_min_obj_and_grad(x, beta)
            return val
        
        def grad(x):
            _, g = _soft_min_obj_and_grad(x, beta)
            return g
        
        try:
            res = minimize(
                obj,
                grad,
                pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            new_pts = res.x.reshape(N, 2)
            
            # Project back to unit disk
            radii = np.hypot(new_pts[:, 0], new_pts[:, 1])
            mask = radii > 1.0
            if np.any(mask):
                new_pts[mask] = new_pts[mask] / radii[mask, np.newaxis]
            
            new_score = _score(new_pts)
            
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
                pts = new_pts.copy()
        except Exception:
            pass
    
    return best_pts


def _final_polish(pts0, time_limit):
    """
    High-precision final polish with very tight tolerances.
    """
    pts = pts0.copy()
    best_score = _score(pts)
    best_pts = pts.copy()
    
    # Very fine beta for high precision
    betas = [0.001, 0.0005, 0.0002]
    
    for beta in betas:
        if time.time() > time_limit:
            break
        
        def obj(x):
            val, _ = _soft_min_obj_and_grad(x, beta)
            return val
        
        def grad(x):
            _, g = _soft_min_obj_and_grad(x, beta)
            return g
        
        try:
            res = minimize(
                obj,
                grad,
                pts.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 100}
            )
            new_pts = res.x.reshape(N, 2)
            
            radii = np.hypot(new_pts[:, 0], new_pts[:, 1])
            mask = radii > 1.0
            if np.any(mask):
                new_pts[mask] = new_pts[mask] / radii[mask, np.newaxis]
            
            new_score = _score(new_pts)
            
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts.copy()
                pts = new_pts.copy()
        except Exception:
            pass
    
    return best_pts


def construct():
    start_time = time.time()
    
    # 1. Multiple initializations
    initializations = [
        _ring_init(),
        _hex_lattice_init(),
        _random_init(42),
        _random_init(123),
    ]
    
    # Score all initializations
    init_scores = [_score(p) for p in initializations]
    best_init_idx = np.argmax(init_scores)
    best_pts = initializations[best_init_idx].copy()
    best_score = init_scores[best_init_idx]
    
    # 2. Force-directed layout from best init
    fd_deadline = _DEADLINE - 60.0
    if time.time() < fd_deadline:
        try:
            fd_pts = _force_directed(best_pts, fd_deadline, iters=3000)
            fd_score = _score(fd_pts)
            if fd_score > best_score:
                best_pts = fd_pts
                best_score = fd_score
        except Exception:
            pass
    
    # 3. Also try force-directed from other good inits if time permits
    if time.time() < _DEADLINE - 80.0:
        for i, init in enumerate(initializations):
            if i == best_init_idx:
                continue
            if init_scores[i] < best_score * 0.9:
                continue
            if time.time() > _DEADLINE - 70.0:
                break
            try:
                fd_pts = _force_directed(init, _DEADLINE - 70.0, iters=1000)
                fd_score = _score(fd_pts)
                if fd_score > best_score:
                    best_pts = fd_pts
                    best_score = fd_score
            except Exception:
                pass
    
    # 4. Progressive polish
    polish_deadline = _DEADLINE - 20.0
    if time.time() < polish_deadline:
        try:
            polished_pts = _polish(best_pts, polish_deadline)
            polished_score = _score(polished_pts)
            if polished_score > best_score:
                best_pts = polished_pts
                best_score = polished_score
        except Exception:
            pass
    
    # 5. Final high-precision polish
    final_deadline = _DEADLINE - 5.0
    if time.time() < final_deadline:
        try:
            final_pts = _final_polish(best_pts, final_deadline)
            final_score = _score(final_pts)
            if final_score > best_score:
                best_pts = final_pts
                best_score = final_score
        except Exception:
            pass
    
    # Final safety check
    final_score = _score(best_pts)
    if final_score <= 0:
        return initializations[best_init_idx]
    
    return best_pts

# EVOLVE-BLOCK-END
