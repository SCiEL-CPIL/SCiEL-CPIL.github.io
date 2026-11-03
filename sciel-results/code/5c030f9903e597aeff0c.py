"""Packing 300 equal circles in a unit circle using Multi-Start Local Search."""
import numpy as np
import time
from scipy.optimize import minimize

N = 300
TIME_LIMIT = 140.0  # safety margin before 150s kill

def _hex_init(n=N, spacing=0.095, jitter=0.0):
    """Dense hexagonal lattice clipped to the unit disk, keep n closest to center."""
    k = int(1.15 / spacing) + 2
    xs, ys = [], []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            xs.append(x)
            ys.append(y)
    pts = np.column_stack([xs, ys])
    r = np.hypot(pts[:, 0], pts[:, 1])
    inside = r <= 1.0
    pts = pts[inside]
    r = r[inside]
    if len(pts) < n:
        # Fallback to tighter spacing if not enough points
        return _hex_init(n, spacing=0.08, jitter=jitter)
    
    order = np.argsort(r, kind="stable")
    pts = pts[order[:n]].copy()
    
    if jitter > 0:
        pts += np.random.normal(0, jitter, pts.shape)
        # Ensure they stay inside roughly
        rad = np.hypot(pts[:, 0], pts[:, 1])
        out = rad > 0.95
        if np.any(out):
            pts[out] = pts[out] * (0.95 / rad[out, None])
            
    return pts

def _score(pts):
    """Achieved common radius (min over pairs and boundary)."""
    n = len(pts)
    # Efficient distance calculation for score
    d2 = np.sum(pts * pts, axis=1)[:, None] - 2.0 * (pts @ pts.T) + np.sum(pts * pts, axis=1)[None, :]
    np.fill_diagonal(d2, np.inf)
    min_pair = np.sqrt(np.min(d2)) / 2.0
    min_bound = 1.0 - np.max(np.hypot(pts[:, 0], pts[:, 1]))
    return min(min_pair, min_bound)

def _fast_local_optimize(pts, time_budget=10.0):
    """
    Fast gradient-based push-apart optimization.
    Uses a simple iterative scheme to maximize the minimum distance.
    This is effectively a projected gradient ascent on the min-distance,
    approximated by a soft-min with high beta, but using simple steps.
    """
    start_time = time.time()
    pts = pts.copy()
    n = len(pts)
    
    # We will iterate: compute forces that push apart close pairs and pull boundary points in.
    # Force for pair (i,j): if d < 2r_target, push apart.
    # Since r_target is unknown, we use a generic repulsion that strengthens as they get closer.
    # Let's use a potential: V = sum 1/d^2 for d < 0.2, and boundary penalty.
    # Actually, simpler: Gradient of min distance.
    # Approximation: Use a very sharp soft-min (beta=1000) but with simple gradient steps.
    
    beta = 500.0
    step_size = 0.01
    
    # Precompute indices
    idx_i, idx_j = np.triu_indices(n, k=1)
    
    for step in range(1000):
        if time.time() - start_time > time_budget:
            break
            
        # Compute distances
        d2 = np.sum(pts * pts, axis=1)[:, None] - 2.0 * (pts @ pts.T) + np.sum(pts * pts, axis=1)[None, :]
        np.fill_diagonal(d2, np.inf)
        d2 = np.maximum(d2, 1e-12)
        
        # Pair radii (half distances)
        d = np.sqrt(d2)
        pair_r = d / 2.0
        pair_r_flat = pair_r[idx_i, idx_j]
        
        # Boundary margins
        rad = np.hypot(pts[:, 0], pts[:, 1])
        bound_r = 1.0 - rad
        
        all_r = np.concatenate([pair_r_flat, bound_r])
        
        # Soft min weights
        x_min = np.min(all_r)
        x_shifted = all_r - x_min
        x_shifted = np.clip(x_shifted, 0, 50.0 / beta)
        exp_terms = np.exp(-beta * x_shifted)
        sum_exp = np.sum(exp_terms)
        weights = exp_terms / sum_exp
        
        # Gradient of soft min w.r.t. positions
        grad = np.zeros_like(pts)
        
        # Pair contributions
        # d (d_ij/2) / d pts_i = (pts_i - pts_j) / (2 * d_ij)
        # We want to MAXIMIZE the min distance, so we move in direction of gradient.
        # grad_r = sum w_k * d r_k / d x
        for k in range(len(idx_i)):
            i = idx_i[k]
            j = idx_j[k]
            w = weights[k]
            if w < 1e-10: continue
            d_val = d[i, j]
            if d_val < 1e-10: continue
            diff = (pts[i] - pts[j]) / (2.0 * d_val)
            grad[2*i] += w * diff[0]
            grad[2*i+1] += w * diff[1]
            grad[2*j] += w * (-diff[0])
            grad[2*j+1] += w * (-diff[1])
            
        # Boundary contributions
        # d (1 - |c_i|) / d pts_i = -pts_i / |c_i|
        # We want to maximize this, so if boundary is tight, we move INWARD (negative gradient of distance to boundary? No.)
        # bound_r = 1 - r. d bound_r / d x = -x/r.
        # If bound_r is small (tight), weight is high. We want to increase bound_r.
        # So we move in direction of -grad(bound_r) = x/r. i.e. outward? No, that decreases bound_r.
        # Wait. r = 1 - |c|. To maximize r, we need to minimize |c|.
        # Gradient of |c| is c/|c|. Gradient of 1-|c| is -c/|c|.
        # We want to maximize the function, so we move in direction of gradient: -c/|c|.
        # This means moving towards the origin. Correct.
        
        pair_count = len(idx_i)
        for i in range(n):
            w = weights[pair_count + i]
            if w < 1e-10: continue
            r_i = rad[i]
            if r_i < 1e-10: continue
            # Move towards origin
            grad[2*i] += w * (-pts[i, 0] / r_i)
            grad[2*i+1] += w * (-pts[i, 1] / r_i)
            
        # Update positions
        pts += step_size * grad
        
        # Project back to unit disk (soft constraint)
        rad_new = np.hypot(pts[:, 0], pts[:, 1])
        out = rad_new > 1.0
        if np.any(out):
            pts[out] = pts[out] / rad_new[out, None]
            
    return pts

def construct():
    start_time = time.time()
    best_pts = None
    best_score = -1.0
    
    # Strategy: Multi-start with diverse hexagonal lattices
    # We try different spacings and jitters to escape local minima
    
    # 1. Standard hex
    pts1 = _hex_init(N, spacing=0.095, jitter=0.0)
    # 2. Tighter hex
    pts2 = _hex_init(N, spacing=0.090, jitter=0.0)
    # 3. Looser hex
    pts3 = _hex_init(N, spacing=0.100, jitter=0.0)
    # 4. Perturbed standard hex
    pts4 = _hex_init(N, spacing=0.095, jitter=0.01)
    # 5. Perturbed tighter hex
    pts5 = _hex_init(N, spacing=0.090, jitter=0.02)
    
    candidates = [pts1, pts2, pts3, pts4, pts5]
    
    # Evaluate initial scores
    scores = [_score(p) for p in candidates]
    best_score = max(scores)
    best_pts = candidates[scores.index(best_score)].copy()
    
    # Sort candidates by initial score (best first)
    cand_scores = sorted(zip(scores, candidates), reverse=True)
    
    # Allocate time budget for local optimization
    # We have ~140s total.
    # Let's give 20s to the top 3 candidates, 10s to the next 2.
    # Then use remaining time for final polish.
    
    time_alloc = [20.0, 15.0, 10.0, 5.0, 5.0]
    
    for i, (score, pts) in enumerate(cand_scores):
        if time.time() - start_time > TIME_LIMIT - 20.0:
            break
            
        alloc = time_alloc[i] if i < len(time_alloc) else 5.0
        # If this candidate is already significantly better, give it more time?
        # Or just run the fast optimizer.
        
        opt_pts = _fast_local_optimize(pts, time_budget=alloc)
        opt_score = _score(opt_pts)
        
        if opt_score > best_score:
            best_score = opt_score
            best_pts = opt_pts.copy()
            
    # Final polish with L-BFGS-B on the best candidate
    if time.time() - start_time < TIME_LIMIT - 10.0:
        # Use a high beta soft-min for precision
        beta_final = 1000.0
        
        def _obj_grad(x_flat):
            pts = x_flat.reshape(-1, 2)
            n = pts.shape[0]
            d2 = np.sum(pts * pts, axis=1)[:, None] - 2.0 * (pts @ pts.T) + np.sum(pts * pts, axis=1)[None, :]
            np.fill_diagonal(d2, np.inf)
            d2 = np.maximum(d2, 1e-12)
            pair_r = np.sqrt(d2) / 2.0
            pair_r_flat = pair_r[np.triu_indices(n, k=1)]
            rad = np.hypot(pts[:, 0], pts[:, 1])
            bound_r = 1.0 - rad
            all_r = np.concatenate([pair_r_flat, bound_r])
            
            x_min = np.min(all_r)
            x_shifted = all_r - x_min
            x_shifted = np.clip(x_shifted, 0, 50.0 / beta_final)
            exp_terms = np.exp(-beta_final * x_shifted)
            sum_exp = np.sum(exp_terms)
            weights = exp_terms / sum_exp
            
            obj = -np.min(all_r) # Just use hard min for objective value, soft for grad
            
            grad_r = np.zeros_like(x_flat)
            idx_i, idx_j = np.triu_indices(n, k=1)
            for k in range(len(idx_i)):
                i, j = idx_i[k], idx_j[k]
                w = weights[k]
                if w < 1e-15: continue
                d_val = np.sqrt(d2[i, j])
                if d_val < 1e-12: continue
                diff = (pts[i] - pts[j]) / (2.0 * d_val)
                grad_r[2*i] += w * diff[0]
                grad_r[2*i+1] += w * diff[1]
                grad_r[2*j] += w * (-diff[0])
                grad_r[2*j+1] += w * (-diff[1])
                
            for i in range(n):
                w = weights[len(idx_i) + i]
                if w < 1e-15: continue
                r_i = rad[i]
                if r_i < 1e-12: continue
                grad_r[2*i] += w * (-pts[i, 0] / r_i)
                grad_r[2*i+1] += w * (-pts[i, 1] / r_i)
                
            return obj, -grad_r
            
        try:
            result = minimize(
                _obj_grad,
                best_pts.flatten(),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 50, 'maxfun': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'disp': False}
            )
            if result.x is not None and np.all(np.isfinite(result.x)):
                new_pts = result.x.reshape(N, 2)
                rad_new = np.hypot(new_pts[:, 0], new_pts[:, 1])
                out_new = rad_new > 1.0
                if np.any(out_new):
                    new_pts[out_new] = new_pts[out_new] / rad_new[out_new, None]
                new_score = _score(new_pts)
                if new_score > best_score:
                    best_score = new_score
                    best_pts = new_pts.copy()
        except Exception:
            pass

    # Final safety
    best_pts = np.asarray(best_pts, dtype=np.float64)
    if best_pts.shape != (N, 2) or not np.all(np.isfinite(best_pts)):
        best_pts = _hex_init(N)
        
    rad = np.hypot(best_pts[:, 0], best_pts[:, 1])
    out = rad > 1.0
    if np.any(out):
        best_pts[out] = best_pts[out] / rad[out, None]
        
    return best_pts
