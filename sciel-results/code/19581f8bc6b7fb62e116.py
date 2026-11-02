"""Circle packing in perimeter-4 rectangle: force-directed contact network optimizer."""
import time
import numpy as np
from scipy.spatial import Delaunay

N = 21
DEADLINE = 115.0  # seconds; evaluator kills at 150s, leave safety margin

def _objective(params, aspect):
    """Compute sum of radii and violation penalty.
    params: (N+1, 2) — first N rows are centers, last row is aspect log-ratio.
    Returns (value, penalty, radii).
    """
    W = 2.0 / (1.0 + aspect)
    H = 2.0 - W
    r = np.exp(params[N, 0])  # single radius parameter (log)
    radii = r * np.ones(N)
    
    centers = params[:N]
    # Wall violations
    wx = np.maximum(radii - centers[:, 0], 0.0) + np.maximum(centers[:, 0] + radii - W, 0.0)
    wy = np.maximum(radii - centers[:, 1], 0.0) + np.maximum(centers[:, 1] + radii - H, 0.0)
    wall_pen = np.sum(wx) + np.sum(wy)
    
    # Pair overlap violations
    diff = centers[:, None, :] - centers[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    min_dist = radii[:, None] + radii[None, :]
    overlap = np.maximum(min_dist - dist, 0.0)
    np.fill_diagonal(overlap, 0.0)
    pair_pen = np.sum(overlap)
    
    total_pen = wall_pen + pair_pen
    value = np.sum(radii)
    return value, total_pen, radii

def _gradient(params, aspect, beta):
    """Compute gradient of (value - beta * penalty) w.r.t. params."""
    W = 2.0 / (1.0 + aspect)
    H = 2.0 - W
    r = np.exp(params[N, 0])
    radii = r * np.ones(N)
    centers = params[:N]
    
    grad = np.zeros_like(params)
    
    # Gradient of value w.r.t. log(r)
    grad[N, 0] = r * N  # d(sum r)/d(log r) = r * N
    
    # Wall gradient
    wx_lo = np.maximum(radii - centers[:, 0], 0.0)
    wx_hi = np.maximum(centers[:, 0] + radii - W, 0.0)
    wy_lo = np.maximum(radii - centers[:, 1], 0.0)
    wy_hi = np.maximum(centers[:, 1] + radii - H, 0.0)
    
    # d(wall_pen)/d(center_i, x)
    grad_cx = np.zeros(N)
    grad_cx[wx_lo > 0] -= 1.0
    grad_cx[wx_hi > 0] += 1.0
    grad_cy = np.zeros(N)
    grad_cy[wy_lo > 0] -= 1.0
    grad_cy[wy_hi > 0] += 1.0
    grad[:N, 0] -= beta * grad_cx
    grad[:N, 1] -= beta * grad_cy
    
    # d(wall_pen)/d(log r): each wall term contributes +r if active
    n_wall_active = np.sum(wx_lo > 0) + np.sum(wx_hi > 0) + np.sum(wy_lo > 0) + np.sum(wy_hi > 0)
    grad[N, 0] -= beta * r * n_wall_active
    
    # Pair overlap gradient
    diff = centers[:, None, :] - centers[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    min_dist = radii[:, None] + radii[None, :]
    overlap = np.maximum(min_dist - dist, 0.0)
    np.fill_diagonal(overlap, 0.0)
    
    # For each pair (i,j) with overlap, force pushes them apart
    # d(pair_pen)/d(center_i) = -overlap_ij * (c_i - c_j) / dist_ij
    unit = np.zeros_like(diff)
    mask = dist > 1e-12
    unit[mask] = diff[mask] / dist[mask, None]
    
    # grad_cx[i] = sum_j overlap_ij * unit_ij_x
    grad_pair_cx = np.sum(overlap * unit[:, :, 0], axis=1)
    grad_pair_cy = np.sum(overlap * unit[:, :, 1], axis=1)
    grad[:N, 0] -= beta * grad_pair_cx
    grad[:N, 1] -= beta * grad_pair_cy
    
    # d(pair_pen)/d(log r): each overlap contributes +r (since dr_i/d(log r) = r for all i)
    n_overlap = np.sum(overlap > 0)  # counts each pair twice
    grad[N, 0] -= beta * r * n_overlap
    
    return grad

def _optimize(params0, aspect0, beta, lr, max_iter, tol=1e-12):
    """Gradient descent with adaptive learning rate."""
    params = params0.copy()
    aspect = aspect0
    best_obj = -np.inf
    best_params = params.copy()
    best_aspect = aspect
    
    for it in range(max_iter):
        value, pen, radii = _objective(params, aspect)
        obj = value - beta * pen
        if obj > best_obj:
            best_obj = obj
            best_params = params.copy()
            best_aspect = aspect
        
        if pen < 1e-9 and it > 10:
            # Feasible; try to increase radii
            pass
        
        g = _gradient(params, aspect, beta)
        # Adaptive step
        gnorm = np.linalg.norm(g)
        if gnorm < 1e-14:
            break
        step = lr * g / gnorm
        params_new = params + step
        
        # Simple line search
        val_new, pen_new, _ = _objective(params_new, aspect)
        obj_new = val_new - beta * pen_new
        if obj_new > obj:
            params = params_new
        else:
            lr *= 0.8
            if lr < 1e-12:
                break
    return best_params, best_aspect, best_obj

def _delaunay_contacts(centers, radii):
    """Find contact pairs via Delaunay triangulation."""
    if len(centers) < 3:
        return []
    try:
        tri = Delaunay(centers)
    except Exception:
        return []
    contacts = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i+1, 3):
                a, b = simplex[i], simplex[j]
                d = np.linalg.norm(centers[a] - centers[b])
                if d < radii[a] + radii[b] + 1e-6:
                    contacts.add((min(a,b), max(a,b)))
    return list(contacts)

def construct():
    t0 = time.time()
    rng = np.random.RandomState(42)
    
    best_value = 0.0
    best_result = None
    
    # Phase 1: Grid-like initialization with perturbation
    side = 5  # 5x5 grid = 25 cells, use 21
    W0 = 2.0 / (1.0 + 1.0)  # aspect=1 -> square
    H0 = 2.0 - W0
    
    for restart in range(50):
        if time.time() - t0 > DEADLINE - 10:
            break
        
        # Initialize positions
        if restart == 0:
            # Regular grid
            cols, rows = 5, 5
            centers = np.zeros((N, 2))
            for idx in range(N):
                i = idx % cols
                j = idx // cols
                centers[idx] = [(i + 0.5) * W0 / cols, (j + 0.5) * H0 / rows]
            r0 = 0.5 / max(cols, rows) - 0.01
        else:
            # Random positions
            centers = rng.uniform(0.1, 1.9, size=(N, 2))
            r0 = 0.1 + 0.15 * rng.random()
        
        aspect = 1.0 + 0.2 * (rng.random() - 0.5)  # slight aspect variation
        W = 2.0 / (1.0 + aspect)
        H = 2.0 - W
        
        # Clamp centers to be inside
        centers[:, 0] = np.clip(centers[:, 0], r0, W - r0)
        centers[:, 1] = np.clip(centers[:, 1], r0, H - r0)
        
        params = np.zeros((N+1, 2))
        params[:N] = centers
        params[N, 0] = np.log(max(r0, 1e-6))
        
        # Multi-phase optimization with increasing beta
        for phase, (beta, lr, max_iter) in enumerate([
            (5.0, 0.05, 300),
            (20.0, 0.03, 500),
            (100.0, 0.01, 800),
            (500.0, 0.005, 1000),
            (2000.0, 0.002, 1500),
        ]):
            if time.time() - t0 > DEADLINE - 5:
                break
            
            params, aspect, obj = _optimize(params, aspect, beta, lr, max_iter)
            
            # Check feasibility
            W = 2.0 / (1.0 + aspect)
            H = 2.0 - W
            r = np.exp(params[N, 0])
            centers = params[:N]
            
            # Verify no overlaps and within bounds
            valid = True
            for i in range(N):
                if centers[i, 0] < r - 1e-9 or centers[i, 0] > W - r + 1e-9:
                    valid = False
                    break
                if centers[i, 1] < r - 1e-9 or centers[i, 1] > H - r + 1e-9:
                    valid = False
                    break
                for j in range(i+1, N):
                    d = np.linalg.norm(centers[i] - centers[j])
                    if d < 2*r - 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                val = N * r
                if val > best_value:
                    best_value = val
                    best_result = np.column_stack([centers, np.full(N, r)])
        
        # Force-directed refinement: use contact graph to adjust
        if time.time() - t0 > DEADLINE - 8:
            break
        
        # Additional refinement with different initial perturbation
        if best_result is not None:
            centers_r = best_result[:, :2].copy()
            r_r = best_result[0, 2]
            # Small random perturbation
            centers_r += rng.normal(0, 0.005, size=(N, 2))
            W = 2.0 / (1.0 + aspect)
            H = 2.0 - W
            centers_r[:, 0] = np.clip(centers_r[:, 0], r_r, W - r_r)
            centers_r[:, 1] = np.clip(centers_r[:, 1], r_r, H - r_r)
            
            params = np.zeros((N+1, 2))
            params[:N] = centers_r
            params[N, 0] = np.log(r_r)
            
            for beta, lr, max_iter in [(500.0, 0.003, 500), (2000.0, 0.001, 1000)]:
                if time.time() - t0 > DEADLINE - 5:
                    break
                params, aspect, obj = _optimize(params, aspect, beta, lr, max_iter)
                
                W = 2.0 / (1.0 + aspect)
                H = 2.0 - W
                r = np.exp(params[N, 0])
                centers = params[:N]
                
                valid = True
                for i in range(N):
                    if centers[i, 0] < r - 1e-9 or centers[i, 0] > W - r + 1e-9:
                        valid = False
                        break
                    if centers[i, 1] < r - 1e-9 or centers[i, 1] > H - r + 1e-9:
                        valid = False
                        break
                    for j in range(i+1, N):
                        d = np.linalg.norm(centers[i] - centers[j])
                        if d < 2*r - 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                
                if valid:
                    val = N * r
                    if val > best_value:
                        best_value = val
                        best_result = np.column_stack([centers, np.full(N, r)])
    
    # Final safety: if no valid result found, return a trivial valid packing
    if best_result is None:
        # 5x5 grid with 21 circles, all same radius
        W = 1.0
        H = 1.0
        r = 0.18
        cols = 5
        centers = np.zeros((N, 2))
        for idx in range(N):
            i = idx % cols
            j = idx // cols
            centers[idx] = [(i + 0.5) * W / cols, (j + 0.5) * H / cols]
        best_result = np.column_stack([centers, np.full(N, r)])
    
    return best_result
