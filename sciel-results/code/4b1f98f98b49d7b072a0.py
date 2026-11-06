"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def _score(pts):
    """Largest common radius for a set of centres."""
    if pts is None or len(pts) == 0:
        return 0.0
    pts = np.asarray(pts, dtype=float)
    if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
        return 0.0
    wall = np.minimum(
        np.minimum(pts[:, 0] + 0.5, 0.5 - pts[:, 0]),
        np.minimum(pts[:, 1] + 0.5, 0.5 - pts[:, 1])
    )
    wall_min = wall.min()
    
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(d, np.inf)
    nn = d.min()
    
    return min(wall_min, 0.5 * nn)

def _neg_score(pts):
    return -_score(pts)

def _soft_min_score(pts, beta=50.0):
    """
    Smooth approximation of the minimum radius using soft-min.
    This is differentiable and suitable for gradient-based optimization.
    """
    pts = np.asarray(pts, dtype=float)
    if pts.shape != (N, 2):
        return 0.0
    
    # Wall distances
    wall = np.minimum(
        np.minimum(pts[:, 0] + 0.5, 0.5 - pts[:, 0]),
        np.minimum(pts[:, 1] + 0.5, 0.5 - pts[:, 1])
    )
    
    # Pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    d = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(d, np.inf)
    pair_dist = 0.5 * d
    
    # Combine all constraints into a single vector
    all_constraints = np.concatenate([wall, pair_dist.ravel()])
    
    # Soft-min using log-sum-exp
    # min(x) ≈ (1/beta) * log(sum(exp(-beta * x)))
    # For numerical stability
    min_val = np.min(all_constraints)
    exp_vals = np.exp(-beta * (all_constraints - min_val))
    soft_min = min_val + np.log(exp_vals.sum()) / beta
    
    return soft_min

def _neg_soft_min(pts, beta=50.0):
    return -_soft_min_score(pts, beta)

def _make_hex_initial():
    """
    Create a hexagonal lattice configuration for 50 points.
    """
    configs = [
        [6, 7, 6, 7, 6, 7, 6, 5],
        [7, 6, 7, 6, 7, 6, 7, 4],
        [8, 7, 6, 7, 6, 7, 6, 3],
        [5, 6, 7, 6, 7, 6, 7, 6],
        [7, 7, 7, 7, 7, 7, 7, 1],
        [6, 6, 6, 6, 6, 6, 6, 6],
        [5, 5, 5, 5, 5, 5, 5, 10],
        [8, 8, 8, 8, 8, 8, 6, 4],
        [9, 8, 8, 8, 8, 8, 5, 4],
        [10, 9, 8, 8, 8, 7, 6, 4],
    ]
    
    best_pts = None
    best_score = -1
    
    for rows in configs:
        if sum(rows) != N:
            continue
            
        pts = []
        max_row = max(rows)
        spacing = 1.0 / max_row
        h = spacing * np.sqrt(3.0) / 2.0
        
        total_height = (len(rows) - 1) * h
        start_y = -total_height / 2.0
        
        for i, n in enumerate(rows):
            y = start_y + i * h
            offset = (0.5 * spacing) if (i % 2 != 0) else 0.0
            xs = np.arange(n) * spacing
            center = (xs[0] + xs[-1]) / 2.0 if n > 1 else 0.0
            xs = xs - center + offset
            for x in xs:
                pts.append([x, y])
        
        pts = np.array(pts)
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts

    return best_pts

def _push_apart(pts, r_target, iters=1000, learning_rate=0.001):
    """
    Local search that pushes overlapping circles apart.
    This targets the binding constraint directly.
    """
    best_pts = pts.copy()
    best_score = _score(pts)
    
    for it in range(iters):
        # Compute all pairwise distances and wall distances
        diff = best_pts[:, None, :] - best_pts[None, :, :]
        d = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(d, np.inf)
        
        # Wall distances
        wall = np.minimum(
            np.minimum(best_pts[:, 0] + 0.5, 0.5 - best_pts[:, 0]),
            np.minimum(best_pts[:, 1] + 0.5, 0.5 - best_pts[:, 1])
        )
        
        current_r = min(wall.min(), 0.5 * d.min())
        
        if current_r >= r_target:
            break
        
        # Find violating pairs and wall violations
        force = np.zeros_like(best_pts)
        
        # Wall violations: push away from nearest wall
        for i in range(N):
            px, py = best_pts[i]
            d_left = px + 0.5
            d_right = 0.5 - px
            d_bottom = py + 0.5
            d_top = 0.5 - py
            
            min_d = min(d_left, d_right, d_bottom, d_top)
            if min_d < r_target:
                if min_d == d_left:
                    force[i, 0] += (r_target - min_d)
                elif min_d == d_right:
                    force[i, 0] -= (r_target - min_d)
                elif min_d == d_bottom:
                    force[i, 1] += (r_target - min_d)
                elif min_d == d_top:
                    force[i, 1] -= (r_target - min_d)
        
        # Pair violations: push apart
        for i in range(N):
            for j in range(i + 1, N):
                dist = d[i, j]
                if dist < 2 * r_target:
                    vec = best_pts[i] - best_pts[j]
                    norm = np.linalg.norm(vec)
                    if norm > 1e-12:
                        vec = vec / norm
                        overlap = 2 * r_target - dist
                        force[i] += overlap * vec
                        force[j] -= overlap * vec
        
        # Apply force with learning rate
        best_pts += learning_rate * force
        
        # Clamp to square
        best_pts = np.clip(best_pts, -0.5, 0.5)
        
        new_score = _score(best_pts)
        if new_score > best_score:
            best_score = new_score
    
    return best_pts, best_score

def _gradient_descent_soft(pts, beta=50.0, iters=500, lr=0.001):
    """
    Gradient descent on the soft-min objective.
    """
    best_pts = pts.copy()
    best_score = _score(pts)
    
    for it in range(iters):
        # Numerical gradient
        eps = 1e-6
        grad = np.zeros_like(best_pts)
        for i in range(N):
            for j in range(2):
                pts_plus = best_pts.copy()
                pts_minus = best_pts.copy()
                pts_plus[i, j] += eps
                pts_minus[i, j] -= eps
                grad[i, j] = (_soft_min_score(pts_plus, beta) - _soft_min_score(pts_minus, beta)) / (2 * eps)
        
        # Update
        best_pts -= lr * grad
        best_pts = np.clip(best_pts, -0.5, 0.5)
        
        new_score = _score(best_pts)
        if new_score > best_score:
            best_score = new_score
        else:
            # Reduce learning rate
            lr *= 0.95
            if lr < 1e-8:
                break
    
    return best_pts, best_score

def _local_refinement(pts, iters=20, seed=42):
    """
    Perturbation + optimize cycles to escape local minima.
    Uses random perturbations followed by L-BFGS-B on soft-min.
    """
    best_pts = pts.copy()
    best_score = _score(pts)
    rng = np.random.RandomState(seed)
    
    for i in range(iters):
        # Random perturbation
        scale = 0.01 * (1.0 - i / iters)
        perturbed = best_pts + rng.normal(0, scale, (N, 2))
        perturbed = np.clip(perturbed, -0.5, 0.5)
        
        # Optimize perturbed configuration
        try:
            res = minimize(_neg_soft_min, perturbed.flatten(), method="L-BFGS-B",
                           options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 10})
            new_pts = res.x.reshape(N, 2)
            new_score = _score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts
        except Exception:
            pass
    
    return best_pts, best_score

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # Safety margin
    
    best_pts = _make_hex_initial()
    best_score = _score(best_pts)
    
    if best_pts is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]
    
    # Phase 1: Nelder-Mead coarse optimization
    try:
        res = minimize(_neg_score, best_pts.flatten(), method="Nelder-Mead",
                       options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 3000, 'adaptive': True})
        if time.time() > deadline - 60.0:
            return best_pts.reshape(N, 2)
            
        new_pts = res.x.reshape(N, 2)
        new_score = _score(new_pts)
        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts
    except Exception:
        pass

    # Phase 2: Push-apart local search
    if time.time() < deadline - 50.0:
        target_r = best_score + 0.001
        grad_pts, grad_score = _push_apart(best_pts, target_r, iters=2000, learning_rate=0.0005)
        if grad_score > best_score:
            best_score = grad_score
            best_pts = grad_pts
    
    # Phase 3: L-BFGS-B on soft-min objective
    if time.time() < deadline - 40.0:
        try:
            res2 = minimize(_neg_soft_min, best_pts.flatten(), method="L-BFGS-B",
                           options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8, 'maxls': 20})
            if time.time() > deadline - 30.0:
                return best_pts.reshape(N, 2)
                
            new_pts = res2.x.reshape(N, 2)
            new_score = _score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts
        except Exception:
            pass

    # Phase 4: Local refinement with perturbation cycles
    if time.time() < deadline - 30.0:
        ref_pts, ref_score = _local_refinement(best_pts, iters=20, seed=42)
        if ref_score > best_score:
            best_score = ref_score
            best_pts = ref_pts
    
    # Phase 5: Another Nelder-Mead pass
    if time.time() < deadline - 20.0:
        try:
            res3 = minimize(_neg_score, best_pts.flatten(), method="Nelder-Mead",
                           options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 2000, 'adaptive': True})
            if time.time() > deadline - 10.0:
                return best_pts.reshape(N, 2)
                
            new_pts = res3.x.reshape(N, 2)
            new_score = _score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts
        except Exception:
            pass

    # Phase 6: High-beta polish with L-BFGS-B
    if time.time() < deadline - 10.0:
        try:
            def _neg_soft_min_high_beta(pts):
                return -_soft_min_score(pts, beta=500.0)
            
            res4 = minimize(_neg_soft_min_high_beta, best_pts.flatten(), method="L-BFGS-B",
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 10})
            
            new_pts = res4.x.reshape(N, 2)
            new_score = _score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts
        except Exception:
            pass

    # Phase 7: Final push-apart polish
    if time.time() < deadline - 5.0:
        target_r = best_score + 0.0005
        grad_pts, grad_score = _push_apart(best_pts, target_r, iters=1000, learning_rate=0.0002)
        if grad_score > best_score:
            best_score = grad_score
            best_pts = grad_pts

    return best_pts.reshape(N, 2)

# EVOLVE-BLOCK-END
