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

def _make_hex_initial():
    """
    Create a hexagonal lattice configuration for 50 points.
    We'll try a few different row configurations.
    """
    configs = [
        [6, 7, 6, 7, 6, 7, 6, 5],
        [7, 6, 7, 6, 7, 6, 7, 4],
        [8, 7, 6, 7, 6, 7, 6, 3],
        [5, 6, 7, 6, 7, 6, 7, 6]
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

def _gradient_step(pts, step_size=0.01, iters=500):
    """
    Simple gradient ascent on the minimum distance using subgradients.
    For the min of distances, the subgradient is the average of the 
    normalized vectors to the points that achieve the minimum.
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
        
        # The radius is min(wall_min, 0.5 * min_pairwise)
        min_pair = d.min()
        wall_min = wall.min()
        current_r = min(wall_min, 0.5 * min_pair)
        
        # Determine which constraints are active
        # If wall_min < 0.5 * min_pair, wall is the bottleneck
        # Otherwise, pairwise is the bottleneck
        
        grad = np.zeros_like(best_pts)
        
        if wall_min <= 0.5 * min_pair:
            # Wall is bottleneck: move points away from the nearest wall
            # Find which points are closest to walls
            min_wall_idx = np.argmin(wall)
            # Compute gradient for that point
            px, py = best_pts[min_wall_idx]
            # Distance to each wall
            d_left = px + 0.5
            d_right = 0.5 - px
            d_bottom = py + 0.5
            d_top = 0.5 - py
            
            # Move away from the nearest wall
            min_d = min(d_left, d_right, d_bottom, d_top)
            if min_d == d_left:
                grad[min_wall_idx, 0] += 1.0
            elif min_d == d_right:
                grad[min_wall_idx, 0] -= 1.0
            elif min_d == d_bottom:
                grad[min_wall_idx, 1] += 1.0
            elif min_d == d_top:
                grad[min_wall_idx, 1] -= 1.0
        else:
            # Pairwise is bottleneck: move points apart
            # Find pairs that achieve the minimum distance
            min_dist = 0.5 * min_pair
            # We want to increase min_pair, so we move points apart
            # Find all pairs with distance close to min_pair
            threshold = min_pair * 1.01
            for i in range(N):
                for j in range(i + 1, N):
                    if d[i, j] <= threshold:
                        # Move i away from j and j away from i
                        vec = best_pts[i] - best_pts[j]
                        norm = np.linalg.norm(vec)
                        if norm > 1e-12:
                            vec = vec / norm
                            grad[i] += vec
                            grad[j] -= vec
        
        # Normalize gradient
        norm = np.linalg.norm(grad)
        if norm > 1e-12:
            grad = grad / norm
        
        # Take a step
        new_pts = best_pts + step_size * grad
        
        # Clamp to square
        new_pts = np.clip(new_pts, -0.5, 0.5)
        
        new_score = _score(new_pts)
        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts
        else:
            # Reduce step size if no improvement
            step_size *= 0.95
            if step_size < 1e-8:
                break
    
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
    
    # First, use Nelder-Mead for coarse optimization
    try:
        res = minimize(_neg_score, best_pts.flatten(), method="Nelder-Mead",
                       options={'xatol': 1e-4, 'fatol': 1e-6, 'maxiter': 2000, 'adaptive': True})
        if time.time() > deadline - 30.0:
            return best_pts.reshape(N, 2)
            
        new_pts = res.x.reshape(N, 2)
        new_score = _score(new_pts)
        if new_score > best_score:
            best_score = new_score
            best_pts = new_pts
    except Exception:
        pass

    # Second, use gradient steps for fine-tuning
    if time.time() < deadline - 10.0:
        grad_pts, grad_score = _gradient_step(best_pts, step_size=0.005, iters=1000)
        if grad_score > best_score:
            best_score = grad_score
            best_pts = grad_pts
    
    # Third, try another Nelder-Mead pass from the gradient result
    if time.time() < deadline - 10.0:
        try:
            res2 = minimize(_neg_score, best_pts.flatten(), method="Nelder-Mead",
                            options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 1500, 'adaptive': True})
            if time.time() > deadline - 5.0:
                return best_pts.reshape(N, 2)
                
            new_pts = res2.x.reshape(N, 2)
            new_score = _score(new_pts)
            if new_score > best_score:
                best_score = new_score
                best_pts = new_pts
        except Exception:
            pass

    # Fourth, final gradient polish
    if time.time() < deadline - 5.0:
        grad_pts, grad_score = _gradient_step(best_pts, step_size=0.001, iters=500)
        if grad_score > best_score:
            best_score = grad_score
            best_pts = grad_pts

    return best_pts.reshape(N, 2)

# EVOLVE-BLOCK-END
