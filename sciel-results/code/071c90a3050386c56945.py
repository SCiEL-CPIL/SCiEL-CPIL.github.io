"""Circle packing in unit square: pack 26 circles, maximize sum of radii."""
import numpy as np
from scipy.optimize import minimize
import time

N = 26


def _penalty(x):
    """
    x: array of shape (N*3,) -> [x0,y0,r0, x1,y1,r1, ...]
    Returns (objective, constraints_violation).
    Objective: -sum(r) (we minimize negative sum)
    Constraints:
      - circles inside [0,1]^2: x_i >= r_i, x_i + r_i <= 1, y_i >= r_i, y_i + r_i <= 1
      - disjoint interiors: dist(i,j) >= r_i + r_j
    """
    pts = x.reshape(N, 3)
    xs, ys, rs = pts[:, 0], pts[:, 1], pts[:, 2]
    
    # Objective: maximize sum of radii -> minimize -sum
    obj = -np.sum(rs)
    
    # Compute all pairwise distances
    # dist[i,j] = sqrt((xi-xj)^2 + (yi-yj)^2)
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    
    # Required separation: r_i + r_j
    sep = rs[:, None] + rs[None, :]
    
    # Violation for disjointness: max(0, sep - dist)
    # Only consider i < j
    iu = np.triu_indices(N, k=1)
    sep_ij = sep[iu]
    dist_ij = dist[iu]
    viol_disjoint = np.maximum(0, sep_ij - dist_ij)
    
    # Boundary violations
    viol_left = np.maximum(0, rs - xs)
    viol_right = np.maximum(0, rs - (1.0 - xs))
    viol_bottom = np.maximum(0, rs - ys)
    viol_top = np.maximum(0, rs - (1.0 - ys))
    
    total_viol = np.sum(viol_disjoint**2) + np.sum(viol_left**2) + np.sum(viol_right**2) + \
                 np.sum(viol_bottom**2) + np.sum(viol_top**2)
    
    return obj, total_viol


def _penalty_objective(x, penalty_weight=1000.0):
    """Combined objective: minimize -sum(r) + penalty_weight * violation."""
    obj, viol = _penalty(x)
    return obj + penalty_weight * viol


def _initial_grid(n):
    """Generate an initial grid layout for n circles."""
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= n:
                break
            cx = (i + 0.5) / side
            cy = (j + 0.5) / side
            r = 0.5 / side - 1e-4
            cells.append([cx, cy, r])
    return np.array(cells[:n])


def _initial_random(n, seed=42):
    """Generate a random initial layout."""
    rng = np.random.RandomState(seed)
    pts = []
    for _ in range(n):
        r = 0.03 + rng.random() * 0.02
        x = r + rng.random() * (1.0 - 2 * r)
        y = r + rng.random() * (1.0 - 2 * r)
        pts.append([x, y, r])
    return np.array(pts)


def _local_search(x0, max_iters=5000, tol=1e-8, penalty_start=100.0, penalty_end=10000.0):
    """
    Perform local optimization using scipy's L-BFGS-B with adaptive penalty.
    """
    x = x0.copy()
    
    # Adaptive penalty schedule
    for phase in range(5):
        pw = penalty_start * (penalty_end / penalty_start) ** (phase / 4.0)
        
        def objective(xx):
            return _penalty_objective(xx, pw)
        
        result = minimize(objective, x, method='L-BFGS-B', 
                         options={'maxiter': max_iters // 5, 'ftol': tol, 'gtol': tol})
        x = result.x
        
        # Check if we're close to feasible
        _, viol = _penalty(x)
        if viol < 1e-6:
            break
    
    return x


def _is_valid(x, tol=1e-7):
    """Check if the configuration is valid (all circles inside and disjoint)."""
    pts = x.reshape(N, 3)
    xs, ys, rs = pts[:, 0], pts[:, 1], pts[:, 2]
    
    # Boundary check
    if np.any(xs < rs - tol) or np.any(xs + rs > 1.0 + tol):
        return False
    if np.any(ys < rs - tol) or np.any(ys + rs > 1.0 + tol):
        return False
    
    # Disjoint check
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    sep = rs[:, None] + rs[None, :]
    
    iu = np.triu_indices(N, k=1)
    if np.any(dist[iu] < sep[iu] - tol):
        return False
    
    return True


def _score(x):
    """Compute the score: sum of radii if valid, else -inf."""
    if not _is_valid(x):
        return -np.inf
    return np.sum(x.reshape(N, 3)[:, 2])


def construct():
    start_time = time.time()
    deadline = start_time + 120.0  # 120s wall-clock limit
    
    best_x = None
    best_score = -np.inf
    
    # Multiple initializations and local searches
    inits = []
    
    # Grid initialization
    inits.append(_initial_grid(N))
    
    # Several random initializations with different seeds
    for seed in range(10):
        inits.append(_initial_random(N, seed=seed))
    
    # Also try a few structured initializations
    # Hexagonal-like packing
    side = 5
    hex_pts = []
    for i in range(side):
        for j in range(side):
            if len(hex_pts) >= N:
                break
            cx = (i + 0.5 + 0.5 * (j % 2)) / (side + 0.5)
            cy = (j + 0.5) / (side + 0.5)
            r = 0.5 / (side + 0.5) - 1e-4
            hex_pts.append([cx, cy, r])
    inits.append(np.array(hex_pts[:N]))
    
    for init in inits:
        if time.time() > deadline - 5.0:
            break
        
        try:
            x_opt = _local_search(init, max_iters=3000, tol=1e-7)
            s = _score(x_opt)
            if s > best_score:
                best_score = s
                best_x = x_opt.copy()
        except Exception:
            continue
    
    # If we still have time, do a final polish on the best solution
    if best_x is not None and time.time() < deadline - 10.0:
        try:
            def objective(xx):
                return _penalty_objective(xx, 10000.0)
            
            result = minimize(objective, best_x, method='L-BFGS-B',
                             options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-10})
            x_polished = result.x
            s = _score(x_polished)
            if s > best_score:
                best_score = s
                best_x = x_polished
        except Exception:
            pass
    
    if best_x is None:
        # Fallback: return grid
        return _initial_grid(N).reshape(-1)
    
    return best_x


if __name__ == "__main__":
    result = construct()
    print(f"Result shape: {result.shape}")
    print(f"Sum of radii: {np.sum(result.reshape(N, 3)[:, 2]):.6f}")
    print(f"Valid: {_is_valid(result)}")
