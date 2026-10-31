"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles."""
import numpy as np
from scipy.optimize import linprog
import time

N = 26
DEADLINE = 150.0  # seconds
SAFETY_MARGIN = 5.0  # return this many seconds before deadline


def _solve_radii_lp(centers):
    """Given centers, solve LP to maximize sum of radii.
    
    Variables: r_0, ..., r_{n-1}
    Maximize: sum(r_i)
    Subject to:
      r_i + r_j <= dist(i,j) for all i < j
      r_i <= min(x_i, 1-x_i, y_i, 1-y_i) for all i
      r_i >= 0
    """
    n = len(centers)
    centers = np.asarray(centers)
    
    # Objective: maximize sum(r) => minimize -sum(r)
    c_obj = -np.ones(n)
    
    # Inequality constraints: A_ub @ r <= b_ub
    # Pairwise: r_i + r_j <= d_ij
    A_rows = []
    b_vals = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_rows.append(row)
            b_vals.append(d)
    
    # Boundary: r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
    for i in range(n):
        bound = min(centers[i][0], 1.0 - centers[i][0],
                    centers[i][1], 1.0 - centers[i][1])
        row = np.zeros(n)
        row[i] = 1.0
        A_rows.append(row)
        b_vals.append(bound)
    
    A_ub = np.array(A_rows)
    b_ub = np.array(b_vals)
    
    # Bounds: 0 <= r_i <= bound_i
    bounds = []
    for i in range(n):
        bound = min(centers[i][0], 1.0 - centers[i][0],
                    centers[i][1], 1.0 - centers[i][1])
        bounds.append((0.0, bound))
    
    result = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if result.success:
        radii = result.x
        # Clip tiny negatives
        radii = np.maximum(radii, 0.0)
        return radii
    else:
        # Fallback: equal small radii
        return np.full(n, 0.01)


def _grid_initial(side):
    """Generate a grid of centers."""
    centers = []
    for i in range(side):
        for j in range(side):
            centers.append(((i + 0.5) / side, (j + 0.5) / side))
    return centers[:N]


def _random_initial():
    """Generate random centers in [0.05, 0.95]^2."""
    return np.random.uniform(0.05, 0.95, size=(N, 2))


def _optimize_centers(centers, max_iters=200, step_size=0.01):
    """Iteratively optimize centers to improve the LP objective.
    
    Uses subgradient method based on LP dual variables.
    """
    centers = np.array(centers, dtype=float)
    best_score = -np.inf
    best_centers = centers.copy()
    
    for it in range(max_iters):
        radii = _solve_radii_lp(centers)
        score = np.sum(radii)
        
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
        
        # Compute subgradient with respect to centers
        # The LP dual gives us which constraints are active
        # Simplified: push overlapping circles apart
        grad = np.zeros_like(centers)
        for i in range(N):
            for j in range(i + 1, N):
                d = centers[i] - centers[j]
                dist = np.linalg.norm(d)
                if dist < 1e-8:
                    # Perturb
                    d = np.array([1e-5, 1e-5])
                    dist = np.linalg.norm(d)
                # If circles are close, push apart
                # Weight by how "tight" the constraint is
                # Approximate: if dist < r_i + r_j + margin, push
                margin = 0.01
                if dist < radii[i] + radii[j] + margin:
                    strength = (radii[i] + radii[j] + margin - dist) / (margin + 1e-8)
                    strength = min(strength, 1.0)
                    unit = d / dist
                    grad[i] += strength * unit
                    grad[j] -= strength * unit
        
        # Also push away from boundaries
        for i in range(N):
            x, y = centers[i]
            # Boundary pressure
            if x < radii[i]:
                grad[i][0] += (radii[i] - x) * 2.0
            if x > 1.0 - radii[i]:
                grad[i][0] -= (x - (1.0 - radii[i])) * 2.0
            if y < radii[i]:
                grad[i][1] += (radii[i] - y) * 2.0
            if y > 1.0 - radii[i]:
                grad[i][1] -= (y - (1.0 - radii[i])) * 2.0
        
        # Update centers
        centers = centers + step_size * grad
        # Clip to valid region
        centers = np.clip(centers, 0.01, 0.99)
    
    return best_centers, best_score


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - SAFETY_MARGIN
    
    best_result = None
    best_score = -np.inf
    
    # Strategy 1: Grid initial
    side = int(np.ceil(np.sqrt(N)))
    for offset_x in [0.0, 0.02, -0.02]:
        for offset_y in [0.0, 0.02, -0.02]:
            if time.time() > deadline:
                break
            centers = _grid_initial(side)
            centers = np.array(centers) + np.array([offset_x, offset_y])
            centers = np.clip(centers, 0.01, 0.99)
            
            opt_centers, score = _optimize_centers(centers, max_iters=100, step_size=0.005)
            radii = _solve_radii_lp(opt_centers)
            final_score = np.sum(radii)
            
            if final_score > best_score:
                best_score = final_score
                result = np.column_stack([opt_centers, radii])
                best_result = result
    
    if time.time() > deadline:
        if best_result is not None:
            return best_result
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
    
    # Strategy 2: Random restarts
    np.random.seed(42)
    for trial in range(50):
        if time.time() > deadline:
            break
        
        centers = _random_initial()
        opt_centers, score = _optimize_centers(centers, max_iters=80, step_size=0.005)
        radii = _solve_radii_lp(opt_centers)
        final_score = np.sum(radii)
        
        if final_score > best_score:
            best_score = final_score
            best_result = np.column_stack([opt_centers, radii])
    
    if best_result is None:
        # Fallback to grid
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) 
                 for i in range(side) for j in range(side)]
        best_result = np.array(cells[:N])
    
    return best_result
