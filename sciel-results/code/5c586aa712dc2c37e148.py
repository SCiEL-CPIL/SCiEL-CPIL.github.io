"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 21

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

DEADLINE = 140.0  # safety margin before 150s kill
START_TIME = time.time()


def _remaining():
    return DEADLINE - (time.time() - START_TIME)


def _circle_overlap_penalty(circles, soft_min_eps=1e-4):
    """
    Penalty for overlapping circles.
    circles: array of shape (n, 3) with [x, y, r]
    We want all pairs to be disjoint: dist(i,j) >= r_i + r_j
    Penalty: sum over pairs of max(0, r_i + r_j - dist)^2
    """
    n = len(circles)
    if n == 0:
        return 0.0
    penalty = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = circles[i, 0] - circles[j, 0]
            dy = circles[i, 1] - circles[j, 1]
            d = np.sqrt(dx * dx + dy * dy)
            min_dist = circles[i, 2] + circles[j, 2]
            if d < min_dist:
                penalty += (min_dist - d) ** 2
    return penalty


def _boundary_penalty(circles, W, H):
    """
    Penalty for circles extending outside the rectangle [0, W] x [0, H].
    Each circle must satisfy: r <= x <= W - r and r <= y <= H - r
    """
    penalty = 0.0
    for i in range(len(circles)):
        x, y, r = circles[i]
        if x < r:
            penalty += (r - x) ** 2
        elif x > W - r:
            penalty += (x - (W - r)) ** 2
        if y < r:
            penalty += (r - y) ** 2
        elif y > H - r:
            penalty += (y - (H - r)) ** 2
    return penalty


def _total_penalty(circles, W, H):
    return _circle_overlap_penalty(circles) + _boundary_penalty(circles, W, H)


def _objective(params, W, H, n):
    """
    Objective to minimize: penalty - lambda * sum(radii)
    We want to maximize sum(radii) subject to no overlaps and within bounds.
    params: flattened array of length 3*n: [x0,y0,r0, x1,y1,r1, ...]
    """
    circles = params.reshape(n, 3)
    # Ensure radii are positive
    radii = np.maximum(circles[:, 2], 0.01)
    penalty = _circle_overlap_penalty(circles) + _boundary_penalty(circles, W, H)
    sum_radii = np.sum(radii)
    # We minimize penalty - lambda*sum_radii. Large lambda encourages larger radii.
    # But we need to balance. Let's use a different approach:
    # Minimize penalty, with radii as part of the variables, and add a small negative
    # term to encourage growth, or just maximize sum_radii subject to penalty < threshold.
    
    # Actually, let's use: minimize (penalty - beta * sum_radii)
    # where beta is a weight. We want penalty to be ~0 and sum_radii to be large.
    beta = 1.0
    return penalty - beta * sum_radii


def _solve_for_fixed_dims(W, H, n, initial_circles, max_iter=5000):
    """
    Solve for circle positions and radii given fixed rectangle W x H.
    Returns the best circles found.
    """
    n = len(initial_circles)
    x0 = initial_circles.flatten()
    
    # Bounds: x in [0, W], y in [0, H], r in [0.01, min(W,H)/2]
    r_max = min(W, H) / 2.0
    bounds = []
    for i in range(n):
        bounds.append((0.0, W))
        bounds.append((0.0, H))
        bounds.append((0.01, r_max))
    
    def obj(params):
        circles = params.reshape(n, 3)
        # Clip radii to positive
        radii = np.maximum(circles[:, 2], 0.01)
        penalty = _circle_overlap_penalty(circles) + _boundary_penalty(circles, W, H)
        sum_radii = np.sum(radii)
        return penalty - 1.0 * sum_radii
    
    try:
        res = minimize(obj, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': max_iter, 'ftol': 1e-10, 'gtol': 1e-8})
        best_circles = res.x.reshape(n, 3)
        # Ensure radii positive
        best_circles[:, 2] = np.maximum(best_circles[:, 2], 0.01)
        return best_circles
    except Exception:
        return initial_circles


def _generate_initial_grid(n, W, H, spacing_factor=1.0):
    """
    Generate initial circle positions on a grid.
    """
    side = int(np.ceil(np.sqrt(n)))
    cells = []
    for i in range(side):
        for j in range(side):
            if len(cells) >= n:
                break
            x = (i + 0.5) * W / side
            y = (j + 0.5) * H / side
            r = 0.45 * min(W, H) / side * spacing_factor
            cells.append([x, y, r])
    return np.array(cells[:n])


def _perturb(circles, epsilon=0.05):
    """
    Add random perturbation to circle positions and radii.
    """
    n = len(circles)
    perturbed = circles.copy()
    for i in range(n):
        perturbed[i, 0] += np.random.uniform(-epsilon, epsilon)
        perturbed[i, 1] += np.random.uniform(-epsilon, epsilon)
        perturbed[i, 2] *= np.random.uniform(0.9, 1.1)
    return perturbed


def construct():
    global START_TIME
    START_TIME = time.time()
    
    best_score = -np.inf
    best_circles = None
    
    n = N
    
    # Try several aspect ratios for the rectangle
    # Perimeter = 2*(W + H) = 4, so W + H = 2
    # Aspect ratio W/H = a, so W = 2a/(1+a), H = 2/(1+a)
    
    aspect_ratios = [1.0, 0.8, 1.25, 0.6, 1.67, 0.5, 2.0, 0.4, 2.5, 0.3, 3.0, 0.25, 4.0, 0.2, 5.0, 0.15]
    
    for ar in aspect_ratios:
        if _remaining() < 10.0:
            break
            
        W = 2.0 * ar / (1.0 + ar)
        H = 2.0 / (1.0 + ar)
        
        # Generate initial configuration
        init = _generate_initial_grid(n, W, H)
        
        # Try multiple perturbations
        n_perturbations = 3
        for p in range(n_perturbations):
            if _remaining() < 5.0:
                break
                
            if p == 0:
                current = init.copy()
            else:
                current = _perturb(init, epsilon=0.03 * min(W, H))
            
            # Solve
            result = _solve_for_fixed_dims(W, H, n, current, max_iter=3000)
            
            # Evaluate
            penalty = _total_penalty(result, W, H)
            sum_radii = np.sum(result[:, 2])
            
            if penalty < 1e-6:
                score = sum_radii
                if score > best_score:
                    best_score = score
                    best_circles = result.copy()
            else:
                # Penalized score
                score = sum_radii - penalty * 10.0
                if score > best_score and penalty < 0.1:
                    best_score = score
                    best_circles = result.copy()
    
    if best_circles is None:
        # Fallback: simple grid
        side = int(np.ceil(np.sqrt(n)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        best_circles = np.array(cells[:n])
        # Scale to fit perimeter 4 square (1x1)
        # This is already in [0,1]x[0,1] which has perimeter 4
        return best_circles
    
    return best_circles

# EVOLVE-BLOCK-END
