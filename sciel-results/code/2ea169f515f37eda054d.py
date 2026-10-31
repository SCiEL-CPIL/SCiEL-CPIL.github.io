"""Optimized 16-point placement to minimize squared max/min pairwise distance ratio."""
import numpy as np
import time
import itertools

N = 16
D = 2
TIME_LIMIT = 115.0  # safety margin before 120s limit


def compute_score(points):
    """Compute (max_dist / min_dist)^2 for a set of points."""
    n = len(points)
    if n < 2:
        return float('inf')
    
    # Compute all pairwise squared distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dists = np.sum(diff ** 2, axis=2)
    
    # Exclude self-distances
    np.fill_diagonal(sq_dists, np.inf)
    
    min_sq = np.min(sq_dists)
    max_sq = np.max(sq_dists)
    
    if min_sq <= 1e-15:
        return float('inf')
    
    return (max_sq / min_sq)


def hexagonal_lattice_init():
    """Initialize points on a hexagonal lattice pattern."""
    # Use a hexagonal close-packing arrangement
    # For 16 points, use a 4x4 grid with hexagonal offset
    points = []
    # Hexagonal spacing
    a = 1.0
    # Row-based hexagonal arrangement
    for i in range(4):
        for j in range(4):
            x = j * a + (i % 2) * a * 0.5
            y = i * a * np.sqrt(3) / 2
            points.append([x, y])
    
    return np.array(points, dtype=float)


def square_grid_init():
    """Initialize points on a square grid."""
    points = []
    for i in range(4):
        for j in range(4):
            points.append([i, j])
    return np.array(points, dtype=float)


def random_init():
    """Random initialization in unit square."""
    return np.random.rand(N, 2)


def objective(points):
    """Objective function to minimize."""
    return compute_score(points)


def gradient_descent_optimize(points, max_iter=5000, lr_init=0.01, lr_decay=0.999):
    """Gradient descent with momentum to minimize the score."""
    points = points.copy()
    n = len(points)
    lr = lr_init
    momentum = np.zeros_like(points)
    beta = 0.9
    
    best_points = points.copy()
    best_score = objective(points)
    
    for it in range(max_iter):
        # Compute pairwise distances
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2) + 1e-15)
        np.fill_diagonal(dists, np.inf)
        
        min_d = np.min(dists)
        max_d = np.max(dists)
        
        if min_d < 1e-10:
            # Perturb overlapping points
            idx = np.argmin(dists)
            i, j = idx // n, idx % n
            points[i] += np.random.randn(2) * 0.01
            continue
        
        score = (max_d / min_d) ** 2
        
        if score < best_score:
            best_score = score
            best_points = points.copy()
        
        # Compute gradients
        # d(score)/d(points) via chain rule
        # score = (max_d/min_d)^2
        # dscore/dmax_d = 2 * max_d / min_d^2
        # dscore/dmin_d = -2 * max_d^2 / min_d^3
        
        dscore_dmax = 2 * max_d / min_d ** 2
        dscore_dmin = -2 * max_d ** 2 / min_d ** 3
        
        grad = np.zeros_like(points)
        
        # Find pairs achieving min and max distance
        min_idx = np.argmin(dists)
        max_idx = np.argmax(dists)
        i_min, j_min = min_idx // n, min_idx % n
        i_max, j_max = max_idx // n, max_idx % n
        
        # Gradient wrt min distance pair
        if i_min != j_min:
            diff_min = points[i_min] - points[j_min]
            dist_min = dists[i_min, j_min]
            if dist_min > 1e-10:
                grad[i_min] += dscore_dmin * diff_min / dist_min
                grad[j_min] -= dscore_dmin * diff_min / dist_min
        
        # Gradient wrt max distance pair
        if i_max != j_max:
            diff_max = points[i_max] - points[j_max]
            dist_max = dists[i_max, j_max]
            if dist_max > 1e-10:
                grad[i_max] += dscore_dmax * diff_max / dist_max
                grad[j_max] -= dscore_dmax * diff_max / dist_max
        
        # Momentum update
        momentum = beta * momentum - lr * grad
        points += momentum
        
        # Clamp to reasonable bounds
        points = np.clip(points, -10, 10)
        
        lr *= lr_decay
        
        if lr < 1e-6:
            break
    
    return best_points, best_score


def anneal_optimize(points, max_iter=10000, t_init=1.0, t_decay=0.999):
    """Simulated annealing to minimize the score."""
    points = points.copy()
    t = t_init
    best_points = points.copy()
    best_score = objective(points)
    current_score = best_score
    
    for it in range(max_iter):
        # Random perturbation
        new_points = points + np.random.randn(N, 2) * 0.1 * t
        
        new_score = objective(new_points)
        
        if new_score < current_score:
            points = new_points
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_points = points.copy()
        else:
            # Accept with probability exp(-(new_score - current_score) / t)
            delta = new_score - current_score
            if delta < 0 or np.random.random() < np.exp(-delta / max(t, 1e-10)):
                points = new_points
                current_score = new_score
        
        t *= t_decay
        if t < 1e-4:
            break
    
    return best_points, best_score


def local_refine(points, n_iter=2000, step_init=0.05, step_decay=0.9995):
    """Local refinement using coordinate descent with adaptive steps."""
    points = points.copy()
    step = step_init
    best_points = points.copy()
    best_score = objective(points)
    
    for it in range(n_iter):
        improved = False
        for i in range(N):
            # Try small perturbations in each coordinate
            for d in range(2):
                for sign in [-1, 1]:
                    new_points = points.copy()
                    new_points[i, d] += sign * step
                    new_score = objective(new_points)
                    if new_score < best_score:
                        points = new_points
                        best_score = new_score
                        best_points = points.copy()
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        
        if not improved:
            step *= 0.5
            if step < 1e-6:
                break
        else:
            step *= step_decay
    
    return best_points, best_score


def construct():
    start_time = time.time()
    
    best_points = None
    best_score = float('inf')
    
    # Try multiple initializations
    inits = [
        hexagonal_lattice_init,
        square_grid_init,
    ]
    
    # Add some random inits
    for _ in range(3):
        inits.append(random_init)
    
    for init_func in inits:
        if time.time() - start_time > TIME_LIMIT - 30:
            break
        
        points = init_func()
        
        # Gradient descent
        gd_points, gd_score = gradient_descent_optimize(points, max_iter=3000)
        if gd_score < best_score:
            best_score = gd_score
            best_points = gd_points.copy()
        
        # Annealing from GD result
        if time.time() - start_time > TIME_LIMIT - 30:
            break
        sa_points, sa_score = anneal_optimize(gd_points, max_iter=5000)
        if sa_score < best_score:
            best_score = sa_score
            best_points = sa_points.copy()
        
        # Local refinement
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        lr_points, lr_score = local_refine(sa_points, n_iter=1000)
        if lr_score < best_score:
            best_score = lr_score
            best_points = lr_points.copy()
    
    # Final polish with more iterations if time permits
    if best_points is not None and time.time() - start_time < TIME_LIMIT - 5:
        final_points, final_score = local_refine(best_points, n_iter=2000, step_init=0.01)
        if final_score < best_score:
            best_points = final_points
            best_score = final_score
    
    if best_points is None:
        best_points = hexagonal_lattice_init()
    
    # Ensure distinct points
    # Check for near-duplicates and separate them
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-6:
                best_points[j] += np.random.randn(2) * 0.01
    
    return best_points
