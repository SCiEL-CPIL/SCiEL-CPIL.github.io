"""Pack 300 equal circles in unit circle using vectorized gradient-based optimization."""
import numpy as np
from scipy.spatial.distance import cdist
import time

N = 300


def _score(points):
    """Compute the packing radius (min of inter-circle distances/2 and boundary distances)."""
    if len(points) < 2:
        return 0.0
    # Inter-circle distances
    d = cdist(points, points)
    np.fill_diagonal(d, np.inf)
    min_inter = np.min(d) / 2.0
    # Boundary distances
    norms = np.linalg.norm(points, axis=1)
    min_boundary = np.min(1.0 - norms)
    return min(min_inter, min_boundary)


def _hex_initialization(n_points):
    """Generate hexagonal lattice points inside unit circle, return first n_points."""
    # Try multiple spacings to find one that gives at least n_points
    for spacing in np.linspace(0.03, 0.12, 30):
        # Generate hex lattice
        dx = spacing
        dy = spacing * np.sqrt(3) / 2
        # Range to cover unit circle
        R = 1.0
        nx = int(R / dx) + 3
        ny = int(R / dy) + 3
        
        points = []
        for j in range(-ny, ny + 1):
            for i in range(-nx, nx + 1):
                x = i * dx + (j % 2) * dx / 2
                y = j * dy
                if x * x + y * y <= (1.0 - spacing / 2) ** 2:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) >= n_points:
            # Sort by distance from center, take closest n_points
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists)
            return points[idx[:n_points]]
    
    # Fallback: random
    return np.random.randn(n_points, 2) * 0.5


def _vectorized_gradient_step(points, lr, beta):
    """
    Vectorized gradient computation for soft-min objective.
    Maximizes min(min_pairwise_dist/2, min_boundary_dist).
    """
    n = len(points)
    
    # Compute all pairwise distances
    d = cdist(points, points)
    np.fill_diagonal(d, np.inf)
    
    # Boundary distances
    norms = np.linalg.norm(points, axis=1)
    boundary = 1.0 - norms
    
    # Soft-min weights for inter-circle distances / 2
    inter = d / 2.0
    m_inter = np.min(inter)
    exp_inter = np.exp(-(inter - m_inter) * beta)
    # Mask diagonal
    exp_inter[np.diag_indices(n)] = 0.0
    s_inter = np.sum(exp_inter)
    w_inter = exp_inter / s_inter
    soft_inter = m_inter + np.log(s_inter) / beta
    
    # Soft-min weights for boundary distances
    m_bound = np.min(boundary)
    exp_bound = np.exp(-(boundary - m_bound) * beta)
    s_bound = np.sum(exp_bound)
    w_bound = exp_bound / s_bound
    soft_bound = m_bound + np.log(s_bound) / beta
    
    # Combined soft-min weight
    combined = np.array([soft_inter, soft_bound])
    m_c = np.min(combined)
    exp_c = np.exp(-(combined - m_c) * beta)
    s_c = np.sum(exp_c)
    w_c = exp_c / s_c
    obj = m_c + np.log(s_c) / beta
    
    # Vectorized gradient computation
    grad = np.zeros_like(points)
    
    # Inter-circle gradient: for each i, sum over j of w_inter[i,j] * (p_i - p_j) / (2*d_ij)
    # (p_i - p_j) / d_ij is the unit vector from j to i
    # We can compute this as: for each i, sum_j w_inter[i,j] * (points[i] - points[j]) / d[i,j]
    
    # diff[i,j] = points[i] - points[j]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # shape (n, n, 2)
    
    # unit_dir[i,j] = diff[i,j] / d[i,j]
    # Avoid division by zero on diagonal
    safe_d = np.where(d > 1e-10, d, 1.0)
    unit_dir = diff / safe_d[:, :, np.newaxis]
    unit_dir[np.diag_indices(n)] = 0.0
    
    # grad_inter[i] = sum_j w_inter[i,j] * unit_dir[i,j] * 0.5 * w_c[0]
    grad_inter = np.sum(w_inter[:, :, np.newaxis] * unit_dir, axis=1) * 0.5 * w_c[0]
    
    # Boundary gradient: grad_bound[i] = w_bound[i] * (-points[i] / norms[i]) * w_c[1]
    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    grad_bound = w_bound[:, np.newaxis] * (-points / safe_norms[:, np.newaxis]) * w_c[1]
    
    grad = grad_inter + grad_bound
    
    # Take gradient ascent step
    new_points = points + lr * grad
    
    # Project back: ensure points stay inside circle
    new_norms = np.linalg.norm(new_points, axis=1)
    mask = new_norms > 0.999
    new_points[mask] *= 0.999 / new_norms[mask]
    
    return new_points, obj


def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    # Initialize with hexagonal lattice
    points = _hex_initialization(N)
    best_points = points.copy()
    best_score = _score(best_points)
    
    # Multi-stage optimization with annealing
    stages = [
        # (num_steps, lr, beta)
        (300, 0.005, 50),
        (400, 0.003, 200),
        (500, 0.002, 500),
        (600, 0.001, 1000),
        (800, 0.0005, 3000),
        (1000, 0.0002, 10000),
    ]
    
    for step_count, lr, beta in stages:
        for step in range(step_count):
            if time.time() > deadline - 15:
                break
            
            points, obj = _vectorized_gradient_step(points, lr, beta)
            
            # Check if improved
            curr_score = _score(points)
            if curr_score > best_score:
                best_score = curr_score
                best_points = points.copy()
        
        if time.time() > deadline - 15:
            break
    
    # Final polish: use exact local optimization with very small steps
    if time.time() < deadline - 5:
        polish_steps = 0
        while time.time() < deadline - 3 and polish_steps < 5000:
            # Try small random perturbations and keep if better
            trial = best_points + np.random.randn(*best_points.shape) * 0.001
            trial_norms = np.linalg.norm(trial, axis=1)
            mask = trial_norms > 0.999
            trial[mask] *= 0.999 / trial_norms[mask]
            trial_score = _score(trial)
            if trial_score > best_score:
                best_score = trial_score
                best_points = trial.copy()
            polish_steps += 1
    
    return best_points
