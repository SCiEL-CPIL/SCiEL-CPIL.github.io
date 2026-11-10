# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist
import time


def _score(points):
    """Compute R = (d_min / d_max)^2."""
    d = pdist(points)
    d_min = d.min()
    d_max = d.max()
    if d_max < 1e-12:
        return 0.0
    if d_min < 1e-12:
        return 0.0
    return (d_min / d_max) ** 2


def _generate_initial_points(seed=0):
    """Generate multiple structured initial configurations."""
    n = 16
    rng = np.random.RandomState(seed)
    candidates = []

    # 1. Regular polygon + center (8-gon + inner 8-gon scaled)
    angles_outer = 2 * np.pi * np.arange(8) / 8
    outer = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    angles_inner = 2 * np.pi * np.arange(8) / 8 + np.pi / 8
    inner = 0.5 * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    candidates.append(np.vstack([outer, inner]))

    # 2. 4x4 grid
    xs = np.linspace(-1, 1, 4)
    ys = np.linspace(-1, 1, 4)
    grid = np.column_stack(np.meshgrid(xs, ys, indexing='ij')).reshape(-1, 2)
    candidates.append(grid)

    # 3. Two concentric 8-gons
    angles1 = 2 * np.pi * np.arange(8) / 8
    angles2 = 2 * np.pi * np.arange(8) / 8 + np.pi / 8
    r1 = 1.0
    r2 = 0.6
    p1 = r1 * np.column_stack([np.cos(angles1), np.sin(angles1)])
    p2 = r2 * np.column_stack([np.cos(angles2), np.sin(angles2)])
    candidates.append(np.vstack([p1, p2]))

    # 4. Hexagonal-ish arrangement: 1 center + 6 ring + 9 outer
    center = np.array([[0.0, 0.0]])
    ring1_angles = 2 * np.pi * np.arange(6) / 6
    ring1 = 0.8 * np.column_stack([np.cos(ring1_angles), np.sin(ring1_angles)])
    # 9 points on outer ring at radius 1.5
    ring2_angles = 2 * np.pi * np.arange(9) / 9
    ring2 = 1.5 * np.column_stack([np.cos(ring2_angles), np.sin(ring2_angles)])
    candidates.append(np.vstack([center, ring1, ring2]))

    # 5. Random
    candidates.append(rng.randn(n, 2) * 0.5)

    # 6. Another structured: 2x8 rectangle with jitter
    xs = np.tile(np.array([-1, 1]), 8)
    ys = np.linspace(-1, 1, 16)
    rect = np.column_stack([xs, ys])
    candidates.append(rect + rng.randn(16, 2) * 0.05)

    return candidates


def _simulated_annealing(points, time_budget=55.0, seed=42):
    """Run simulated annealing to maximize min/max distance ratio."""
    n = 16
    rng = np.random.RandomState(seed)
    
    current = points.copy()
    current_score = _score(current)
    best = current.copy()
    best_score = current_score
    
    start_time = time.time()
    T = 1.0
    T_min = 1e-8
    cooling_rate = 0.9995
    
    # Precompute pair indices for faster distance computation
    idx_i, idx_j = np.triu_indices(n, k=1)
    
    while time.time() - start_time < time_budget:
        # Generate perturbation
        if rng.random() < 0.7:
            # Move one or two points
            k = rng.randint(1, 3)
            indices = rng.choice(n, size=k, replace=False)
            delta = rng.randn(k, 2) * T * 0.3
            new_points = current.copy()
            new_points[indices] += delta
        else:
            # Swap two points' positions with slight jitter
            i, j = rng.choice(n, size=2, replace=False)
            new_points = current.copy()
            temp = new_points[i].copy()
            new_points[i] = new_points[j] + rng.randn(2) * T * 0.1
            new_points[j] = temp + rng.randn(2) * T * 0.1
        
        new_score = _score(new_points)
        
        # Accept if better or with probability exp((new_score - current_score) / T)
        if new_score >= current_score:
            current = new_points
            current_score = new_score
            if current_score > best_score:
                best = current.copy()
                best_score = current_score
        else:
            # Avoid division by zero or log of negative
            delta_score = new_score - current_score
            if delta_score > -1e6:
                prob = np.exp(delta_score / (T + 1e-12))
                if rng.random() < prob:
                    current = new_points
                    current_score = new_score
        
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    return best, best_score


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.
    """
    n = 16
    deadline = time.time() + 580.0  # Safety margin before 630s kill
    
    # Generate initial candidates
    initial_points = _generate_initial_points(seed=42)
    
    best_points = initial_points[0].copy()
    best_score = _score(best_points)
    
    # Run SA from each initial configuration
    for i, init_pts in enumerate(initial_points):
        if time.time() > deadline - 60.0:
            break
        
        time_budget = min(50.0, deadline - time.time() - 10.0)
        if time_budget < 5.0:
            break
        
        result_pts, result_score = _simulated_annealing(
            init_pts, time_budget=time_budget, seed=42 + i * 100
        )
        
        if result_score > best_score:
            best_points = result_pts
            best_score = result_score
    
    # Final refinement: run a short SA from the best point found
    remaining = deadline - time.time() - 5.0
    if remaining > 10.0:
        result_pts, result_score = _simulated_annealing(
            best_points, time_budget=min(remaining, 40.0), seed=999
        )
        if result_score > best_score:
            best_points = result_pts
            best_score = result_score
    
    return best_points


# EVOLVE-BLOCK-END
