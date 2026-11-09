import numpy as np
import time

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    def get_radius(c):
        bound_dists = np.minimum(0.5 - np.abs(c[:, 0]), 0.5 - np.abs(c[:, 1]))
        min_bound = np.min(bound_dists)
        
        diffs = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists) / 2.0
        
        return np.min([min_bound, min_pair])

    # Initialization: Hexagonal Grid
    s = 0.08
    points = []
    y = -0.5
    row_idx = 0
    while y <= 0.5:
        x_offset = (s / 2) if row_idx % 2 == 1 else 0.0
        x = -0.5 + x_offset
        while x <= 0.5:
            points.append((x, y))
            x += s
        y += s * np.sqrt(3) / 2
        row_idx += 1
        
    points = np.array(points)
    mask = (points[:, 0] >= -0.5) & (points[:, 0] <= 0.5) & (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
    points = points[mask]
    
    if len(points) > N:
        dists_to_center = np.sum(points**2, axis=1)
        indices = np.argsort(dists_to_center)[:N]
        centers = points[indices].copy()
    elif len(points) < N:
        random_points = np.random.uniform(-0.5, 0.5, size=(N - len(points), 2))
        centers = np.vstack([points, random_points])
    else:
        centers = points.copy()

    centers = np.clip(centers, -0.5, 0.5)

    current_radius = get_radius(centers)
    best_radius = current_radius
    best_centers = centers.copy()
    
    # Multi-restart SA
    max_restarts = 5
    restarts_used = 0
    no_improve_steps = 0
    max_no_improve = 5000
    
    T = 0.05
    step_size = 0.05
    
    while time.time() < deadline:
        idx = np.random.randint(N)
        delta = np.random.normal(0, step_size, size=2)
        new_centers = centers.copy()
        new_centers[idx] += delta
        new_centers[idx] = np.clip(new_centers[idx], -0.5, 0.5)
        
        new_radius = get_radius(new_centers)
        
        if new_radius > current_radius:
            accept = True
        else:
            delta_r = new_radius - current_radius
            if T > 1e-6:
                prob = np.exp(delta_r / T)
            else:
                prob = 0.0
            accept = np.random.random() < prob
            
        if accept:
            centers = new_centers
            current_radius = new_radius
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = centers.copy()
                no_improve_steps = 0
            else:
                no_improve_steps += 1
        else:
            no_improve_steps += 1
            
        # Cooling
        T *= 0.999
        if T < 0.001:
            T = 0.001
            
        # Adaptive step size
        step_size = 0.01 + 0.04 * (T / 0.05)
        
        # Restart logic
        if no_improve_steps > max_no_improve and restarts_used < max_restarts:
            restarts_used += 1
            # Restart from best solution with perturbation
            centers = best_centers + np.random.normal(0, 0.02, size=(N, 2))
            centers = np.clip(centers, -0.5, 0.5)
            current_radius = get_radius(centers)
            T = 0.05
            step_size = 0.05
            no_improve_steps = 0

    return best_centers
