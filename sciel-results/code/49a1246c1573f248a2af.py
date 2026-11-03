import numpy as np

N = 300

def construct():
    # Initial hexagonal grid
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    k = int(1.0 / spacing) + 2
    rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
            for i in range(-k, k + 1) for j in range(-k, k + 1)]
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    points = points[radius <= 1.0 - spacing / 2.0]
    
    # Select N points closest to center
    if len(points) > N:
        points = points[np.argsort(radius, kind="stable")[:N]]
    elif len(points) < N:
        # Fallback: random points
        points = np.random.rand(N, 2) * 2 - 1
        mask = np.linalg.norm(points, axis=1) <= 1.0
        while np.sum(mask) < N:
            new_points = np.random.rand(1, 2) * 2 - 1
            if np.linalg.norm(new_points[0]) <= 1.0:
                points = np.vstack([points, new_points])
                mask = np.linalg.norm(points, axis=1) <= 1.0
        points = points[:N]

    points = points.astype(np.float64)
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    # Gradient Descent on soft-min
    alpha = 20.0
    step_size = 0.05
    num_iterations = 3000
    
    for it in range(num_iterations):
        diff = points[i_idx] - points[j_idx]
        d_pair = np.linalg.norm(diff, axis=1)
        d_pair[d_pair < 1e-10] = 1e-10
        
        r_bound = np.linalg.norm(points, axis=1)
        d_bound = 1.0 - r_bound
        d_bound[d_bound < 1e-10] = 1e-10
        
        d_all = np.concatenate([d_pair, d_bound])
        weights = np.exp(-alpha * d_all)
        Z = np.sum(weights)
        
        grad = np.zeros_like(points)
        
        w_pair = weights[:len(i_idx)] / Z
        v_norm = diff / d_pair[:, np.newaxis]
        
        np.add.at(grad, i_idx, w_pair[:, np.newaxis] * v_norm)
        np.add.at(grad, j_idx, -w_pair[:, np.newaxis] * v_norm)
        
        w_bound = weights[len(i_idx):] / Z
        r_bound_safe = r_bound.copy()
        r_bound_safe[r_bound_safe < 1e-10] = 1e-10
        grad_bound = -points / r_bound_safe[:, np.newaxis]
        grad += w_bound[:, np.newaxis] * grad_bound
        
        points -= step_size * grad
        
        r_new = np.linalg.norm(points, axis=1)
        mask = r_new > 1.0
        if np.any(mask):
            points[mask] /= r_new[mask, np.newaxis]
            
        if it % 50 == 0 and it > 0:
            alpha *= 0.95
            step_size *= 0.98

    # Polish
    alpha = 100.0
    step_size = 0.01
    for _ in range(500):
        diff = points[i_idx] - points[j_idx]
        d_pair = np.linalg.norm(diff, axis=1)
        d_pair[d_pair < 1e-10] = 1e-10
        
        r_bound = np.linalg.norm(points, axis=1)
        d_bound = 1.0 - r_bound
        d_bound[d_bound < 1e-10] = 1e-10
        
        d_all = np.concatenate([d_pair, d_bound])
        weights = np.exp(-alpha * d_all)
        Z = np.sum(weights)
        
        grad = np.zeros_like(points)
        
        w_pair = weights[:len(i_idx)] / Z
        v_norm = diff / d_pair[:, np.newaxis]
        
        np.add.at(grad, i_idx, w_pair[:, np.newaxis] * v_norm)
        np.add.at(grad, j_idx, -w_pair[:, np.newaxis] * v_norm)
        
        w_bound = weights[len(i_idx):] / Z
        r_bound_safe = r_bound.copy()
        r_bound_safe[r_bound_safe < 1e-10] = 1e-10
        grad_bound = -points / r_bound_safe[:, np.newaxis]
        grad += w_bound[:, np.newaxis] * grad_bound
        
        points -= step_size * grad
        
        r_new = np.linalg.norm(points, axis=1)
        mask = r_new > 1.0
        if np.any(mask):
            points[mask] /= r_new[mask, np.newaxis]

    return points
