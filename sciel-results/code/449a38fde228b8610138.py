import numpy as np

N = 50

def compute_radius(centers):
    # Vectorized distance calculation
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    
    # Wall distances
    # min_i min_k (0.5 - |c_ik|) is equivalent to 0.5 - max_{i,k} |c_ik|
    max_abs_c = np.max(np.abs(centers))
    min_wall_dist = 0.5 - max_abs_c
    
    return min(min_pair_dist / 2.0, min_wall_dist)

def construct():
    best_centers = None
    best_radius = 0.0
    
    n_restarts = 10
    max_iters = 10000
    
    for seed in range(n_restarts):
        np.random.seed(seed)
        
        if seed < 5:
            # Grid initialization
            n_per_dim = 4 # 4^3 = 64 points, take 50
            grid_points = []
            for i in range(n_per_dim):
                for j in range(n_per_dim):
                    for k in range(n_per_dim):
                        grid_points.append([
                            -0.4 + i * 0.8 / (n_per_dim - 1),
                            -0.4 + j * 0.8 / (n_per_dim - 1),
                            -0.4 + k * 0.8 / (n_per_dim - 1)
                        ])
            grid_points = np.array(grid_points[:N])
            centers = grid_points + np.random.uniform(-0.02, 0.02, (N, 3))
        else:
            centers = np.random.uniform(-0.45, 0.45, (N, 3))
            
        centers = np.clip(centers, -0.5, 0.5)
        current_radius = compute_radius(centers)
        
        T = 0.005
        T_min = 1e-7
        cooling_rate = 0.9995
        step_size = 0.02
        
        for i in range(max_iters):
            idx = np.random.randint(N)
            move = np.random.normal(0, step_size, 3)
            
            new_centers = centers.copy()
            new_centers[idx] = centers[idx] + move
            new_centers[idx] = np.clip(new_centers[idx], -0.5, 0.5)
            
            new_radius = compute_radius(new_centers)
            
            if new_radius >= current_radius:
                centers = new_centers
                current_radius = new_radius
            else:
                delta = new_radius - current_radius
                # Avoid division by zero if T is very small
                if T > T_min:
                    prob = np.exp(delta / T)
                    if np.random.random() < prob:
                        centers = new_centers
                        current_radius = new_radius
            
            T *= cooling_rate
            
            if current_radius > best_radius:
                best_radius = current_radius
                best_centers = centers.copy()
                
    return best_centers
