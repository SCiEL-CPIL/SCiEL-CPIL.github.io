import numpy as np

N = 100

def construct():
    # 1. Initialization: Jittered Grid to break symmetry
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.375, 0.375, nz)
    
    centers = np.array([(x, y, z) for x in xs for y in ys for z in zs])
    
    # Add random jitter to break symmetry and help escape local minima
    np.random.seed(42)
    jitter = np.random.uniform(-0.02, 0.02, centers.shape)
    centers = np.clip(centers + jitter, -0.5, 0.5)

    # 2. Optimization: Simulated Annealing with Squared Distance Penalty
    
    # Fast penalty function using squared distances
    def calc_penalty_sq(c, r_target):
        # Boundary violations
        abs_c = np.abs(c)
        boundary_dist = 0.5 - abs_c
        boundary_viol = np.maximum(0, r_target - boundary_dist)
        penalty = np.sum(boundary_viol**2)
        
        # Pairwise overlaps
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d_sq = np.sum(diff**2, axis=2)
        target_d_sq = (2 * r_target)**2
        
        # Set diagonal to inf so it doesn't contribute to penalty
        np.fill_diagonal(d_sq, np.inf)
        
        violations = np.maximum(0, target_d_sq - d_sq)
        penalty += np.sum(violations**2)
        
        return penalty

    def calc_radius(c):
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        max_abs_coord = np.max(np.abs(c), axis=1)
        min_boundary = 0.5 - np.max(max_abs_coord)
        return min(min_pair / 2.0, min_boundary)

    r = 0.10
    dr = 0.0005
    max_r = 0.115
    
    T = 0.005
    T_min = 1e-6
    cooling = 0.99
    max_iters_per_r = 2000
    move_scale = 0.05
    
    while r < max_r:
        current_c = centers.copy()
        current_penalty = calc_penalty_sq(current_c, r)
        
        # If penalty is already low, we can skip SA or reduce iters
        if current_penalty < 1e-6:
            r += dr
            continue
            
        for _ in range(max_iters_per_r):
            if T < T_min:
                break
                
            i = np.random.randint(N)
            move_mag = move_scale * (0.5 + T)
            delta = np.random.uniform(-move_mag, move_mag, 3)
            
            new_c = current_c.copy()
            new_c[i] = np.clip(current_c[i] + delta, -0.5, 0.5)
            
            new_penalty = calc_penalty_sq(new_c, r)
            delta_e = new_penalty - current_penalty
            
            if delta_e < 0 or np.random.random() < np.exp(-delta_e / T):
                current_c = new_c
                current_penalty = new_penalty
                
            T *= cooling
            
        # Check success
        if current_penalty < 1e-5:
            centers = current_c
            r += dr
        else:
            # Failed
            break

    # Final Polish: Hill Climbing
    def hc_optimize(c, steps=5000):
        c = c.copy()
        r_best = calc_radius(c)
        
        for _ in range(steps):
            i = np.random.randint(N)
            delta = np.random.normal(0, 0.005, 3)
            new_c = c.copy()
            new_c[i] = np.clip(c[i] + delta, -0.5, 0.5)
            
            new_r = calc_radius(new_c)
            if new_r > r_best:
                c = new_c
                r_best = new_r
                
        return c, r_best

    centers, _ = hc_optimize(centers)
    
    return centers
