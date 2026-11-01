import numpy as np
import time

N = 200

def construct():
    start_time = time.time()
    deadline = 110.0  # Leave 10s margin
    
    # 1. Hexagonal Grid Initialization
    # Target radius approx 0.0366
    # Hex lattice spacing s = 2*r
    # Let's start with a slightly smaller r to ensure points fit, then expand
    r_init = 0.035
    s = 2 * r_init
    
    # Generate hex grid points
    points = []
    dx = s
    dy = s * np.sqrt(3) / 2
    
    # We need to cover [-0.5, 0.5] x [-0.5, 0.5]
    # x range: -0.5 to 0.5
    # y range: -0.5 to 0.5
    
    x_start = -0.5
    y_start = -0.5
    
    # Number of columns
    num_cols = int(np.ceil((1.0) / dx)) + 1
    # Number of rows
    num_rows = int(np.ceil((1.0) / dy)) + 1
    
    for i in range(num_cols):
        x = x_start + i * dx
        for j in range(num_rows):
            y = y_start + j * dy
            if i % 2 != 0:
                y += dy / 2
            # Check if inside square with margin
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points.append([x, y])
                
    points = np.array(points)
    
    # If we have more than N, randomly select N
    if len(points) > N:
        indices = np.random.choice(len(points), N, replace=False)
        points = points[indices]
    elif len(points) < N:
        # Add random points
        while len(points) < N:
            p = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)])
            points = np.vstack([points, p])
            
    X = points.copy()
    
    # Helper to calculate max radius
    def calc_radius(X):
        # Pairwise distances
        # X is (N, 2)
        # Diff is (N, N, 2)
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        # Set diagonal to inf
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Wall distances
        # dist to x=-0.5 is X[:,0] + 0.5
        # dist to x=0.5 is 0.5 - X[:,0]
        # dist to y=-0.5 is X[:,1] + 0.5
        # dist to y=0.5 is 0.5 - X[:,1]
        d_left = X[:, 0] + 0.5
        d_right = 0.5 - X[:, 0]
        d_bottom = X[:, 1] + 0.5
        d_top = 0.5 - X[:, 1]
        min_wall_dist = np.min(np.vstack([d_left, d_right, d_bottom, d_top]))
        
        r = min(min_pair_dist / 2.0, min_wall_dist)
        return r

    current_r = calc_radius(X)
    best_r = current_r
    best_X = X.copy()
    
    # Simulated Annealing
    T = 0.01
    T_min = 1e-6
    cooling = 0.999
    
    iterations = 0
    while time.time() - start_time < deadline:
        # Pick random point
        i = np.random.randint(N)
        
        # Random move
        step_size = 0.01 * T ** 0.5 # Adaptive step?
        # Let's use fixed step size scaled by T
        step = 0.01 * np.sqrt(T)
        dx = np.random.normal(0, step)
        dy = np.random.normal(0, step)
        
        new_X = X.copy()
        new_X[i, 0] += dx
        new_X[i, 1] += dy
        
        # Check bounds
        if new_X[i, 0] < -0.5 or new_X[i, 0] > 0.5 or new_X[i, 1] < -0.5 or new_X[i, 1] > 0.5:
            # Reflect or reject? Rejecting is simpler and keeps points inside.
            # But if we reject too often, we waste time.
            # Let's clamp to boundary for now, but penalty will handle it if we used soft constraint.
            # Here we have hard constraint via rejection.
            # Actually, if we clamp, the point might get stuck at the wall.
            # Let's just reject if out of bounds.
            continue
        
        new_r = calc_radius(new_X)
        
        # Acceptance
        if new_r > current_r:
            X = new_X
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_X = X.copy()
        else:
            # Accept with probability exp((new_r - current_r) / T)
            # Since r is small, delta is small.
            delta = new_r - current_r
            prob = np.exp(delta / T)
            if np.random.rand() < prob:
                X = new_X
                current_r = new_r
        
        T *= cooling
        if T < T_min:
            T = T_min
            
        iterations += 1
        
        # Restart if stuck?
        # If no improvement for a long time, maybe perturb best_X
        # For now, keep it simple.
        
    return best_X
