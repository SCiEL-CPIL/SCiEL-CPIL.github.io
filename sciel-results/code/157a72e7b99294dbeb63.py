import numpy as np

N = 26

# EVOLVE-BLOCK-START
def construct():
    """
    Packs N circles in the unit square to maximize the sum of radii.
    Uses a combination of greedy radius relaxation and simulated annealing on positions.
    """
    # Parameters
    n_iter = 20000
    T_start = 1.0
    T_end = 0.001
    alpha = (T_end / T_start) ** (1.0 / n_iter)
    
    # Initialization: Grid layout to start with a valid, non-overlapping configuration
    side = int(np.ceil(np.sqrt(N)))
    xs = np.zeros(N)
    ys = np.zeros(N)
    rs = np.zeros(N)
    
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N:
                break
            xs[idx] = (i + 0.5) / side
            ys[idx] = (j + 0.5) / side
            # Initial radius slightly less than half cell size to ensure no overlap
            rs[idx] = 0.5 / side - 0.01
            idx += 1
            
    # Helper function to calculate the max radius for circle i given positions and other radii
    def max_radius(i, xs, ys, rs):
        # Boundary constraints
        r_max = min(xs[i], 1.0 - xs[i], ys[i], 1.0 - ys[i])
        if r_max < 1e-6:
            return 1e-6 # Prevent negative or zero if stuck on edge
        
        # Overlap constraints with other circles
        for j in range(N):
            if i == j:
                continue
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            dist = np.sqrt(dx*dx + dy*dy)
            # To not overlap: r_i + r_j <= dist => r_i <= dist - r_j
            r_candidate = dist - rs[j]
            if r_candidate < r_max:
                r_max = r_candidate
        
        if r_max < 1e-6:
            return 1e-6
        return r_max

    # Function to relax all radii greedily
    def relax_radii(xs, ys, rs):
        new_rs = np.copy(rs)
        # Iterate a few times to allow propagation of changes
        for _ in range(3):
            for i in range(N):
                new_rs[i] = max_radius(i, xs, ys, new_rs)
        return new_rs

    # Initial relaxation
    rs = relax_radii(xs, ys, rs)
    current_sum = np.sum(rs)
    
    # Simulated Annealing on positions
    T = T_start
    best_sum = current_sum
    best_xs = np.copy(xs)
    best_ys = np.copy(ys)
    best_rs = np.copy(rs)
    
    for step in range(n_iter):
        # Pick a random circle to move
        i = np.random.randint(N)
        
        # Propose a move (Gaussian perturbation)
        step_size = 0.05 * (1.0 + T) # Scale step with temperature
        dx = np.random.normal(0, step_size)
        dy = np.random.normal(0, step_size)
        
        new_xs = np.copy(xs)
        new_ys = np.copy(ys)
        new_xs[i] += dx
        new_ys[i] += dy
        
        # Keep within bounds for the center (radius will be handled by relaxation)
        # We allow centers to go slightly out, but relaxation will clamp radius to 0 if too far out.
        # However, to keep it sane, let's keep centers within [0, 1]
        new_xs[i] = np.clip(new_xs[i], 0.0, 1.0)
        new_ys[i] = np.clip(new_ys[i], 0.0, 1.0)
        
        # Relax radii for the new configuration
        # We only need to update radius of i? No, moving i might allow j to grow, etc.
        # For speed, we can just update i's radius and see if sum improves.
        # But global relaxation is more robust. Since N=26, it's fast enough.
        new_rs = relax_radii(new_xs, new_ys, rs)
        
        new_sum = np.sum(new_rs)
        delta = new_sum - current_sum
        
        # Acceptance criterion
        if delta > 0 or np.random.random() < np.exp(delta / T):
            xs = new_xs
            ys = new_ys
            rs = new_rs
            current_sum = new_sum
            
            if current_sum > best_sum:
                best_sum = current_sum
                best_xs = np.copy(xs)
                best_ys = np.copy(ys)
                best_rs = np.copy(rs)
        
        # Cool down
        T *= alpha
        
    # Final relaxation to ensure maximum size for the best found positions
    best_rs = relax_radii(best_xs, best_ys, best_rs)
    
    # Construct result array
    result = np.column_stack((best_xs, best_ys, best_rs))
    
    # Final safety check: ensure no overlaps and inside bounds
    # If any overlap exists due to floating point errors, shrink radii slightly
    valid = True
    for _ in range(10):
        valid = True
        for i in range(N):
            x, y, r = result[i]
            if x < r or x > 1-r or y < r or y > 1-r:
                # Shrink radius to fit boundary
                new_r = min(x, 1-x, y, 1-y)
                if new_r < r:
                    result[i, 2] = max(0, new_r - 1e-6)
                    valid = False
        for i in range(N):
            for j in range(i+1, N):
                x1, y1, r1 = result[i]
                x2, y2, r2 = result[j]
                d = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                if d < r1 + r2:
                    # Shrink the larger radius
                    if r1 > r2:
                        result[i, 2] = max(0, d - r2 - 1e-6)
                    else:
                        result[j, 2] = max(0, d - r1 - 1e-6)
                    valid = False
        if valid:
            break
            
    return result
# EVOLVE-BLOCK-END
