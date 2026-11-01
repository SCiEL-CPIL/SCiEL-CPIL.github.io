"""Improved 50-sphere packing: smooth surrogate + repulsion annealing + exact polish."""
import numpy as np
import time

N = 50

def _compute_radius(centers):
    """Exact radius: min(min_pair_dist/2, min_wall_dist)."""
    # Wall distances
    wall_dists = np.min(np.stack([
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1]),
        0.5 - np.abs(centers[:, 2])
    ], axis=1), axis=1)
    r_wall = np.min(wall_dists)
    
    # Pair distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    r_pair = min_pair_dist / 2.0
    
    return min(r_pair, r_wall)

def _soft_min_radius(centers, beta):
    """
    Smooth surrogate: soft-min of all constraints.
    Constraints: for each pair (i,j), d_ij - 2r >= 0, and for each sphere i, wall_dist_i - r >= 0.
    We maximize r subject to all constraints >= 0.
    Equivalently, minimize -r + penalty for violations.
    
    Here we use a different approach: 
    r_soft = soft_min of all (d_ij/2) and (wall_dist_i), using log-sum-exp.
    """
    # Compute all pair distances / 2
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_vals = dists / 2.0  # shape (N, N), diagonal is inf
    
    # Wall distances
    wall_dists = np.min(np.stack([
        0.5 - np.abs(centers[:, 0]),
        0.5 - np.abs(centers[:, 1]),
        0.5 - np.abs(centers[:, 2])
    ], axis=1), axis=1)  # shape (N,)
    
    # Combine all values into one vector
    pair_flat = pair_vals.ravel()
    # Filter out inf
    mask = np.isfinite(pair_flat)
    pair_flat = pair_flat[mask]
    
    all_vals = np.concatenate([pair_flat, wall_dists])
    
    # Soft min via log-sum-exp (with negative to get min)
    # soft_min(x) = -log(sum(exp(-beta * x))) / beta
    # For numerical stability, subtract max
    neg_beta_x = -beta * all_vals
    max_val = np.max(neg_beta_x)
    log_sum = max_val + np.log(np.sum(np.exp(neg_beta_x - max_val)))
    soft_min_val = -log_sum / beta
    
    return soft_min_val

def _init_grid():
    """4x4x4 grid, take first 50."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return grid[:N].copy()

def _init_fcc():
    """FCC-like initialization."""
    # FCC unit cell has 4 atoms. Scale to fit.
    # In a cube of side 1, FCC with 4 atoms per cell.
    # Let's generate a 3x3x3 array of FCC cells, take 50 points.
    # FCC basis: (0,0,0), (0.5,0.5,0), (0.5,0,0.5), (0,0.5,0.5) in cell coords [0,1]
    basis = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [0.0, 0.5, 0.5]
    ])
    
    # Generate 3x3x3 cells
    cell_size = 1.0 / 3.0
    points = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                origin = np.array([i, j, k]) * cell_size - 0.5
                for b in basis:
                    pt = origin + b * cell_size
                    if -0.5 <= pt[0] <= 0.5 and -0.5 <= pt[1] <= 0.5 and -0.5 <= pt[2] <= 0.5:
                        points.append(pt)
    
    points = np.array(points)
    if len(points) >= N:
        return points[:N].copy()
    else:
        # Pad with random
        extra = np.random.uniform(-0.5, 0.5, (N - len(points), 3))
        return np.vstack([points, extra])

def _init_bcc():
    """BCC-like initialization."""
    # BCC: (0,0,0) and (0.5,0.5,0.5) in cell coords
    basis = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5]
    ])
    
    cell_size = 1.0 / 3.0
    points = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                origin = np.array([i, j, k]) * cell_size - 0.5
                for b in basis:
                    pt = origin + b * cell_size
                    if -0.5 <= pt[0] <= 0.5 and -0.5 <= pt[1] <= 0.5 and -0.5 <= pt[2] <= 0.5:
                        points.append(pt)
    
    points = np.array(points)
    if len(points) >= N:
        return points[:N].copy()
    else:
        extra = np.random.uniform(-0.5, 0.5, (N - len(points), 3))
        return np.vstack([points, extra])

def _init_random():
    """Random initialization."""
    return np.random.uniform(-0.45, 0.45, (N, 3))

def _repulsion_anneal(centers, time_limit, target_r=None):
    """
    Simulated annealing with repulsion forces to escape local minima.
    Accepts temporary overlaps to explore the landscape.
    """
    start_time = time.time()
    centers = centers.copy()
    
    # Initial temperature
    T = 0.01
    T_min = 0.0001
    cooling = 0.995
    
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()
    
    # If target_r is given, we optimize feasibility
    if target_r is None:
        target_r = best_radius
    
    while time.time() - start_time < time_limit:
        # Pick a random sphere
        i = np.random.randint(N)
        c_i = centers[i].copy()
        
        # Compute repulsion force from all other spheres
        dists = np.sqrt(np.sum((centers - c_i)**2, axis=1))
        dists[i] = np.inf
        
        # Wall distances
        d_wall_x = min(0.5 - c_i[0], c_i[0] + 0.5)
        d_wall_y = min(0.5 - c_i[1], c_i[1] + 0.5)
        d_wall_z = min(0.5 - c_i[2], c_i[2] + 0.5)
        
        # Force from spheres: push away from closest
        force = np.zeros(3)
        min_d = np.min(dists)
        argmin_d = np.argmin(dists)
        
        # If closest sphere is too close, push away
        if min_d < 2 * target_r:
            diff = c_i - centers[argmin_d]
            norm = np.linalg.norm(diff)
            if norm > 1e-9:
                # Strength proportional to overlap
                overlap = 2 * target_r - min_d
                force += (diff / norm) * overlap
        
        # Wall forces
        if d_wall_x < target_r:
            force[0] += -np.sign(c_i[0]) * (target_r - d_wall_x)
        if d_wall_y < target_r:
            force[1] += -np.sign(c_i[1]) * (target_r - d_wall_y)
        if d_wall_z < target_r:
            force[2] += -np.sign(c_i[2]) * (target_r - d_wall_z)
        
        # Normalize and scale by temperature
        norm_f = np.linalg.norm(force)
        if norm_f > 1e-9:
            force = force / norm_f * T
        
        # Add random noise
        noise = np.random.normal(0, T * 0.5, 3)
        displacement = force + noise
        
        new_c = c_i + displacement
        new_c = np.clip(new_c, -0.499, 0.499)
        
        # Compute energy change (radius change)
        old_radius = _compute_radius(centers)
        centers[i] = new_c
        new_radius = _compute_radius(centers)
        
        # Accept if better, or with probability exp(-delta/T)
        delta = new_radius - old_radius
        if delta > 0 or np.random.random() < np.exp(delta / max(T, 1e-10)):
            pass  # Accept
        else:
            centers[i] = c_i  # Reject
        
        # Update best
        current_radius = _compute_radius(centers)
        if current_radius > best_radius:
            best_radius = current_radius
            best_centers = centers.copy()
        
        # Cool down
        T = max(T * cooling, T_min)
        
        # Occasionally reheat to escape deep local minima
        if T < 0.001 and np.random.random() < 0.01:
            T = 0.005
    
    return best_centers, best_radius

def _refine_smooth(centers, time_limit, beta_start=10.0, beta_end=100.0):
    """
    Gradient-based refinement using smooth surrogate.
    Uses scipy.optimize.minimize with L-BFGS-B.
    """
    from scipy.optimize import minimize
    
    start_time = time.time()
    centers = centers.copy().flatten()
    
    def objective(x):
        c = x.reshape(N, 3)
        # Clip to valid range
        c = np.clip(c, -0.499, 0.499)
        # Maximize soft_min_radius, so minimize negative
        return -_soft_min_radius(c, beta_start)
    
    def gradient(x):
        # Numerical gradient
        eps = 1e-7
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            grad[i] = (objective(x_plus) - objective(x_minus)) / (2 * eps)
        return grad
    
    # Run L-BFGS-B
    result = minimize(
        objective,
        centers,
        method='L-BFGS-B',
        jac=gradient,
        options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    best_centers = np.clip(result.x.reshape(N, 3), -0.499, 0.499)
    best_radius = _compute_radius(best_centers)
    
    # If time remains, do a second pass with higher beta
    if time.time() - start_time < time_limit * 0.5:
        beta = beta_end
        def objective2(x):
            c = x.reshape(N, 3)
            c = np.clip(c, -0.499, 0.499)
            return -_soft_min_radius(c, beta)
        
        def gradient2(x):
            eps = 1e-7
            grad = np.zeros_like(x)
            for i in range(len(x)):
                x_plus = x.copy()
                x_plus[i] += eps
                x_minus = x.copy()
                x_minus[i] -= eps
                grad[i] = (objective2(x_plus) - objective2(x_minus)) / (2 * eps)
            return grad
        
        result2 = minimize(
            objective2,
            best_centers.flatten(),
            method='L-BFGS-B',
            jac=gradient2,
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        best_centers = np.clip(result2.x.reshape(N, 3), -0.499, 0.499)
        best_radius = _compute_radius(best_centers)
    
    return best_centers, best_radius

def _refine_exact(centers, time_limit):
    """
    Exact local refinement: coordinate descent on the true nonsmooth objective.
    For each sphere, try to move it to increase the minimum radius.
    """
    start_time = time.time()
    centers = centers.copy()
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()
    
    step = 0.01
    decay = 0.999
    min_step = 0.00001
    
    while time.time() - start_time < time_limit:
        improved = False
        for i in range(N):
            if time.time() - start_time > time_limit:
                break
            
            c_i = centers[i]
            
            # Compute current min constraint for sphere i
            dists = np.sqrt(np.sum((centers - c_i)**2, axis=1))
            dists[i] = np.inf
            min_d_pair = np.min(dists)
            
            d_wall_x = min(0.5 - c_i[0], c_i[0] + 0.5)
            d_wall_y = min(0.5 - c_i[1], c_i[1] + 0.5)
            d_wall_z = min(0.5 - c_i[2], c_i[2] + 0.5)
            min_d_wall = min(d_wall_x, d_wall_y, d_wall_z)
            
            old_local_min = min(min_d_pair / 2.0, min_d_wall)
            
            # Try moving in 6 directions (±x, ±y, ±z) and 4 diagonal directions
            directions = [
                np.array([1, 0, 0]), np.array([-1, 0, 0]),
                np.array([0, 1, 0]), np.array([0, -1, 0]),
                np.array([0, 0, 1]), np.array([0, 0, -1]),
                np.array([1, 1, 0])/np.sqrt(2), np.array([-1, -1, 0])/np.sqrt(2),
                np.array([1, 0, 1])/np.sqrt(2), np.array([-1, 0, -1])/np.sqrt(2),
                np.array([0, 1, 1])/np.sqrt(2), np.array([0, -1, -1])/np.sqrt(2),
            ]
            
            best_move = None
            best_local_min = old_local_min
            
            for d in directions:
                new_c = c_i + step * d
                new_c = np.clip(new_c, -0.499, 0.499)
                
                new_dists = np.sqrt(np.sum((centers - new_c)**2, axis=1))
                new_dists[i] = np.inf
                new_min_d_pair = np.min(new_dists)
                
                new_d_wall_x = min(0.5 - new_c[0], new_c[0] + 0.5)
                new_d_wall_y = min(0.5 - new_c[1], new_c[1] + 0.5)
                new_d_wall_z = min(0.5 - new_c[2], new_c[2] + 0.5)
                new_min_d_wall = min(new_d_wall_x, new_d_wall_y, new_d_wall_z)
                
                new_local_min = min(new_min_d_pair / 2.0, new_min_d_wall)
                
                if new_local_min > best_local_min + 1e-12:
                    best_local_min = new_local_min
                    best_move = new_c
            
            if best_move is not None:
                centers[i] = best_move
                improved = True
        
        # Update best
        current_radius = _compute_radius(centers)
        if current_radius > best_radius:
            best_radius = current_radius
            best_centers = centers.copy()
        
        # Decay step
        step *= decay
        if step < min_step:
            step = 0.001
            # Add small noise
            noise = np.random.normal(0, 0.0005, centers.shape)
            centers = np.clip(centers + noise, -0.499, 0.499)
    
    return best_centers, best_radius

def _final_polish(centers, time_limit):
    """
    High-precision polish: very small steps, tight tolerance.
    """
    start_time = time.time()
    centers = centers.copy()
    best_radius = _compute_radius(centers)
    best_centers = centers.copy()
    
    step = 0.001
    decay = 0.9995
    min_step = 0.000001
    
    while time.time() - start_time < time_limit:
        improved = False
        for i in range(N):
            if time.time() - start_time > time_limit:
                break
            
            c_i = centers[i]
            
            # Try small perturbations in random directions
            for _ in range(5):
                direction = np.random.normal(0, 1, 3)
                direction /= np.linalg.norm(direction)
                
                new_c = c_i + step * direction
                new_c = np.clip(new_c, -0.499, 0.499)
                
                # Quick check: only accept if it improves local min
                dists = np.sqrt(np.sum((centers - new_c)**2, axis=1))
                dists[i] = np.inf
                new_min_d_pair = np.min(dists)
                
                new_d_wall_x = min(0.5 - new_c[0], new_c[0] + 0.5)
                new_d_wall_y = min(0.5 - new_c[1], new_c[1] + 0.5)
                new_d_wall_z = min(0.5 - new_c[2], new_c[2] + 0.5)
                new_min_d_wall = min(new_d_wall_x, new_d_wall_y, new_d_wall_z)
                
                new_local_min = min(new_min_d_pair / 2.0, new_min_d_wall)
                
                # Old local min
                old_dists = np.sqrt(np.sum((centers - c_i)**2, axis=1))
                old_dists[i] = np.inf
                old_min_d_pair = np.min(old_dists)
                
                old_d_wall_x = min(0.5 - c_i[0], c_i[0] + 0.5)
                old_d_wall_y = min(0.5 - c_i[1], c_i[1] + 0.5)
                old_d_wall_z = min(0.5 - c_i[2], c_i[2] + 0.5)
                old_min_d_wall = min(old_d_wall_x, old_d_wall_y, old_d_wall_z)
                
                old_local_min = min(old_min_d_pair / 2.0, old_min_d_wall)
                
                if new_local_min > old_local_min + 1e-14:
                    centers[i] = new_c
                    improved = True
                    break
        
        current_radius = _compute_radius(centers)
        if current_radius > best_radius:
            best_radius = current_radius
            best_centers = centers.copy()
        
        step *= decay
        if step < min_step:
            break
    
    return best_centers, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_centers = None
    best_radius = 0.0
    
    # Initialize with different strategies
    inits = [
        _init_grid,
        _init_fcc,
        _init_bcc,
        _init_random,
        _init_random,
        _init_random,
    ]
    
    # Time budget allocation
    # 6 restarts, each gets a portion of the time
    # Each restart: anneal (30%) -> smooth refine (40%) -> exact refine (20%) -> polish (10%)
    
    for idx, init_func in enumerate(inits):
        if time.time() > deadline:
            break
        
        elapsed = time.time() - start_time
        remaining = deadline - start_time
        time_per_restart = remaining / (len(inits) - idx)
        
        # Initialize
        centers = init_func()
        
        # Phase 1: Repulsion annealing (30% of restart time)
        anneal_time = time_per_restart * 0.3
        if time.time() < deadline - anneal_time - 1:
            centers, r = _repulsion_anneal(centers, anneal_time)
            if r > best_radius:
                best_radius = r
                best_centers = centers.copy()
        
        # Phase 2: Smooth refine (40% of restart time)
        smooth_time = time_per_restart * 0.4
        if time.time() < deadline - smooth_time - 1:
            centers, r = _refine_smooth(centers, smooth_time)
            if r > best_radius:
                best_radius = r
                best_centers = centers.copy()
        
        # Phase 3: Exact refine (20% of restart time)
        exact_time = time_per_restart * 0.2
        if time.time() < deadline - exact_time - 1:
            centers, r = _refine_exact(centers, exact_time)
            if r > best_radius:
                best_radius = r
                best_centers = centers.copy()
        
        # Phase 4: Final polish (10% of restart time)
        polish_time = time_per_restart * 0.1
        if time.time() < deadline - polish_time - 0.5:
            centers, r = _final_polish(centers, polish_time)
            if r > best_radius:
                best_radius = r
                best_centers = centers.copy()
        
        # Early termination if we've reached the known best
        if best_radius > 0.1359:
            break
    
    # Final check and return
    if best_centers is None:
        best_centers = _init_grid()
    
    # Ensure all centers are strictly inside the cube
    best_centers = np.clip(best_centers, -0.499, 0.499)
    
    return best_centers
