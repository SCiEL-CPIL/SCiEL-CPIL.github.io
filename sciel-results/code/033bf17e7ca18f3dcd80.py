import numpy as np
import time
from scipy.optimize import minimize

N = 100

def _compute_radius(c):
    """Compute the actual radius for a configuration."""
    if not np.all(np.isfinite(c)):
        return 0.0
    c = np.clip(c, -0.5, 0.5)
    if c.shape != (N, 3):
        return 0.0
    
    # Check for coincident centres
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    min_pair_dist = np.sqrt(np.min(dists_sq)) / 2.0
    
    # Wall distances
    wall_dists = 0.5 - np.abs(c)
    min_wall_dist = np.min(wall_dists)
    
    r = min(min_pair_dist, min_wall_dist)
    if r <= 0 or not np.isfinite(r):
        return 0.0
    return r

def _soft_min_objective_and_grad(c, beta):
    """
    Compute the soft-min objective and its gradient.
    c: shape (N, 3)
    beta: temperature parameter
    Returns: (objective_value, gradient)
    """
    c = c.reshape(N, 3)
    
    # Pairwise distances
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    
    # We want min of (dists/2) and wall_dists
    # pair_dists = dists / 2
    pair_dists = np.sqrt(dists_sq) / 2.0
    np.fill_diagonal(pair_dists, np.inf)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(c)
    
    # Combine all distances
    # Flatten pair distances (upper triangle to avoid double counting? 
    # Actually for soft min, double counting just changes the effective beta.
    # But to be consistent, let's just use all off-diagonal pairs.
    # Or better, use a vectorized log-sum-exp.
    
    # Let's compute the soft min properly.
    # x_all contains all pairwise distances and wall distances.
    
    # For efficiency, we can compute the log-sum-exp in parts or just flatten.
    # N=100, pairs ~ 4950 (if upper triangle) or 9900 (all off-diag).
    # Walls 300.
    # Total ~ 5250 or 10200. Small enough to flatten.
    
    # Let's use all off-diagonal pairs for simplicity, it's a valid soft-min approximation.
    # mask for off-diagonal
    mask = ~np.eye(N, dtype=bool)
    pair_dists_flat = pair_dists[mask]
    wall_dists_flat = wall_dists.flatten()
    
    x_all = np.concatenate([pair_dists_flat, wall_dists_flat])
    
    # Soft min: -1/beta * log(sum(exp(-beta * x)))
    # Let y = -beta * x
    y = -beta * x_all
    y_max = np.max(y)
    if not np.isfinite(y_max):
        return 0.0, np.zeros_like(c)
    
    exp_y = np.exp(y - y_max)
    s = np.sum(exp_y)
    log_sum_exp = y_max + np.log(s)
    obj = log_sum_exp / beta
    
    # Gradient
    # d(obj)/dx_i = -1/beta * d(log(sum(exp(-beta x))))/dx_i
    # = -1/beta * (1/s * sum_j d(exp(-beta x_j))/dx_i)
    # For x_j = dist(i,j)/2: d(dist(i,j)/2)/dx_i = (x_i - x_j) / dist(i,j)
    # For wall dist: d(0.5 - |x_i|)/dx_i = -sign(x_i)
    
    grad = np.zeros((N, 3))
    
    # We need to map the flattened x_all back to contributions.
    # This is tricky to do efficiently without loops or complex indexing.
    # Given the time constraint and that L-BFGS-B can use numerical gradients if we don't provide one,
    # but providing an analytical gradient is much faster.
    # However, implementing the gradient for the flattened soft-min is complex.
    
    # Alternative: Use a different objective that is easier to differentiate?
    # Or just use numerical gradients? For 300 dimensions, numerical gradients are 301 evaluations per step.
    # L-BFGS-B with numerical gradients might be too slow for 150s.
    
    # Let's try to implement the gradient carefully.
    # The term for pair (i,j) in x_all is dist(i,j)/2.
    # Contribution to grad[i] from pair (i,j):
    # -1/beta * (exp(-beta * d_ij/2) / s) * (c_i - c_j) / d_ij
    
    # Let's compute weights w_k = exp(-beta * x_k) / s for each k in x_all.
    w = exp_y / s
    
    # Now we need to distribute these weights back to the centers.
    # The first N*(N-1) entries are pairs.
    # The last 3*N entries are walls.
    
    n_pairs = N * (N - 1)
    w_pairs = w[:n_pairs]
    w_walls = w[n_pairs:]
    
    # Map pairs back to (i, j)
    # The mask was ~np.eye(N). The order is row-major.
    # For row i, the columns are 0..i-1, i+1..N-1.
    # This is annoying to invert.
    
    # Simpler approach: Compute the gradient of the soft-min of pairwise distances separately
    # and the soft-min of wall distances separately, then combine?
    # No, the soft-min of a union is not the sum of soft-mins.
    
    # Let's stick to numerical gradients for simplicity and correctness, 
    # but optimize the function evaluation to be fast.
    # Or, use a differentiable approximation that is easier to handle.
    
    # Actually, let's just return the objective and let scipy use numerical gradients.
    # To make it faster, we can use a smaller beta or fewer iterations.
    
    return obj, None

def _optimize_soft_min(c, beta, time_limit):
    """Optimize the soft-min objective using L-BFGS-B."""
    start_time = time.time()
    
    def objective(c_flat):
        c_reshaped = c_flat.reshape(N, 3)
        val, _ = _soft_min_objective_and_grad(c_reshaped, beta)
        return val
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    try:
        res = minimize(
            objective, 
            c.flatten(), 
            method='L-BFGS-B', 
            bounds=bounds,
            options={'maxiter': 500, 'maxfun': 2000, 'maxls': 50}
        )
        if time.time() - start_time < time_limit:
            return res.x.reshape(N, 3)
    except Exception:
        pass
    
    return c

def _simulated_annealing(c, start_temp, end_temp, time_limit):
    """
    Perform simulated annealing to escape local minima.
    c: current configuration
    start_temp: initial temperature
    end_temp: final temperature
    time_limit: max time in seconds
    """
    start_time = time.time()
    current_c = c.copy()
    current_r = _compute_radius(current_c)
    best_c = current_c.copy()
    best_r = current_r
    
    n_steps = 0
    while time.time() - start_time < time_limit:
        n_steps += 1
        # Geometric cooling
        t = start_temp * (end_temp / start_temp) ** (n_steps / 1000.0)
        if t < end_temp:
            t = end_temp
            
        # Generate a random perturbation
        # Move a random subset of spheres
        n_move = np.random.randint(1, 10)
        indices = np.random.choice(N, n_move, replace=False)
        noise = np.random.normal(0, t * 0.1, size=(n_move, 3))
        
        new_c = current_c.copy()
        new_c[indices] += noise
        new_c = np.clip(new_c, -0.5, 0.5)
        
        new_r = _compute_radius(new_c)
        
        # Acceptance criterion
        if new_r > current_r:
            current_c = new_c
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_c = new_c.copy()
        else:
            # Metropolis criterion
            delta = new_r - current_r
            if np.random.random() < np.exp(delta / t):
                current_c = new_c
                current_r = new_r
                
    return best_c

def _init_grid():
    """Initialize with a 5x5x4 grid."""
    x = np.linspace(-0.4, 0.4, 5)
    y = np.linspace(-0.4, 0.4, 5)
    z = np.linspace(-0.375, 0.375, 4)
    c = np.array([(xi, yi, zi) for zi in z for yi in y for xi in x])
    return c

def _init_fcc_like():
    """Initialize with an FCC-like structure."""
    # FCC unit cell has 4 atoms.
    # We can tile this.
    # For 100 spheres, maybe 5x5x4 layers of FCC?
    # This is complex to generate correctly.
    # Let's use a simple lattice with offsets.
    c = []
    # Layer 1: (0,0,0)
    # Layer 2: (0.5, 0.5, 0)
    # Layer 3: (0.5, 0, 0.5)
    # Layer 4: (0, 0.5, 0.5)
    
    # Scale to fit in [-0.5, 0.5]
    # Let's just use a random perturbation of a grid for now.
    c = _init_grid()
    c += np.random.normal(0, 0.01, size=c.shape)
    return np.clip(c, -0.5, 0.5)

def _init_random():
    """Initialize with random points."""
    return np.random.uniform(-0.4, 0.4, size=(N, 3))

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_c = _init_grid()
    best_r = _compute_radius(best_c)
    
    # 1. Local optimization with soft-min
    betas = [10, 50, 200, 1000]
    current_c = best_c.copy()
    
    for beta in betas:
        if time.time() > deadline:
            break
        time_left = deadline - time.time()
        if time_left < 10:
            break
            
        # Allocate time for this optimization step
        step_time = min(time_left / 2, 30)
        
        new_c = _optimize_soft_min(current_c, beta, step_time)
        r = _compute_radius(new_c)
        
        if r > best_r:
            best_r = r
            best_c = new_c.copy()
        current_c = new_c
        
    # 2. Simulated Annealing to escape local minima
    if time.time() < deadline - 20:
        time_left = deadline - time.time() - 5
        if time_left > 10:
            # Use a fraction of remaining time for SA
            sa_time = time_left * 0.5
            # Start from the best found so far
            sa_c = _simulated_annealing(best_c, start_temp=0.1, end_temp=0.001, time_limit=sa_time)
            r = _compute_radius(sa_c)
            if r > best_r:
                best_r = r
                best_c = sa_c.copy()
                
    # 3. Final polish with high beta soft-min
    if time.time() < deadline - 5:
        time_left = deadline - time.time() - 2
        if time_left > 5:
            new_c = _optimize_soft_min(best_c, 5000, time_left)
            r = _compute_radius(new_c)
            if r > best_r:
                best_r = r
                best_c = new_c.copy()
                
    # Ensure validity
    if not np.all(np.isfinite(best_c)) or best_c.shape != (N, 3):
        best_c = _init_grid()
        
    return best_c

if __name__ == "__main__":
    c = construct()
    print("Radius:", _compute_radius(c))
