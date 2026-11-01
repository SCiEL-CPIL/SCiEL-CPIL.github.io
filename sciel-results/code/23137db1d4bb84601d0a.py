import numpy as np
import time
from scipy.spatial import cKDTree

N = 100

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin (5s before 150s limit)
    
    def get_radius(centers):
        """Compute the maximum common radius for given centers."""
        if not np.all(np.isfinite(centers)):
            return 0.0
        if np.any(centers > 0.5) or np.any(centers < -0.5):
            return 0.0
        # Wall distance
        wall_dists = 0.5 - np.abs(centers)
        r_wall = np.min(wall_dists)
        # Pairwise distance using KDTree for efficiency
        tree = cKDTree(centers)
        # Get all pairwise distances (upper triangle)
        dists, _ = tree.query(centers, k=2)  # k=2 includes self, so take second column
        r_pair = np.min(dists[:, 1]) / 2.0
        return min(r_wall, r_pair)

    def simulated_annealing(centers, time_limit, initial_temp=0.02, cooling=0.9995):
        """Simulated annealing to optimize sphere packing."""
        centers = centers.copy()
        current_r = get_radius(centers)
        best_centers = centers.copy()
        best_r = current_r
        temp = initial_temp
        
        # Precompute initial tree for speed
        tree = cKDTree(centers)
        
        while time.time() < time_limit and temp > 1e-6:
            # Random perturbation of one or more centers
            n_perturb = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
            indices = np.random.choice(N, n_perturb, replace=False)
            
            # Perturbation magnitude scales with temperature
            step_size = temp * 0.5
            new_centers = centers.copy()
            for idx in indices:
                new_centers[idx] += np.random.normal(0, step_size, 3)
            
            # Clip to stay in bounds
            new_centers = np.clip(new_centers, -0.499, 0.499)
            
            new_r = get_radius(new_centers)
            
            # Metropolis acceptance criterion
            delta = new_r - current_r
            if delta > 0 or np.random.random() < np.exp(delta / (temp + 1e-10)):
                current_r = new_r
                centers = new_centers
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()
            
            # Cool down
            temp *= cooling
            
            # Periodically rebuild tree
            if np.random.random() < 0.01:
                tree = cKDTree(centers)
        
        return best_centers, best_r

    def local_refine(centers, time_limit):
        """Local refinement using gradient-based method."""
        centers = centers.copy()
        r = get_radius(centers)
        best_centers = centers.copy()
        best_r = r
        
        for it in range(5000):
            if time.time() > time_limit:
                break
            
            # Compute current radius and identify bottleneck constraints
            wall_dists = 0.5 - np.abs(centers)
            r_wall = np.min(wall_dists)
            
            tree = cKDTree(centers)
            dists, _ = tree.query(centers, k=2)
            min_pair_dist = np.min(dists[:, 1])
            r_pair = min_pair_dist / 2.0
            
            current_r = min(r_wall, r_pair)
            
            # Compute gradient: push centers to increase minimum radius
            grad = np.zeros_like(centers)
            
            # Wall constraints: if wall_dist < r, push inward
            for k in range(3):
                d_wall = 0.5 - np.abs(centers[:, k])
                mask = d_wall < current_r + 0.01
                if np.any(mask):
                    f_mag = np.clip((current_r + 0.01 - d_wall[mask]) * 0.5, 0, None)
                    dirs = -np.sign(centers[mask, k])
                    dirs[dirs == 0] = 1
                    grad[mask, k] += f_mag * dirs
            
            # Pair constraints: if dist < 2*r, push apart
            target_dist = 2 * (current_r + 0.01)
            pairs = tree.query_pairs(target_dist)
            
            for i, j in pairs:
                if i == j:
                    continue
                diff = centers[i] - centers[j]
                d = np.linalg.norm(diff)
                if d > 1e-10 and d < target_dist:
                    f_mag = (target_dist - d) / d * 0.5
                    grad[i] += f_mag * diff
                    grad[j] -= f_mag * diff
            
            # Normalize gradient to avoid huge steps
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-10:
                grad = grad / grad_norm
            
            # Adaptive step size
            base_step = 0.03
            decay = 1.0 / (1.0 + it / 50.0)
            step = base_step * decay
            
            # Try the step
            new_centers = centers + step * grad
            new_centers = np.clip(new_centers, -0.499, 0.499)
            
            new_r = get_radius(new_centers)
            
            if new_r > r:
                r = new_r
                centers = new_centers
                if r > best_r:
                    best_r = r
                    best_centers = centers.copy()
            else:
                # Try smaller step
                step *= 0.5
                new_centers = centers + step * grad
                new_centers = np.clip(new_centers, -0.499, 0.499)
                new_r = get_radius(new_centers)
                if new_r > r:
                    r = new_r
                    centers = new_centers
                    if r > best_r:
                        best_r = r
                        best_centers = centers.copy()
        
        return best_centers, best_r

    # Generate initial configurations
    np.random.seed(42)
    
    best_centers = None
    best_radius = 0.0
    
    # Configuration 1: Regular grid with jitter
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.42, 0.42, nx)
    ys = np.linspace(-0.42, 0.42, ny)
    zs = np.linspace(-0.42, 0.42, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    base_centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    # Configuration 2: Face-centered cubic-like
    a = 0.7  # lattice constant
    fcc_points = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                fcc_points.append([i*a, j*a, k*a])
                fcc_points.append([i*a, j*a + a/2, k*a + a/2])
                fcc_points.append([i*a + a/2, j*a, k*a + a/2])
                fcc_points.append([i*a + a/2, j*a + a/2, k*a])
    
    fcc_array = np.array(fcc_points)
    mask = np.all(np.abs(fcc_array) <= 0.45, axis=1)
    fcc_array = fcc_array[mask]
    if len(fcc_array) >= N:
        idx = np.random.choice(len(fcc_array), N, replace=False)
        fcc_centers = fcc_array[idx]
    else:
        fcc_centers = base_centers.copy()
    
    # Try multiple initializations
    initializations = []
    
    # Grid with various jitter levels
    for jitter in [0.0, 0.02, 0.04, 0.06]:
        centers = base_centers + np.random.normal(0, jitter, base_centers.shape)
        centers = np.clip(centers, -0.49, 0.49)
        initializations.append(centers)
    
    # FCC with jitter
    for jitter in [0.0, 0.02, 0.04]:
        centers = fcc_centers + np.random.normal(0, jitter, fcc_centers.shape)
        centers = np.clip(centers, -0.49, 0.49)
        initializations.append(centers)
    
    # Random initialization
    for _ in range(3):
        centers = np.random.uniform(-0.45, 0.45, (N, 3))
        initializations.append(centers)
    
    # Phase 1: Simulated annealing on multiple initializations
    for idx, init_centers in enumerate(initializations):
        if time.time() > deadline - 30:
            break
        
        remaining = deadline - 30 - time.time()
        if remaining < 5:
            break
        
        # Allocate time based on initialization quality
        r_init = get_radius(init_centers)
        
        if r_init > 0.08:
            time_budget = remaining * 0.4
        elif r_init > 0.06:
            time_budget = remaining * 0.3
        else:
            time_budget = remaining * 0.2
        
        sa_time_limit = time.time() + min(time_budget, 15)
        
        sa_centers, sa_r = simulated_annealing(init_centers, sa_time_limit, initial_temp=0.03, cooling=0.9998)
        
        if sa_r > best_radius:
            best_radius = sa_r
            best_centers = sa_centers
    
    # Phase 2: Local refinement on best solutions
    if best_centers is not None and time.time() < deadline - 10:
        remaining = deadline - 10 - time.time()
        if remaining > 5:
            # Do additional local optimization
            opt_centers, r = local_refine(best_centers, deadline - 3)
            if r > best_radius:
                best_radius = r
                best_centers = opt_centers
    
    # Ensure valid output
    if best_centers is None or not np.all(np.isfinite(best_centers)):
        return base_centers
    
    # Final clip to ensure validity
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
