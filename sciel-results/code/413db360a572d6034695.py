import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 100

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    def get_radius(centers):
        """Compute the maximum common radius for given centers."""
        if not np.all(np.isfinite(centers)):
            return 0.0
        if np.any(centers > 0.5) or np.any(centers < -0.5):
            return 0.0
        wall_dists = 0.5 - np.abs(centers)
        r_wall = np.min(wall_dists)
        tree = cKDTree(centers)
        dists, _ = tree.query(centers, k=2)
        r_pair = np.min(dists[:, 1]) / 2.0
        return min(r_wall, r_pair)

    def penalty_objective_and_grad(x, r_target, mu):
        """
        Penalty objective: maximize r by minimizing penalty for violations.
        x: flat array of centers (N*3,)
        r_target: current target radius
        mu: penalty weight
        
        Objective: -r_target + mu * sum(max(0, 2*r_target - d_ij)^2) + mu * sum(max(0, r_target - wall_dist_i)^2)
        """
        centers = x.reshape(N, 3)
        
        # Wall violations
        wall_dists = 0.5 - np.abs(centers)
        wall_viol = np.maximum(0, r_target - wall_dists)
        wall_penalty = np.sum(wall_viol ** 2)
        
        # Pairwise violations using KDTree for efficiency
        tree = cKDTree(centers)
        target_dist = 2 * r_target
        pairs = tree.query_pairs(target_dist + 0.01)
        
        pair_penalty = 0.0
        pair_grad = np.zeros_like(centers)
        
        for i, j in pairs:
            diff = centers[i] - centers[j]
            d = np.linalg.norm(diff)
            if d < target_dist and d > 1e-12:
                violation = target_dist - d
                pair_penalty += violation ** 2
                # Gradient: d/dxi sum(violation^2) where violation = 2r - d
                # d(violation)/dxi = -d(xi - xj)/d = -diff/d
                # d(violation^2)/dxi = 2*violation * (-diff/d) = -2*violation*diff/d
                factor = -2.0 * violation / d
                pair_grad[i] += factor * diff
                pair_grad[j] -= factor * diff
        
        # Wall gradient
        wall_grad = np.zeros_like(centers)
        for k in range(3):
            wd = 0.5 - np.abs(centers[:, k])
            mask = wd < r_target
            if np.any(mask):
                viol = r_target - wd[mask]
                # d(viol^2)/dxi_k = 2*viol * d(viol)/dxi_k
                # viol = r - (0.5 - |xi_k|) => d(viol)/dxi_k = sign(xi_k)
                wall_grad[mask, k] += 2.0 * viol * np.sign(centers[mask, k])
        
        obj = -r_target + mu * (pair_penalty + wall_penalty)
        grad = mu * (pair_grad + wall_grad)
        
        return obj, grad

    def optimize_with_lbfgs(centers, time_limit, r_target=None, mu_init=100.0):
        """
        Optimize using L-BFGS-B with penalty method.
        Iteratively increases r_target and minimizes penalty.
        """
        centers = centers.copy().flatten()
        if r_target is None:
            r_target = get_radius(centers.reshape(N, 3))
        
        mu = mu_init
        best_centers = centers.copy()
        best_r = r_target
        
        # Outer loop: increase r_target
        for outer_iter in range(30):
            if time.time() > time_limit:
                break
            
            # Inner L-BFGS optimization at current r_target and mu
            result = minimize(
                penalty_objective_and_grad,
                centers,
                args=(r_target, mu),
                method='L-BFGS-B',
                jac=True,
                bounds=[(-0.5, 0.5)] * (N * 3),
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-6}
            )
            
            if result.success or result.fun < 1e-6:
                # Check if penalty is small enough to accept this r_target
                centers = result.x
                r_check = get_radius(centers.reshape(N, 3))
                if r_check >= r_target - 1e-6:
                    best_r = max(best_r, r_check)
                    best_centers = centers.copy()
                    # Increase r_target
                    r_target *= 1.01
                    mu *= 0.8  # Reduce penalty as we get closer
                else:
                    # Penalty not small enough, increase mu
                    mu *= 1.5
            else:
                # L-BFGS didn't converge well, try with higher mu
                mu *= 1.5
            
            if mu > 1e6:
                break
        
        return best_centers.reshape(N, 3), best_r

    def simulated_annealing(centers, time_limit, initial_temp=0.02, cooling=0.9995):
        """Simulated annealing to optimize sphere packing."""
        centers = centers.copy()
        current_r = get_radius(centers)
        best_centers = centers.copy()
        best_r = current_r
        temp = initial_temp
        
        while time.time() < time_limit and temp > 1e-6:
            n_perturb = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
            indices = np.random.choice(N, n_perturb, replace=False)
            
            step_size = temp * 0.5
            new_centers = centers.copy()
            for idx in indices:
                new_centers[idx] += np.random.normal(0, step_size, 3)
            
            new_centers = np.clip(new_centers, -0.499, 0.499)
            
            new_r = get_radius(new_centers)
            
            delta = new_r - current_r
            if delta > 0 or np.random.random() < np.exp(delta / (temp + 1e-10)):
                current_r = new_r
                centers = new_centers
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()
            
            temp *= cooling
        
        return best_centers, best_r

    # Generate initial configurations
    np.random.seed(42)
    
    best_centers = None
    best_radius = 0.0
    
    # Configuration 1: Regular grid
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.42, 0.42, nx)
    ys = np.linspace(-0.42, 0.42, ny)
    zs = np.linspace(-0.42, 0.42, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    base_centers = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    # Configuration 2: FCC-like
    a = 0.7
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
    
    for jitter in [0.0, 0.02, 0.04]:
        centers = base_centers + np.random.normal(0, jitter, base_centers.shape)
        centers = np.clip(centers, -0.49, 0.49)
        initializations.append(centers)
    
    for jitter in [0.0, 0.02]:
        centers = fcc_centers + np.random.normal(0, jitter, fcc_centers.shape)
        centers = np.clip(centers, -0.49, 0.49)
        initializations.append(centers)
    
    for _ in range(2):
        centers = np.random.uniform(-0.45, 0.45, (N, 3))
        initializations.append(centers)
    
    # Phase 1: Simulated annealing on multiple initializations
    for idx, init_centers in enumerate(initializations):
        if time.time() > deadline - 40:
            break
        
        remaining = deadline - 40 - time.time()
        if remaining < 3:
            break
        
        r_init = get_radius(init_centers)
        
        if r_init > 0.08:
            time_budget = min(remaining * 0.3, 10)
        elif r_init > 0.06:
            time_budget = min(remaining * 0.2, 8)
        else:
            time_budget = min(remaining * 0.15, 5)
        
        sa_time_limit = time.time() + max(time_budget, 2)
        
        sa_centers, sa_r = simulated_annealing(init_centers, sa_time_limit, initial_temp=0.03, cooling=0.9998)
        
        if sa_r > best_radius:
            best_radius = sa_r
            best_centers = sa_centers
    
    # Phase 2: L-BFGS-B penalty optimization on best solutions
    if best_centers is not None and time.time() < deadline - 10:
        remaining = deadline - 10 - time.time()
        if remaining > 5:
            lbfgs_centers, r = optimize_with_lbfgs(best_centers, deadline - 3)
            if r > best_radius:
                best_radius = r
                best_centers = lbfgs_centers
    
    # Ensure valid output
    if best_centers is None or not np.all(np.isfinite(best_centers)):
        return base_centers
    
    # Final clip to ensure validity
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
