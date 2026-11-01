import numpy as np
import time
from scipy.spatial import cKDTree
from scipy.optimize import minimize

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    def compute_radius(c):
        """Compute the maximum common radius for given centers."""
        if not np.all(np.isfinite(c)):
            return 0.0
        if np.any(c > 0.5) or np.any(c < -0.5):
            return 0.0
        wall_dists = 0.5 - np.abs(c)
        r_wall = np.min(wall_dists)
        try:
            tree = cKDTree(c)
            dists, _ = tree.query(c, k=2)
            r_pair = np.min(dists[:, 1]) / 2.0
        except Exception:
            r_pair = 0.0
        return min(r_wall, r_pair)

    def penalty_objective_and_grad(x, r, penalty_weight):
        """
        Penalty objective: minimize sum of squared violations of constraints.
        x is flat array of 300 coordinates (N spheres * 3 dims).
        r is the target radius to try to achieve.
        """
        centers = x.reshape(N, 3)
        
        # Wall constraints: 0.5 - |c_ik| >= r
        # Violation: max(0, r - (0.5 - |c_ik|))
        wall_viol = np.maximum(0, r - (0.5 - np.abs(centers)))
        
        # Pair constraints: |c_i - c_j| >= 2r
        # We need to check all pairs. For efficiency, use KDTree to find close pairs.
        try:
            tree = cKDTree(centers)
            # Find pairs within distance 2*r + epsilon
            # If no pairs are violating, pair_viol is 0
            # We compute sum of squared violations for pairs closer than 2r
            # Note: query_pairs gives unique pairs (i, j) with i < j
            pairs = tree.query_pairs(2 * r + 0.01)
            
            pair_viol = 0.0
            pair_grad = np.zeros_like(centers)
            
            for i, j in pairs:
                diff = centers[i] - centers[j]
                d = np.linalg.norm(diff)
                if d < 2 * r and d > 1e-10:
                    viol = 2 * r - d
                    pair_viol += viol * viol
                    # Gradient of viol^2 = 2 * viol * d(viol/dx)
                    # d(viol/dx_i) = -diff/d
                    # d(viol/dx_j) = +diff/d
                    factor = 2 * viol / d
                    pair_grad[i] -= factor * diff
                    pair_grad[j] += factor * diff
        except Exception:
            # Fallback if tree fails, assume high penalty
            pair_viol = 100.0
            pair_grad = np.zeros_like(centers)
        
        wall_viol_sum = np.sum(wall_viol ** 2)
        total_viol = wall_viol_sum + pair_viol
        
        # Gradient
        # Wall gradient: d/dx_k (max(0, r - (0.5 - |x_k|))^2)
        # If r > 0.5 - |x_k|:
        #   d/dx_k = 2 * (r - (0.5 - |x_k|)) * d(r - 0.5 + |x_k|)/dx_k
        #   d(|x_k|)/dx_k = sign(x_k)
        #   So grad = 2 * viol * sign(x_k)
        wall_grad = np.zeros_like(centers)
        mask = wall_viol > 0
        for k in range(3):
            wall_grad[mask, k] = 2 * wall_viol[mask, k] * np.sign(centers[mask, k])
            # Handle x_k == 0: sign is 0, but subgradient can be anything in [-1, 1].
            # If viol > 0 and x_k == 0, we are at the minimum of |x_k|, so we should push away?
            # Actually if x_k=0, wall dist is 0.5. If r > 0.5, it's always violated.
            # But r is small.
            # If x_k is very close to 0, sign is 0. Let's use a small epsilon direction or just 0.
            # Actually, if x_k=0, moving in any direction increases |x_k| and reduces violation.
            # The gradient is undefined. We can pick a random direction or 0.
            # Let's stick to 0 for simplicity, or use a fixed direction like 1.
            # Let's use 1.0 if sign is 0 and viol > 0.
            zero_mask = mask & (np.abs(centers[:, k]) < 1e-10)
            wall_grad[zero_mask, k] = 2 * wall_viol[zero_mask, k] * 1.0

        total_grad = wall_grad + pair_grad
        
        obj = penalty_weight * total_viol
        grad = penalty_weight * total_grad
        
        return obj, grad.flatten()

    def refine_with_lbfgs(centers, time_limit, r_target):
        """
        Use L-BFGS-B to minimize the penalty function for a given target radius r_target.
        Returns refined centers and the achieved radius.
        """
        x0 = centers.flatten()
        
        # We want to find if we can achieve r_target.
        # If the penalty goes to 0, we can increase r_target.
        # For now, just minimize penalty for this r_target.
        
        # Set bounds
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        # Limit iterations and time
        # We can't easily limit time inside minimize, so we use maxiter and a tight bound.
        # Or we can run minimize in a loop with small maxiter? No, that's inefficient.
        # Let's just run it with a reasonable maxiter and hope it finishes.
        # Given 150s limit, and this is one of many calls, we need to be careful.
        # A single L-BFGS call on 300 vars should be fast (<1s usually).
        
        try:
            res = minimize(
                penalty_objective_and_grad,
                x0,
                args=(r_target, 10.0), # penalty_weight=10
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
            )
        except Exception:
            return centers, compute_radius(centers)
        
        new_centers = res.x.reshape(N, 3)
        new_r = compute_radius(new_centers)
        
        return new_centers, new_r

    def simulated_annealing(centers, time_limit, initial_temp=0.05, cooling=0.9995):
        """
        Simulated Annealing to explore the space.
        """
        centers = centers.copy()
        current_r = compute_radius(centers)
        best_centers = centers.copy()
        best_r = current_r
        temp = initial_temp
        
        while time.time() < time_limit and temp > 1e-5:
            # Random perturbation
            n_move = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
            indices = np.random.choice(N, n_move, replace=False)
            step_size = temp * 0.5
            new_centers = centers.copy()
            new_centers[indices] += np.random.normal(0, step_size, (n_move, 3))
            new_centers = np.clip(new_centers, -0.4999, 0.4999)
            
            new_r = compute_radius(new_centers)
            
            # Metropolis Criterion
            delta = new_r - current_r
            if delta > 0 or np.random.random() < np.exp(delta / (temp + 1e-10)):
                current_r = new_r
                centers = new_centers
                if current_r > best_r:
                    best_r = current_r
                    best_centers = centers.copy()
            
            temp *= cooling
            
        return best_centers, best_r

    def hybrid_optimize(centers, time_limit):
        """
        Hybrid approach: SA to find good basin, then L-BFGS refinement.
        """
        centers = centers.copy()
        best_centers = centers.copy()
        best_r = compute_radius(centers)
        
        # Phase 1: SA for a portion of time
        sa_time = time_limit * 0.4
        sa_limit = time.time() + sa_time
        sa_centers, sa_r = simulated_annealing(centers, sa_limit)
        
        if sa_r > best_r:
            best_r = sa_r
            best_centers = sa_centers
            
        # Phase 2: L-BFGS refinement on the best found so far
        # Try to improve the radius by increasing target r and minimizing penalty
        # We will do a few iterations of "increase r, minimize penalty, check if valid"
        
        current_centers = best_centers.copy()
        current_r = best_r
        
        # Binary search or incremental increase?
        # Let's try to push r higher.
        # Start with current_r and try to increase it.
        
        # We have remaining time
        # Each L-BFGS call takes ~0.1-0.5s.
        # Let's do a loop:
        
        while time.time() < time_limit:
            # Try to increase r by a small amount
            # How much? Maybe 1% or a fixed small step?
            # If we succeed (penalty ~ 0), we keep it. If not, we back off.
            
            # Let's try a step based on current r
            step = max(0.0005, current_r * 0.01)
            target_r = current_r + step
            
            # Refine
            new_centers, new_r = refine_with_lbfgs(current_centers, time.time() + 1.0, target_r)
            
            if new_r > current_r:
                current_r = new_r
                current_centers = new_centers
                if current_r > best_r:
                    best_r = current_r
                    best_centers = current_centers.copy()
            else:
                # If we couldn't improve, maybe try a smaller step or break
                # If new_r is not significantly better than current_r, we might be stuck.
                # Let's try a smaller step
                if step < 1e-4:
                    break
                step *= 0.5
                target_r = current_r + step
                new_centers, new_r = refine_with_lbfgs(current_centers, time.time() + 1.0, target_r)
                if new_r > current_r:
                    current_r = new_r
                    current_centers = new_centers
                    if current_r > best_r:
                        best_r = current_r
                        best_centers = current_centers.copy()
                else:
                    # Stagnation in L-BFGS phase
                    # Do a small SA jump to escape
                    if time.time() < time_limit - 2:
                        jump_centers, jump_r = simulated_annealing(current_centers, time.time() + 1.0, initial_temp=0.02)
                        if jump_r > current_r:
                            current_r = jump_r
                            current_centers = jump_centers
                            if current_r > best_r:
                                best_r = current_r
                                best_centers = current_centers.copy()
                        else:
                            break
                    else:
                        break
                        
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
    
    # Configuration 2: Face-centered cubic-like
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
    
    initializations = []
    
    # Grid with various jitter levels
    for jitter in [0.0, 0.02, 0.04]:
        centers = base_centers + np.random.normal(0, jitter, base_centers.shape)
        centers = np.clip(centers, -0.49, 0.49)
        initializations.append(centers)
    
    # FCC with jitter
    for jitter in [0.0, 0.02]:
        centers = fcc_centers + np.random.normal(0, jitter, fcc_centers.shape)
        centers = np.clip(centers, -0.49, 0.49)
        initializations.append(centers)
        
    # Random initialization
    centers = np.random.uniform(-0.45, 0.45, (N, 3))
    initializations.append(centers)

    # Optimize each initialization
    for idx, init_centers in enumerate(initializations):
        if time.time() > deadline - 15:
            break
            
        remaining = deadline - 15 - time.time()
        if remaining < 10:
            break
            
        # Allocate time
        time_budget = min(remaining * 0.8, 25)
        
        hybrid_centers, hybrid_r = hybrid_optimize(init_centers, time_budget)
        
        if hybrid_r > best_radius:
            best_radius = hybrid_r
            best_centers = hybrid_centers

    # Ensure valid output
    if best_centers is None or not np.all(np.isfinite(best_centers)):
        return base_centers

    # Final clip to ensure validity
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
