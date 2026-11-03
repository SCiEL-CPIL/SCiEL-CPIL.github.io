import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Helper functions
    def compute_radius(points):
        """Compute the maximum radius for a given set of points."""
        if not np.all(np.isfinite(points)):
            return -1.0
        if np.any(points < -0.5) or np.any(points > 0.5):
            return -1.0
        
        # Wall distance
        wall_dists = 0.5 - np.abs(points)
        min_wall = np.min(wall_dists)
        
        # Pairwise distance
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = np.min(dists)
        
        r = min(min_wall, min_pair / 2.0)
        
        if min_pair < 1e-12:
            return 0.0
            
        return r

    def softmin_objective(x, beta):
        """
        Smooth surrogate for the negative of the minimum radius.
        We minimize -r, so we maximize r.
        r = min(min_wall, min_pair/2)
        -r = -min(min_wall, min_pair/2) = max(-min_wall, -min_pair/2)
        
        Actually, let's minimize a smooth approximation of -r.
        Let f = -r.
        r = min_i wall_i, min_{i<j} pair_{ij}/2
        So -r = max_i (-wall_i), max_{i<j} (-pair_{ij}/2)
        
        Softmax of these terms approximates the max.
        We want to minimize this.
        
        Alternatively, we can use the log-sum-exp trick for the min.
        min(a, b) ≈ -1/beta * log(exp(-beta*a) + exp(-beta*b))
        So r ≈ -1/beta * log( sum exp(-beta * constraints) )
        where constraints are wall_dists and pair_dists/2.
        
        We want to maximize r, so minimize -r.
        -r ≈ 1/beta * log( sum exp(-beta * constraints) )
        
        Let's define the objective as this smooth -r.
        """
        # Unpack x to points
        points = x.reshape(-1, 3)
        
        # Wall distances
        wall_dists = 0.5 - np.abs(points)
        flat_walls = wall_dists.flatten()
        
        # Pairwise distances / 2
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_constraints = dists / 2.0
        # Only take upper triangle to avoid double counting and self
        iu = np.triu_indices(N, k=1)
        flat_pairs = pair_constraints[iu]
        
        all_constraints = np.concatenate([flat_walls, flat_pairs])
        
        # Softmin approximation
        # min(x) ≈ -1/beta * log(sum(exp(-beta*x)))
        # We want to maximize min(x), so minimize -min(x)
        # -min(x) ≈ 1/beta * log(sum(exp(-beta*x)))
        
        # For numerical stability in log-sum-exp:
        # log(sum(exp(-beta*x))) = -beta*max(x) + log(sum(exp(-beta*(x-max(x)))))
        # Wait, if x are positive (distances), -beta*x are negative.
        # Let m = min(all_constraints).
        # sum(exp(-beta*x)) = exp(-beta*m) * sum(exp(-beta*(x-m)))
        # log(sum(...)) = -beta*m + log(sum(exp(-beta*(x-m))))
        # So -min(x) ≈ 1/beta * (-beta*m + log(sum(exp(-beta*(x-m)))))
        # = -m + (1/beta)*log(sum(exp(-beta*(x-m))))
        
        # As beta -> inf, this approaches -min(x) exactly?
        # Let x = [m, m+eps, ...]
        # sum(exp(-beta*(x-m))) = 1 + exp(-beta*eps) + ... ≈ 1
        # log(1) = 0
        # Result: -m + 0 = -m. Correct.
        
        m = np.min(all_constraints)
        # If m is very small, exp(-beta*(x-m)) might overflow if x-m is negative?
        # But m is the min, so x-m >= 0. So exp(-beta*(x-m)) <= 1. No overflow.
        # Underflow is fine.
        
        shifted = all_constraints - m
        log_sum_exp = np.log(np.sum(np.exp(-beta * shifted)))
        
        obj = -m + log_sum_exp / beta
        
        return obj

    def softmin_gradient(x, beta):
        """
        Gradient of the softmin objective.
        obj = -m + (1/beta) * log(sum(exp(-beta*(x-m))))
        Let S = sum(exp(-beta*(x-m)))
        d/dx_i log(S) = (1/S) * d/dx_i S
        d/dx_i S = sum_j exp(-beta*(x_j-m)) * (-beta * d(x_j-m)/dx_i)
        Note: m depends on x. This makes the gradient complex if we treat m as a variable.
        However, the softmin function is defined as:
        f(x) = -1/beta * log(sum(exp(-beta*x)))
        grad f(x) = (sum(exp(-beta*x) * x) * beta * ... ) / sum(exp(-beta*x)) ?
        
        Let's use the standard softmin gradient.
        f(x) = -1/beta * log(sum_j exp(-beta*x_j))
        df/dx_i = -1/beta * (1/sum) * sum_j exp(-beta*x_j) * (-beta * delta_{ij})
        = (1/sum) * exp(-beta*x_i)
        
        So the gradient is simply the softmax weights!
        w_i = exp(-beta*x_i) / sum_j exp(-beta*x_j)
        grad f = w
        
        Wait, this is for f(x) = -1/beta * log(sum(exp(-beta*x))).
        Our objective is -r ≈ f(constraints).
        So dObj/dx_i = sum_k (df/dconstraint_k) * (dconstraint_k/dx_i)
        df/dconstraint_k = w_k (the softmax weight)
        
        So we need to compute the gradient of the constraints with respect to x,
        weighted by the softmax weights.
        """
        points = x.reshape(-1, 3)
        
        # Wall distances
        wall_dists = 0.5 - np.abs(points)
        flat_walls = wall_dists.flatten()
        
        # Pairwise distances / 2
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_constraints = dists / 2.0
        iu = np.triu_indices(N, k=1)
        flat_pairs = pair_constraints[iu]
        
        all_constraints = np.concatenate([flat_walls, flat_pairs])
        
        # Compute softmax weights
        m = np.min(all_constraints)
        shifted = all_constraints - m
        exp_vals = np.exp(-beta * shifted)
        sum_exp = np.sum(exp_vals)
        weights = exp_vals / sum_exp
        
        # Gradient of objective w.r.t constraints
        grad_constraints = weights
        
        # Gradient of constraints w.r.t x
        grad_x = np.zeros_like(points)
        
        # 1. Wall gradients
        # wall_dists = 0.5 - |points|
        # d/dx (0.5 - |x|) = -sign(x)
        # flat_walls corresponds to points.flatten()
        # points is (N, 3)
        # signs = -np.sign(points)
        # grad_x += signs * grad_constraints[:N*3].reshape(N, 3)
        
        # Note: at x=0, sign is 0. This is a kink, but softmin smooths it.
        # For numerical stability, we can use a small epsilon in sign or just sign.
        signs = -np.sign(points)
        # Handle zeros: if point is exactly 0, derivative is undefined.
        # But softmin weights are spread out.
        # Let's just use sign.
        grad_x += signs * grad_constraints[:N*3].reshape(N, 3)
        
        # 2. Pair gradients
        # constraint_k = dist(i, j) / 2
        # d/dx_i (dist(i, j)/2) = (x_i - x_j) / dist(i, j)
        # d/dx_j (dist(i, j)/2) = (x_j - x_i) / dist(i, j)
        
        # We need to accumulate gradients for each pair
        # iu[0] are i indices, iu[1] are j indices
        # weights are for flat_pairs
        
        # dists is (N, N)
        # We need dists[i, j] for each pair in iu
        # Let's create a zero matrix for pair gradients and fill it?
        # Or just loop? N=50, pairs=1225. Looping 1225 times is fast in Python?
        # 1225 * 3 ops. Yes, very fast.
        # But we can vectorize.
        
        # Let's vectorize the accumulation.
        # For each pair (i, j), weight w
        # grad_x[i] += w * (x_i - x_j) / dist(i, j)
        # grad_x[j] += w * (x_j - x_i) / dist(i, j)
        
        # diff is (N, N, 3). diff[i, j] = x_i - x_j
        # dists is (N, N)
        
        # We only care about upper triangle pairs.
        # Let's create a weight matrix W (N, N) with zeros
        W = np.zeros((N, N))
        W[iu] = grad_constraints[N*3:]
        W = W + W.T # Symmetric
        
        # grad_x[i] = sum_j W[i, j] * (x_i - x_j) / dist(i, j)
        # Let's compute term = W / dists
        # dists has inf on diagonal. W is 0 on diagonal.
        # term = W / dists might have 0/inf = 0. Good.
        
        # But dists is symmetric.
        # term[i, j] = W[i, j] / dists[i, j]
        
        # grad_x = sum_j term[i, j] * diff[i, j]
        # diff[i, j] = x_i - x_j
        
        # Let's compute term carefully.
        # Where dists is 0 (diagonal), term should be 0.
        # Where dists is inf, term is 0.
        
        # Use np.divide with where
        term = np.zeros_like(dists)
        np.divide(W, dists, out=term, where=(dists > 1e-12))
        
        # Now grad_x[i] = sum_j term[i, j] * diff[i, j]
        # diff is (N, N, 3)
        # term is (N, N)
        # grad_x = np.sum(term[:, :, np.newaxis] * diff, axis=1)
        
        grad_x += np.sum(term[:, :, np.newaxis] * diff, axis=1)
        
        return grad_x.flatten()

    # Initializations
    best_radius = 0.0
    best_points = None
    
    # 1. Grid initialization
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # We have 64 points, need 50.
    # Randomly select 50? Or specific ones?
    # Let's try a few different initializations.
    
    initializations = []
    
    # Init 1: First 50 of grid
    initializations.append(grid[:50].copy())
    
    # Init 2: Random 50 from grid
    idx = np.random.choice(len(grid), 50, replace=False)
    initializations.append(grid[idx].copy())
    
    # Init 3: Jittered random 50 from grid
    idx = np.random.choice(len(grid), 50, replace=False)
    pts = grid[idx].copy()
    pts += np.random.uniform(-0.05, 0.05, pts.shape)
    pts = np.clip(pts, -0.49, 0.49)
    initializations.append(pts)
    
    # Init 4: Random points in cube
    pts = np.random.uniform(-0.45, 0.45, (50, 3))
    initializations.append(pts)
    
    # Optimization Loop
    for init_idx, init_pts in enumerate(initializations):
        if time.time() > deadline:
            break
            
        # Start with a moderate beta
        beta = 50.0
        x = init_pts.flatten()
        
        # L-BFGS-B
        # We want to maximize r, so minimize -r.
        # Our objective is smooth -r.
        
        # Run L-BFGS-B
        # Bounds: -0.5 to 0.5
        bounds = [(-0.5, 0.5)] * (N * 3)
        
        # We can run multiple stages with increasing beta
        for stage in range(5):
            if time.time() > deadline:
                break
                
            res = minimize(
                softmin_objective, 
                x, 
                jac=softmin_gradient, 
                args=(beta,), 
                method='L-BFGS-B', 
                bounds=bounds, 
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            x = res.x
            points = x.reshape(-1, 3)
            r = compute_radius(points)
            
            if r > best_radius:
                best_radius = r
                best_points = points.copy()
            
            # Increase beta for sharper approximation
            beta *= 2.0
            beta = min(beta, 500.0) # Cap beta to avoid numerical issues
            
        # After L-BFGS-B, do a local search polish with exact radius
        # This helps with the final "min" kinks
        if time.time() < deadline - 10:
            points = best_points.copy() if best_points is not None else points
            step_size = 0.01
            for i in range(100):
                if time.time() > deadline - 5:
                    break
                    
                r = compute_radius(points)
                if r > best_radius:
                    best_radius = r
                    best_points = points.copy()
                    step_size *= 0.95
                else:
                    step_size *= 0.9
                    
                # Compute gradient of exact radius? No, it's non-differentiable.
                # Use finite differences or just random kick?
                # Let's use a simple gradient ascent on the "active" constraints.
                # Find the min wall and min pair.
                wall_dists = 0.5 - np.abs(points)
                min_wall_idx = np.argmin(wall_dists)
                min_wall_point = min_wall_idx // 3
                min_wall_dim = min_wall_idx % 3
                
                diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff * diff, axis=2))
                np.fill_diagonal(dists, np.inf)
                min_pair_idx = np.argmin(dists)
                min_pair_i = min_pair_idx // N
                min_pair_j = min_pair_idx % N
                
                # Move points to increase these distances
                # For wall: move away from wall
                p = points[min_wall_point]
                if p[min_wall_dim] > 0:
                    p[min_wall_dim] -= step_size
                else:
                    p[min_wall_dim] += step_size
                
                # For pair: move apart
                d = points[min_pair_i] - points[min_pair_j]
                norm = np.linalg.norm(d)
                if norm > 1e-12:
                    d = d / norm
                    points[min_pair_i] += d * step_size
                    points[min_pair_j] -= d * step_size
                
                # Clip
                points = np.clip(points, -0.4999, 0.4999)
                
                if step_size < 1e-5:
                    break

    # Final check
    if best_points is None:
        return grid[:50]
        
    # Ensure validity
    best_points = np.clip(best_points, -0.5, 0.5)
    return best_points

# For testing / direct call
if __name__ == "__main__":
    pts = construct()
    print("Shape:", pts.shape)
    # Compute radius
    wall_dists = 0.5 - np.abs(pts)
    min_wall = np.min(wall_dists)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    r = min(min_wall, min_pair / 2.0)
    print("Radius:", r)
