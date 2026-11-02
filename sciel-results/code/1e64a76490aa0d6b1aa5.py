"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 32

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    def _elapsed():
        return time.time() - start_time

    def _is_valid(c, r):
        n = len(c)
        for i in range(n):
            if c[i][0] < -1e-9 or c[i][0] > 1 + 1e-9 or c[i][1] < -1e-9 or c[i][1] > 1 + 1e-9:
                return False
            if r[i] < -1e-9:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(c[i] - c[j])
                if d < r[i] + r[j] - 1e-9:
                    return False
        for i in range(n):
            if c[i][0] < r[i] - 1e-9 or c[i][0] > 1 - r[i] + 1e-9:
                return False
            if c[i][1] < r[i] - 1e-9 or c[i][1] > 1 - r[i] + 1e-9:
                return False
        return True

    def _sum_r(c, r):
        return float(np.sum(r))

    def _make_valid(c, r):
        """Push circles apart and clip to ensure validity."""
        c = c.copy()
        r = r.copy()
        # Clip centers
        for i in range(len(c)):
            c[i][0] = np.clip(c[i][0], r[i], 1 - r[i])
            c[i][1] = np.clip(c[i][1], r[i], 1 - r[i])
        # Iterative push-apart
        for _ in range(200):
            moved = False
            for i in range(len(c)):
                for j in range(i + 1, len(c)):
                    d_vec = c[i] - c[j]
                    d = np.linalg.norm(d_vec)
                    min_d = r[i] + r[j]
                    if d < min_d - 1e-12:
                        if d < 1e-12:
                            d_vec = np.array([1e-6, 0.0])
                            d = 1e-6
                        push = (min_d - d) / d * 0.5
                        c[i] += d_vec * push
                        c[j] -= d_vec * push
                        moved = True
            # Re-clip
            for i in range(len(c)):
                c[i][0] = np.clip(c[i][0], r[i], 1 - r[i])
                c[i][1] = np.clip(c[i][1], r[i], 1 - r[i])
            if not moved:
                break
        return c, r

    def _compute_radii(c):
        n = len(c)
        radii = np.zeros(n)
        for i in range(n):
            r_wall = min(c[i][0], 1 - c[i][0], c[i][1], 1 - c[i][1])
            min_d = r_wall
            for j in range(n):
                if i != j:
                    d = np.linalg.norm(c[i] - c[j])
                    min_d = min(min_d, d / 2.0)
            radii[i] = min_d
        return radii

    def _hex_init(n):
        """Hexagonal packing initialization."""
        centers = np.zeros((n, 2))
        # Hexagonal lattice: rows offset
        cols = int(math.ceil(math.sqrt(n * 2.0 / math.sqrt(3))))
        row_h = 1.0 / (cols * math.sqrt(3) / 2 + 1)
        col_w = 1.0 / (cols + 0.5)
        idx = 0
        for row in range(cols + 1):
            if idx >= n:
                break
            x_off = (row % 2) * col_w / 2
            for col in range(cols + 1):
                if idx >= n:
                    break
                x = x_off + col * col_w
                y = row * row_h * math.sqrt(3) / 2 + row_h / 2
                if x <= 1.0 and y <= 1.0:
                    centers[idx] = [x, y]
                    idx += 1
        # Fill remaining with random
        while idx < n:
            centers[idx] = np.random.uniform(0.1, 0.9, 2)
            idx += 1
        return centers

    def _rect_init(n):
        """Rectangular grid initialization."""
        rows = int(math.ceil(math.sqrt(n)))
        cols = int(math.ceil(n / rows))
        centers = np.zeros((n, 2))
        for i in range(n):
            r = i // cols
            c = i % cols
            centers[i][0] = (c + 0.5) / cols
            centers[i][1] = (r + 0.5) / rows
        return centers

    def _random_init(n, seed):
        np.random.seed(seed)
        return np.random.uniform(0.05, 0.95, (n, 2))

    def _objective(c, penalty=500.0):
        """Smooth penalty objective: minimize -sum(r) + penalty * overlap_penalty."""
        n = len(c)
        r = _compute_radii(c)
        # Penalty for any constraint violation (should be zero if r computed correctly)
        # The radii are defined as the max valid radius, so no penalty needed
        return -np.sum(r)

    def _objective_and_grad(c, penalty=500.0):
        """Compute objective and gradient using smooth min approximation."""
        n = len(c)
        c = np.clip(c, 1e-8, 1 - 1e-8)
        
        radii = np.zeros(n)
        grad = np.zeros((n, 2))
        
        for i in range(n):
            # Collect all constraints for circle i
            # Wall constraints: c[i][0] >= 0, c[i][0] <= 1, c[i][1] >= 0, c[i][1] <= 1
            # Pair constraints: ||c[i] - c[j]|| >= r_i + r_j -> r_i <= ||c[i]-c[j]||/2
            
            # For each circle i, r_i = min(wall_dist, min_j ||c[i]-c[j]||/2)
            # We use a soft-min to get smooth gradients
            
            constraints = []
            # Wall distances
            constraints.append(c[i][0])
            constraints.append(1 - c[i][0])
            constraints.append(c[i][1])
            constraints.append(1 - c[i][1])
            
            for j in range(n):
                if i != j:
                    d = np.linalg.norm(c[i] - c[j])
                    constraints.append(d / 2.0)
            
            constraints = np.array(constraints)
            
            # Soft min with beta
            beta = 50.0
            m = np.min(constraints)
            # log-sum-exp for soft min: soft_min = m - log(sum(exp(beta*(x-m))))/beta
            log_sum = np.log(np.sum(np.exp(beta * (constraints - m))))
            soft_min_val = m - log_sum / beta
            
            # Gradient of soft_min w.r.t. each constraint:
            # d(soft_min)/d(constraint_k) = -exp(beta*(x_k - m)) / sum(exp(beta*(x_k - m)))
            exps = np.exp(beta * (constraints - m))
            weights = exps / np.sum(exps)
            
            # Now propagate gradient to c[i]
            # constraint 0: c[i][0] -> d/dc[i][0] = weights[0]
            # constraint 1: 1-c[i][0] -> d/dc[i][0] = -weights[1]
            # constraint 2: c[i][1] -> d/dc[i][1] = weights[2]
            # constraint 3: 1-c[i][1] -> d/dc[i][1] = -weights[3]
            # constraint 4+: ||c[i]-c[j]||/2 for j=0..n-1 (j!=i)
            # d(||c[i]-c[j]||/2)/dc[i] = (c[i]-c[j])/(2*||c[i]-c[j]||)
            
            grad_i = np.zeros(2)
            grad_i[0] += weights[0]
            grad_i[0] -= weights[1]
            grad_i[1] += weights[2]
            grad_i[1] -= weights[3]
            
            k = 4
            for j in range(n):
                if i != j:
                    d_vec = c[i] - c[j]
                    d = np.linalg.norm(d_vec)
                    if d > 1e-12:
                        grad_i += weights[k] * d_vec / (2 * d)
                    k += 1
            
            radii[i] = soft_min_val
            grad[i] = grad_i
        
        f = -np.sum(radii)
        # Gradient of f = -sum(radii) w.r.t. c is -grad
        grad_f = -grad
        
        return f, grad_f

    def _push_apart(c, r, iters=100):
        """Push overlapping circles apart."""
        c = c.copy()
        n = len(c)
        for _ in range(iters):
            moved = False
            for i in range(n):
                for j in range(i + 1, n):
                    d_vec = c[i] - c[j]
                    d = np.linalg.norm(d_vec)
                    min_d = r[i] + r[j]
                    if d < min_d - 1e-12:
                        if d < 1e-12:
                            d_vec = np.array([1e-6, 0.0])
                            d = 1e-6
                        push = (min_d - d) / d * 0.5
                        c[i] += d_vec * push
                        c[j] -= d_vec * push
                        moved = True
            # Clip to square
            for i in range(n):
                c[i][0] = np.clip(c[i][0], r[i], 1 - r[i])
                c[i][1] = np.clip(c[i][1], r[i], 1 - r[i])
            if not moved:
                break
        return c

    def _optimize(c0, max_time=None):
        """Run L-BFGS-B optimization from initial point c0."""
        from scipy.optimize import minimize
        
        if max_time is None:
            max_time = deadline - _elapsed() - 3.0
        if max_time < 2.0:
            return c0, _compute_radii(c0)
        
        best_c = c0.copy()
        best_r = _compute_radii(c0)
        best_sum = np.sum(best_r)
        
        def obj_func(x):
            c = x.reshape(len(c0), 2)
            f, _ = _objective_and_grad(c)
            return f
        
        def jac_func(x):
            c = x.reshape(len(c0), 2)
            _, g = _objective_and_grad(c)
            return g.flatten()
        
        try:
            result = minimize(
                obj_func,
                c0.flatten(),
                jac=jac_func,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if result.fun < -best_sum + 1e-10:
                c_new = np.clip(result.x.reshape(len(c0), 2), 1e-8, 1 - 1e-8)
                r_new = _compute_radii(c_new)
                if np.sum(r_new) > best_sum:
                    best_c = c_new
                    best_r = r_new
                    best_sum = np.sum(r_new)
        except Exception:
            pass
        
        return best_c, best_r

    def _tight_polish(c, r, max_time=None):
        """High-precision local polish."""
        if max_time is None:
            max_time = deadline - _elapsed() - 2.0
        if max_time < 1.0:
            return c, r
        
        n = len(c)
        best_c = c.copy()
        best_r = r.copy()
        best_sum = np.sum(best_r)
        
        # Coordinate descent with fine steps
        for iteration in range(50):
            if time.time() > start_time + max_time:
                break
            
            improved = False
            for i in range(n):
                best_pos = best_c[i].copy()
                best_r_i = best_r[i]
                
                # Try small perturbations
                step_sizes = [0.001, 0.0005, 0.0001]
                for ss in step_sizes:
                    for dx in [-ss, 0, ss]:
                        for dy in [-ss, 0, ss]:
                            if dx == 0 and dy == 0:
                                continue
                            new_pos = best_c[i] + np.array([dx, dy])
                            if new_pos[0] < best_r[i] or new_pos[0] > 1 - best_r[i]:
                                continue
                            if new_pos[1] < best_r[i] or new_pos[1] > 1 - best_r[i]:
                                continue
                            
                            new_c = best_c.copy()
                            new_c[i] = new_pos
                            r_new_i = min(
                                new_pos[0], 1 - new_pos[0], new_pos[1], 1 - new_pos[1]
                            )
                            for j in range(n):
                                if i != j:
                                    d = np.linalg.norm(new_pos - new_c[j])
                                    r_new_i = min(r_new_i, d / 2.0)
                            
                            if r_new_i > best_r_i + 1e-12:
                                best_r_i = r_new_i
                                best_pos = new_pos.copy()
                                improved = True
            
            if improved:
                best_c[i] = best_pos
                best_r = _compute_radii(best_c)
                if np.sum(best_r) > best_sum:
                    best_sum = np.sum(best_r)
        
        return best_c, best_r

    # Main optimization loop
    best_centers = None
    best_radii = None
    best_sum = -1e10
    
    np.random.seed(42)
    
    # Generate diverse initializations
    inits = []
    inits.append(_rect_init(N))
    inits.append(_hex_init(N))
    for seed in [1, 2, 3, 4, 5]:
        inits.append(_random_init(N, seed))
    
    # Time budget per initialization
    n_inits = len(inits)
    time_per_init = (deadline - start_time - 10.0) / n_inits
    
    for init_idx, c0 in enumerate(inits):
        if time.time() > deadline - 10.0:
            break
        
        c0 = np.clip(c0, 0.01, 0.99)
        
        # Compute initial radii and make valid
        r0 = _compute_radii(c0)
        c0, r0 = _make_valid(c0, r0)
        
        # Optimize
        c_opt, r_opt = _optimize(c0, max_time=time_per_init - 5.0)
        
        if np.sum(r_opt) > best_sum:
            best_sum = np.sum(r_opt)
            best_centers = c_opt.copy()
            best_radii = r_opt.copy()
    
    # Final tight polish on the best solution
    if best_centers is not None and time.time() < deadline - 5.0:
        best_centers, best_radii = _tight_polish(best_centers, best_radii, max_time=deadline - time.time() - 3.0)
    
    # Ensure validity
    if best_centers is not None:
        best_centers, best_radii = _make_valid(best_centers, best_radii)
        final_sum = np.sum(best_radii)
    else:
        # Fallback: simple grid
        best_centers = _rect_init(N)
        best_radii = _compute_radii(best_centers)
        best_centers, best_radii = _make_valid(best_centers, best_radii)
        final_sum = np.sum(best_radii)
    
    # Build result
    result = np.zeros((N, 3))
    for i in range(N):
        result[i] = [best_centers[i][0], best_centers[i][1], best_radii[i]]
    
    return result
# EVOLVE-BLOCK-END
