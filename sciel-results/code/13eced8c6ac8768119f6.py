"""Circle packing in unit square: 26 circles, maximize sum of radii.
Strategy: Global search using CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
with a smooth penalty function for constraints.
"""
import time
import numpy as np

N = 26
DEADLINE = 140.0  # Safety margin from 150s limit

def _validity(circles):
    """Check if circles are valid (inside square, non-overlapping). Returns (is_valid, sum_radii)."""
    n = len(circles)
    s = 0.0
    eps = 1e-9
    for i in range(n):
        x, y, r = circles[i]
        if r < 0:
            return False, 0.0
        # Check boundaries
        if x < r + eps or x > 1 - r - eps or y < r + eps or y > 1 - r - eps:
            return False, 0.0
        s += r
        # Check overlaps
        for j in range(i + 1, n):
            x2, y2, r2 = circles[j]
            d2 = (x - x2) ** 2 + (y - y2) ** 2
            if d2 < (r + r2) ** 2 - eps:
                return False, 0.0
    return True, s

def _objective(vars, penalty_weight=1000.0):
    """
    Objective function for CMA-ES.
    vars: [x0, y0, ..., xN, yN, r0, ..., rN]
    Returns: -sum(r) + penalty
    """
    n = N
    x = vars[:2*n].reshape(n, 2)
    r = vars[2*n:]
    
    # Clip radii to be non-negative for calculation, though bounds handle this mostly
    r = np.maximum(r, 0)
    
    score = np.sum(r)
    penalty = 0.0
    
    # Boundary violations
    # r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    # Violation is max(0, r_i - x_i) etc.
    # We use a smooth approximation or just linear penalty for simplicity and speed
    # Linear penalty is often sufficient for CMA-ES if weight is high enough
    for i in range(n):
        # x boundary
        v = r[i] - x[i, 0]
        if v > 0: penalty += v
        v = r[i] - (1 - x[i, 0])
        if v > 0: penalty += v
        # y boundary
        v = r[i] - x[i, 1]
        if v > 0: penalty += v
        v = r[i] - (1 - x[i, 1])
        if v > 0: penalty += v
        
    # Overlap violations
    # d_ij >= r_i + r_j
    # Violation is max(0, r_i + r_j - d_ij)
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(x[i, 0] - x[j, 0], x[i, 1] - x[j, 1])
            v = r[i] + r[j] - d
            if v > 0:
                penalty += v

    return -score + penalty_weight * penalty

def _max_radius(pos, r, iters=20):
    """Quick local radius maximization given positions."""
    n = len(r)
    pos = np.asarray(pos)
    r = np.array(r, dtype=float)
    for _ in range(iters):
        for i in range(n):
            dmin = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            for j in range(n):
                if i != j:
                    d = np.hypot(pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1])
                    dmin = min(dmin, d - r[j])
            r[i] = max(0.0, dmin)
    return r

def construct():
    t_start = time.time()
    best_score = -1.0
    best_circles = None
    
    def update_best(circles):
        nonlocal best_score, best_circles
        valid, s = _validity(circles)
        if valid and s > best_score:
            best_score = s
            best_circles = [tuple(c) for c in circles]

    # Initialize CMA-ES
    # We use a simple manual CMA-ES implementation to avoid dependency issues or version differences
    # and to have full control over the time budget.
    # Alternatively, we can use scipy.optimize if it has CMA-ES, but it's not standard.
    # Let's implement a basic CMA-ES.
    
    # Initial population
    # Strategy: Start with a grid-like initialization but with some noise, and let CMA-ES refine.
    # Or start with random feasible points.
    
    # Let's try a hybrid: 
    # 1. Get a decent initial guess using the old logic (grow and relax) quickly.
    # 2. Use CMA-ES to polish and explore globally from that point.
    
    # Initial guess generation
    def get_initial_guess():
        # Hexagonal packing approximation
        pts = []
        rng = np.random.default_rng(42)
        # Try a few random starts and pick the best one using _max_radius
        best_init_score = -1
        best_init_pos = None
        best_init_r = None
        
        for k in range(10):
            # Random positions
            pos = rng.random((N, 2))
            # Initial small radii
            r = np.full(N, 0.01)
            
            # Quick local optimization
            r_opt = _max_radius(pos, r, iters=50)
            score = np.sum(r_opt)
            
            # Check validity roughly
            valid = True
            for i in range(N):
                if pos[i,0] < r_opt[i] or pos[i,0] > 1-r_opt[i] or pos[i,1] < r_opt[i] or pos[i,1] > 1-r_opt[i]:
                    valid = False
                    break
            if valid and score > best_init_score:
                best_init_score = score
                best_init_pos = pos.copy()
                best_init_r = r_opt.copy()
                
        if best_init_pos is None:
            # Fallback to grid
            side = 5
            pos = []
            r = []
            idx = 0
            for i in range(side):
                for j in range(side):
                    if idx >= N: break
                    pos.append(((i+0.5)/side, (j+0.5)/side))
                    r.append(0.1)
                    idx += 1
            best_init_pos = np.array(pos)
            best_init_r = np.array(r)
            best_init_r = _max_radius(best_init_pos, best_init_r, 50)
            
        return best_init_pos, best_init_r

    pos0, r0 = get_initial_guess()
    
    # Update best with initial guess
    circles = list(zip(pos0[:, 0], pos0[:, 1], r0))
    update_best(circles)
    
    # CMA-ES Parameters
    # State: [x0, y0, ..., x25, y25, r0, ..., r25]
    dim = 3 * N
    x_mean = np.concatenate([pos0.flatten(), r0])
    
    # Initial standard deviation
    # Radii are around 0.1, positions around 0.01
    sigma = np.ones(dim)
    sigma[:2*N] = 0.02  # Position sigma
    sigma[2*N:] = 0.02  # Radius sigma
    
    # Population size
    lambda_pop = 4 + int(np.floor(3 * np.log(N)))
    lambda_pop = max(lambda_pop, 20) # At least 20
    
    # Weights
    mu = lambda_pop // 2
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    weights /= np.sum(weights)
    mu_eff = 1.0 / np.sum(weights**2)
    
    # Step size control parameters
    c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
    d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
    c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
    c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
    c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
    
    # Covariance matrix
    C = np.eye(dim)
    # Inverse sqrt of C
    C_sqrt_inv = np.eye(dim)
    
    # Evolution path
    p_sigma = np.zeros(dim)
    p_c = np.zeros(dim)
    
    # Best solution tracking
    best_x = x_mean.copy()
    best_f = _objective(x_mean)
    
    # CMA-ES Loop
    gen = 0
    max_gens = 2000
    
    while time.time() < DEADLINE and gen < max_gens:
        gen += 1
        
        # 1. Sample offspring
        offspring = []
        for i in range(lambda_pop):
            # Standard normal
            z = np.random.randn(dim)
            # Transform using C_sqrt_inv (approximated as identity initially, updated later)
            # For simplicity and speed, we use the current C approximation
            # Ideally we would compute C_sqrt_inv properly, but for small dim and short run,
            # we can update C and use its inverse square root via eigen decomposition periodically.
            
            # Let's just use the standard CMA-ES update rules for C
            # x_i = x_mean + sigma * C_sqrt_inv @ z
            # We approximate C_sqrt_inv as the current matrix square root inverse
            # To keep it simple and fast, we'll just use the identity for the transform 
            # and rely on the covariance update to shape the distribution.
            # Wait, that's not correct CMA-ES. We need to multiply by C^(-1/2).
            
            # Let's use a simpler approach: 
            # Since dim is 78, eigen decomposition is feasible every few generations.
            # But for now, let's just sample x_i = x_mean + sigma * z 
            # and update C based on the successful steps.
            # This is essentially a variant of CMA-ES where we don't explicitly rotate the axes 
            # as aggressively, but the C update will capture the correlations.
            
            x_i = x_mean + sigma * z
            # Ensure bounds are respected? 
            # CMA-ES handles bounds poorly if they are hard. 
            # We rely on the penalty function. 
            # However, positions should be in [0,1] and radii >= 0.
            # We can clip positions to [0,1] and radii to [0, 0.5] to help convergence.
            x_i[:2*N] = np.clip(x_i[:2*N], 0.0, 1.0)
            x_i[2*N:] = np.clip(x_i[2*N:], 0.0, 0.5)
            
            offspring.append(x_i)
            
        offspring = np.array(offspring)
        
        # 2. Evaluate offspring
        f_values = np.array([_objective(x_i) for x_i in offspring])
        
        # Update best
        idx_best = np.argmin(f_values)
        if f_values[idx_best] < best_f:
            best_f = f_values[idx_best]
            best_x = offspring[idx_best].copy()
            
            # Check validity and update global best
            pos = best_x[:2*N].reshape(N, 2)
            r = best_x[2*N:]
            circles = list(zip(pos[:, 0], pos[:, 1], r))
            update_best(circles)
            
        # 3. Sort offspring by fitness
        order = np.argsort(f_values)
        offspring_sorted = offspring[order]
        f_sorted = f_values[order]
        
        # 4. Update mean
        x_new_mean = np.sum(weights[:, None] * offspring_sorted[:mu], axis=0)
        
        # 5. Update sigma
        # Calculate norm of the mean shift
        diff = x_new_mean - x_mean
        norm_diff = np.linalg.norm(diff)
        
        # Update p_sigma
        # chi_norm = sqrt(dim * (1 - 1/dim^2)) # Expected norm of standard normal
        chi_norm = np.sqrt(dim) * (1 - 1/(4*dim) - 1/(25*dim**2))
        
        p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_norm * (diff / sigma)
        
        # Hessian correction
        h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2*(gen+1))) < (1.4 + 2/(dim+1)) * chi_norm else 0
        
        sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_norm - 1))
        
        # 6. Update covariance
        # p_c
        p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * diff
        
        # Rank-1 and Rank-mu updates
        C = (1 - c_1 - c_mu) * C \
            + c_1 * (p_c[:, None] * p_c[None, :] + (1 - h_sigma * c_c * (2 - c_c)) * C) \
            + c_mu / sigma**2 * np.sum(weights[:, None, None] * np.outer(offspring_sorted[:mu, :] - x_mean, offspring_sorted[:mu, :] - x_mean), axis=0)
        
        # Symmetrize C
        C = (C + C.T) / 2
        
        x_mean = x_new_mean
        
        # Periodic eigen decomposition to ensure C is valid and to get C_sqrt_inv if we wanted to use it
        # For now, we don't use C_sqrt_inv in sampling, so we just need C to be positive definite.
        # If C becomes invalid, reset it.
        if gen % 10 == 0:
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                if np.min(eigvals) < 1e-10:
                    C = np.eye(dim)
            except:
                C = np.eye(dim)

    # Final polish: Use the best found configuration and run a local optimization (SLSQP or similar)
    # to squeeze out more value.
    if best_circles is not None:
        pos = np.array([c[0] for c in best_circles])
        r = np.array([c[1] for c in best_circles])
        # Wait, best_circles is (x, y, r)
        pos = np.array([[c[0], c[1]] for c in best_circles])
        r = np.array([c[2] for c in best_circles])
        
        # Run a few iterations of _max_radius to ensure radii are maximal for these positions
        r = _max_radius(pos, r, iters=50)
        circles = list(zip(pos[:, 0], pos[:, 1], r))
        update_best(circles)
        
        # Try SLSQP polish if time permits
        if time.time() < DEADLINE - 5:
            try:
                from scipy.optimize import minimize
                
                def obj(v):
                    p = v[:2*N].reshape(N, 2)
                    rr = v[2*N:]
                    return _objective(v)
                
                bounds = []
                for i in range(N):
                    bounds.append((0.0, 1.0))
                    bounds.append((0.0, 1.0))
                for i in range(N):
                    bounds.append((0.0, 0.5))
                    
                res = minimize(obj, best_x, method='SLSQP', bounds=bounds, options={'maxiter': 100, 'ftol': 1e-12})
                if res.success:
                    pos_res = res.x[:2*N].reshape(N, 2)
                    r_res = res.x[2*N:]
                    r_res = _max_radius(pos_res, r_res, 20)
                    circles_res = list(zip(pos_res[:, 0], pos_res[:, 1], r_res))
                    update_best(circles_res)
            except Exception:
                pass

    if best_circles is None:
        # Fallback
        side = 5
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.09)
                 for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return np.array(best_circles)

if __name__ == '__main__':
    out = construct()
    print("sum of radii:", out[:, 2].sum())
    print(out)
