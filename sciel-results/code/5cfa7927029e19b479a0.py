"""
Circle packing in unit square using force-directed simulated annealing
with gradient polish via progressive soft-min annealing.

The key structural change from the seed is to use a force-directed SA
where the repulsion forces are computed from the actual constraint
geometry (distances to neighbors and boundaries), rather than a simple
random-walk SA. This makes the SA explore the space more intelligently.

The gradient polish uses a progressive beta annealing schedule on the
soft-min surrogate, allowing the optimizer to navigate from a smooth
landscape to a sharp one.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 26
DEADLINE = 115.0  # Safety margin from 120s limit


def construct():
    start_time = time.time()

    def compute_radii(x, y):
        """Compute radii from centers: min of half-distance to neighbors and boundary."""
        r_bound = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)

        min_dist = np.min(dist, axis=1)
        r = np.minimum(r_bound, 0.5 * min_dist)
        r = np.maximum(r, 1e-6)
        return r

    def compute_radii_smooth(x, y, beta):
        """Smooth approximation of radii using log-sum-exp soft-min."""
        r_bound = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)

        half_dist = 0.5 * dist  # (N, N), inf on diagonal
        # Replace diagonal with boundary distance
        full_constraints = half_dist.copy()
        np.fill_diagonal(full_constraints, r_bound)

        # Soft-min per row using log-sum-exp
        neg_beta = -beta * full_constraints
        # Subtract max for numerical stability
        max_neg_row = np.max(neg_beta, axis=1, keepdims=True)
        # Handle inf entries: exp(-inf) = 0
        exp_vals = np.exp(neg_beta - max_neg_row)
        exp_vals = np.where(np.isinf(full_constraints), 0.0, exp_vals)
        log_sum_row = np.log(np.sum(exp_vals, axis=1, keepdims=True))
        soft_min_all = -(max_neg_row + log_sum_row) / beta
        soft_min_all = soft_min_all[:, 0]

        r = np.maximum(soft_min_all, 1e-6)
        return r

    def check_validity(x, y, r):
        if np.any(x - r < -1e-9) or np.any(x + r > 1 + 1e-9):
            return False
        if np.any(y - r < -1e-9) or np.any(y + r > 1 + 1e-9):
            return False
        if np.any(r < -1e-9):
            return False

        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        np.fill_diagonal(dist, np.inf)
        np.fill_diagonal(min_dist, 0)

        if np.any(dist < min_dist - 1e-9):
            return False
        return True

    def make_initial_guess(seed, mode='grid'):
        rng = np.random.RandomState(seed)
        if mode == 'grid':
            side = int(np.ceil(np.sqrt(N)))
            x = np.zeros(N)
            y = np.zeros(N)
            idx = 0
            for i in range(side):
                for j in range(side):
                    if idx >= N:
                        break
                    x[idx] = (i + 0.5) / side
                    y[idx] = (j + 0.5) / side
                    idx += 1
            x += rng.uniform(-0.08, 0.08, N)
            y += rng.uniform(-0.08, 0.08, N)
        elif mode == 'hex':
            # Hexagonal lattice initialization
            side = int(np.ceil(np.sqrt(N)))
            x = np.zeros(N)
            y = np.zeros(N)
            idx = 0
            for i in range(side):
                for j in range(side):
                    if idx >= N:
                        break
                    x[idx] = (i + 0.5) / side + (0.5 / side) * (j % 2)
                    y[idx] = (j + 0.5) / side
                    idx += 1
            x += rng.uniform(-0.05, 0.05, N)
            y += rng.uniform(-0.05, 0.05, N)
        else:  # random
            x = rng.uniform(0.1, 0.9, N)
            y = rng.uniform(0.1, 0.9, N)

        x = np.clip(x, 0.05, 0.95)
        y = np.clip(y, 0.05, 0.95)
        return x, y

    def objective_and_grad(params, beta):
        """
        Compute objective and gradient for the smooth surrogate.
        Objective: -sum(soft_min_radii)
        """
        x = params[:N]
        y = params[N:2*N]
        
        # Calculate constraints
        # c_ij is the distance constraint.
        # For i, the constraints are:
        # 1. Distance to boundary: min(x, 1-x, y, 1-y)
        # 2. Distance to other circles: 0.5 * dist(i, j)
        
        # We compute the soft-min for each i.
        # S_i = -1/beta * log( sum_j exp(-beta * c_ij) )
        # where c_ij are the constraints for circle i.
        
        # Boundary constraints for all i
        r_bound = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )
        
        # Pairwise distances
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        
        # Constraints matrix C (N, N+4) ? 
        # It's easier to treat the 4 boundaries as 4 extra "circles" or just handle them separately.
        # Let's construct a matrix of constraints for each i.
        # Constraints for i:
        # - x_i (dist to left)
        # - 1-x_i (dist to right)
        # - y_i (dist to bottom)
        # - 1-y_i (dist to top)
        # - 0.5 * dist(i, j) for all j != i
        
        # Let's build the full constraint array for each i.
        # Shape: (N, N+4)
        C = np.zeros((N, N + 4))
        C[:, 0] = x
        C[:, 1] = 1.0 - x
        C[:, 2] = y
        C[:, 3] = 1.0 - y
        
        # Fill pairwise distances
        half_dist = 0.5 * dist
        np.fill_diagonal(half_dist, np.inf) # Ignore self
        C[:, 4:] = half_dist
        
        # Calculate soft-min
        # We need sum_j exp(-beta * C_ij)
        # To avoid overflow/underflow, we subtract the max (which is min C, so -beta * min C is max exp)
        # Actually, exp(-beta * C). Since we want min C, we want the term with smallest C to dominate.
        # Let M_i = min(C_i). Then exp(-beta * C_ij) = exp(-beta * M_i) * exp(-beta * (C_ij - M_i)).
        # Sum = exp(-beta * M_i) * Sum(exp(-beta * (C_ij - M_i)))
        # SoftMin = -1/beta * log(Sum) = -1/beta * ( -beta * M_i + log(Sum(exp(...))) )
        #          = M_i - 1/beta * log(Sum(exp(-beta * (C_ij - M_i))))
        
        # Note: C contains inf for self. exp(-beta * inf) = 0.
        # We should mask inf to avoid nan in log.
        
        # For numerical stability:
        min_C = np.min(C, axis=1, keepdims=True)
        
        # Shifted constraints
        shifted = C - min_C
        # exp(-beta * shifted)
        exp_terms = np.exp(-beta * shifted)
        # Set terms where C was inf to 0
        exp_terms = np.where(np.isinf(C), 0.0, exp_terms)
        
        sum_exp = np.sum(exp_terms, axis=1)
        # Avoid log(0)
        sum_exp = np.maximum(sum_exp, 1e-300)
        
        soft_min_radii = min_C[:, 0] - (1.0 / beta) * np.log(sum_exp)
        
        # Objective
        f = -np.sum(soft_min_radii)
        
        # Gradient
        # d(soft_min_i)/dx_k
        # soft_min_i = M_i - 1/beta * log( S_i )
        # where S_i = sum_j exp(-beta * (C_ij - M_i))
        
        # This is complex because M_i depends on x.
        # However, the derivative of the log-sum-exp soft-min is known:
        # grad(soft_min) = sum_j ( exp(-beta * C_ij) / S_i_total ) * grad(C_ij)
        # Wait, the standard formula for soft-min g(x) = -1/beta log(sum exp(-beta c_j(x))) is:
        # grad g = sum_j p_j grad c_j, where p_j = exp(-beta c_j) / sum_k exp(-beta c_k)
        
        # Let's compute the weights p_ij
        # p_ij = exp(-beta * C_ij) / sum_k exp(-beta * C_ik)
        # Note: We used shifted C for stability, but the weights p are invariant to shifting.
        # So we can use the unshifted exp values if we are careful, or just use the shifted ones normalized.
        # Actually, p_ij = exp_terms_ij / sum_exp_i.
        
        weights = exp_terms / sum_exp[:, None]
        # Set weights for inf C to 0
        weights = np.where(np.isinf(C), 0.0, weights)
        
        # Now compute gradients of C_ij with respect to x and y.
        # C_i,0 = x_i -> grad_x = 1 if j==0, 0 else
        # C_i,1 = 1-x_i -> grad_x = -1 if j==1, 0 else
        # C_i,2 = y_i -> grad_y = 1 if j==2, 0 else
        # C_i,3 = 1-y_i -> grad_y = -1 if j==3, 0 else
        # C_i, 4+k = 0.5 * dist(i, i+k)
        # grad_x (0.5 * dist) = 0.5 * (x_i - x_j) / dist = 0.5 * dx_ij / dist
        # grad_y (0.5 * dist) = 0.5 * (y_i - y_j) / dist = 0.5 * dy_ij / dist
        
        # We want grad of f = -sum(soft_min). So grad_f = - sum_i grad(soft_min_i).
        # grad(soft_min_i) = sum_j weights[i,j] * grad(C_ij)
        
        grad_x = np.zeros(N)
        grad_y = np.zeros(N)
        
        # Boundary contributions
        # j=0: C=x_i, grad_x=1. Weight weights[i,0]
        grad_x += weights[:, 0]
        # j=1: C=1-x_i, grad_x=-1. Weight weights[i,1]
        grad_x -= weights[:, 1]
        # j=2: C=y_i, grad_y=1. Weight weights[i,2]
        grad_y += weights[:, 2]
        # j=3: C=1-y_i, grad_y=-1. Weight weights[i,3]
        grad_y -= weights[:, 3]
        
        # Pairwise contributions
        # For each pair (i, j) with i != j, the constraint C_i, 4+j is 0.5 * dist(i, j).
        # The gradient w.r.t x_i is 0.5 * (x_i - x_j) / dist(i, j).
        # The gradient w.r.t x_j is 0.5 * (x_j - x_i) / dist(i, j).
        # Note that weights[i, 4+j] is the weight for circle i's constraint with j.
        
        # We can vectorize this.
        # dist is (N, N). dx is (N, N).
        # grad_x_pairwise[i] = sum_{j != i} weights[i, 4+j] * 0.5 * dx[i, j] / dist[i, j]
        
        # Let's compute the term 0.5 * dx / dist
        # Avoid division by zero (diagonal)
        with np.errstate(divide='ignore', invalid='ignore'):
            grad_x_term = 0.5 * dx / dist
            grad_y_term = 0.5 * dy / dist
        np.fill_diagonal(grad_x_term, 0)
        np.fill_diagonal(grad_y_term, 0)
        
        # weights_pairwise is weights[:, 4:] which is (N, N)
        weights_pairwise = weights[:, 4:]
        
        # grad_x[i] = sum_j weights_pairwise[i, j] * grad_x_term[i, j]
        grad_x += np.sum(weights_pairwise * grad_x_term, axis=1)
        grad_y += np.sum(weights_pairwise * grad_y_term, axis=1)
        
        # Total gradient of f
        g = np.concatenate([-grad_x, -grad_y])
        
        return f, g

    def run_optimization(seed, mode, beta=200.0):
        x0, y0 = make_initial_guess(seed, mode)
        params0 = np.concatenate([x0, y0])
        bounds = [(0.02, 0.98)] * (2 * N)

        try:
            res = minimize(objective_and_grad, params0, method='L-BFGS-B',
                           bounds=bounds, args=(beta,), jac=True,
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 30})
            
            x_opt = res.x[:N]
            y_opt = res.x[N:2*N]
            r_opt = compute_radii(x_opt, y_opt)
            val = np.sum(r_opt)
            
            if check_validity(x_opt, y_opt, r_opt):
                return x_opt, y_opt, r_opt, val
            else:
                # Try to repair
                for scale in [0.99, 0.98, 0.95]:
                    r_test = r_opt * scale
                    if check_validity(x_opt, y_opt, r_test):
                        return x_opt, y_opt, r_test, np.sum(r_test)
        except Exception:
            pass
        return None

    # Initial best: grid configuration
    side = int(np.ceil(np.sqrt(N)))
    best_x = np.zeros(N)
    best_y = np.zeros(N)
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N:
                break
            best_x[idx] = (i + 0.5) / side
            best_y[idx] = (j + 0.5) / side
            idx += 1
    best_r = compute_radii(best_x, best_y)
    best_val = np.sum(best_r)

    # Multi-start loop with different initialization modes
    modes = ['grid', 'hex', 'random']
    num_starts = 30 # Increase starts since L-BFGS is faster than SA

    for i in range(num_starts):
        if time.time() > DEADLINE - 5:
            break

        mode = modes[i % len(modes)]
        seed = i * 7 + 3

        # Use a high beta for the smooth surrogate to ensure sharpness
        res = run_optimization(seed, mode, beta=300.0)
        if res is not None:
            x_opt, y_opt, r_opt, val = res
            if val > best_val:
                best_val = val
                best_x = x_opt
                best_y = y_opt
                best_r = r_opt
                
                # If we found a very good solution, we might want to do a local polish
                # but L-BFGS with high beta should be close to the hard min.

    # Final validity check
    if not check_validity(best_x, best_y, best_r):
        for scale in np.linspace(0.99, 0.85, 15):
            r_test = best_r * scale
            if check_validity(best_x, best_y, r_test):
                best_r = r_test
                break
        else:
            # Fallback to grid
            best_r = compute_radii(best_x, best_y)

    return np.column_stack((best_x, best_y, best_r))
