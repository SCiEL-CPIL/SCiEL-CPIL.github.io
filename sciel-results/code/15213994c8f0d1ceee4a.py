import numpy as np
from scipy.optimize import differential_evolution, minimize
import time

N = 26
DEADLINE = 140.0  # Safety margin from 150s limit

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
        r = np.maximum(r, 1e-8)
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

    def score_config(x, y):
        r = compute_radii(x, y)
        if check_validity(x, y, r):
            return np.sum(r)
        return -np.inf

    def objective(params):
        """
        Objective function for Differential Evolution.
        We want to maximize sum of radii. DE minimizes, so we return -sum(r) + penalty.
        Penalty is large if configuration is invalid.
        """
        x = params[:N]
        y = params[N:2*N]
        
        # Check boundaries
        if np.any(x < 0) or np.any(x > 1) or np.any(y < 0) or np.any(y > 1):
            return 1e6
            
        r = compute_radii(x, y)
        
        # Check overlaps
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        np.fill_diagonal(dist, np.inf)
        np.fill_diagonal(min_dist, 0)
        
        # Penalty for overlaps: sum of (min_dist - dist) where min_dist > dist
        overlaps = np.maximum(0, min_dist - dist)
        penalty = np.sum(overlaps) * 1000
        
        return -np.sum(r) + penalty

    # Bounds for DE: [0, 1] for all coordinates
    bounds = [(0, 1)] * (2 * N)
    
    best_score = -np.inf
    best_x = None
    best_y = None
    best_r = None

    # Run multiple Differential Evolution starts
    # DE is stochastic, so multiple runs with different seeds help
    n_starts = 5
    
    for seed in range(n_starts):
        if time.time() - start_time > DEADLINE - 20:
            break
            
        try:
            # Use DE to find a good configuration
            # maxiter is limited to ensure we finish in time
            # popsize 15 is a reasonable default
            res = differential_evolution(
                objective, 
                bounds, 
                seed=seed,
                maxiter=200,
                popsize=15,
                tol=1e-4,
                polish=True  # Use L-BFGS to polish the result
            )
            
            x_de = res.x[:N]
            y_de = res.x[N:]
            
            # Compute radii and check validity
            r_de = compute_radii(x_de, y_de)
            if check_validity(x_de, y_de, r_de):
                score_de = np.sum(r_de)
                if score_de > best_score:
                    best_score = score_de
                    best_x = x_de.copy()
                    best_y = y_de.copy()
                    best_r = r_de.copy()
        except Exception:
            pass

    # Fallback if DE failed
    if best_x is None:
        # Simple grid initialization
        x = []
        y = []
        for i in range(5):
            for j in range(5):
                x.append((j + 0.5) / 5.0)
                y.append((i + 0.5) / 5.0)
        x.append(0.5)
        y.append(0.5)
        x = np.array(x)
        y = np.array(y)
        best_r = compute_radii(x, y)
        best_x = x
        best_y = y
        best_score = np.sum(best_r)

    # Final polish with L-BFGS on the best solution found by DE
    if time.time() - start_time < DEADLINE - 5 and best_x is not None:
        
        def soft_objective_and_grad(params, beta):
            x = params[:N]
            y = params[N:2*N]

            b_x = np.minimum(x, 1.0 - x)
            b_y = np.minimum(y, 1.0 - y)
            r_bound = np.minimum(b_x, b_y)

            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dist = np.sqrt(dx**2 + dy**2 + 1e-30)
            np.fill_diagonal(dist, 1e10)

            c = 0.5 * dist
            np.fill_diagonal(c, r_bound)

            neg_beta_c = -beta * c
            max_neg = np.max(neg_beta_c, axis=1, keepdims=True)
            exp_terms = np.exp(neg_beta_c - max_neg)
            sum_exp = np.sum(exp_terms, axis=1, keepdims=True)
            log_sum = np.log(sum_exp)
            r = -(max_neg + log_sum) / beta
            r = r[:, 0]

            w = exp_terms / sum_exp

            dist_safe = dist.copy()
            np.fill_diagonal(dist_safe, 1.0)

            dc_dx_i = 0.5 * dx / dist_safe
            dc_dy_i = 0.5 * dy / dist_safe
            dc_dx_j = -0.5 * dx / dist_safe
            dc_dy_j = -0.5 * dy / dist_safe

            beta_b = 200.0
            bounds_4 = np.stack([x, 1.0 - x, y, 1.0 - y], axis=1)
            neg_bb = -beta_b * bounds_4
            max_bb = np.max(neg_bb, axis=1, keepdims=True)
            exp_bb = np.exp(neg_bb - max_bb)
            sum_bb = np.sum(exp_bb, axis=1, keepdims=True)
            w_b = exp_bb / sum_bb

            grad_rb_dx = w_b[:, 0] - w_b[:, 1]
            grad_rb_dy = w_b[:, 2] - w_b[:, 3]

            np.fill_diagonal(dc_dx_i, grad_rb_dx)
            np.fill_diagonal(dc_dy_i, grad_rb_dy)
            np.fill_diagonal(dc_dx_j, grad_rb_dx)
            np.fill_diagonal(dc_dy_j, grad_rb_dy)

            w_dc_dx_j = w * dc_dx_j
            w_dc_dx_i = w * dc_dx_i
            w_dc_dy_j = w * dc_dy_j
            w_dc_dy_i = w * dc_dy_i

            term1_dx = np.sum(w_dc_dx_j, axis=0)
            term2_dx = np.sum(w_dc_dx_i, axis=1) - np.diag(w) * np.diag(dc_dx_i)
            dr_dx = term1_dx + term2_dx

            term1_dy = np.sum(w_dc_dy_j, axis=0)
            term2_dy = np.sum(w_dc_dy_i, axis=1) - np.diag(w) * np.diag(dc_dy_i)
            dr_dy = term1_dy + term2_dy

            grad = np.concatenate([-dr_dx, -dr_dy])
            obj_val = -np.sum(r)
            return obj_val, grad

        params0 = np.concatenate([best_x, best_y])
        bounds_opt = [(0.005, 0.995)] * (2 * N)
        
        try:
            # Use a high beta for precise polishing
            res = minimize(soft_objective_and_grad, params0,
                           method='L-BFGS-B', bounds=bounds_opt,
                           args=(1000.0,),
                           jac=True,
                           options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-13})
            
            x_polish = res.x[:N]
            y_polish = res.x[N:]
            r_polish = compute_radii(x_polish, y_polish)
            
            if check_validity(x_polish, y_polish, r_polish):
                score_polish = np.sum(r_polish)
                if score_polish > best_score:
                    best_score = score_polish
                    best_x = x_polish
                    best_y = y_polish
                    best_r = r_polish
        except Exception:
            pass

    # Ensure validity one last time
    if not check_validity(best_x, best_y, best_r):
        for scale in np.linspace(0.999, 0.95, 20):
            r_test = best_r * scale
            if check_validity(best_x, best_y, r_test):
                best_r = r_test
                break
        else:
            best_r = compute_radii(best_x, best_y)

    return np.column_stack((best_x, best_y, best_r))
