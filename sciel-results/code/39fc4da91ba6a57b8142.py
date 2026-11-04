import numpy as np
from scipy.optimize import minimize
import time

N = 26
DEADLINE = 145.0  # Safety margin from 150s limit

def construct():
    start_time = time.time()
    
    def _elapsed():
        return time.time() - start_time

    def _remaining():
        return DEADLINE - _elapsed()

    def _soft_objective(params, beta=100.0):
        """
        Objective function for smooth optimization.
        Minimizes -sum(r) where r is the soft-min of distances to boundaries and neighbors.
        """
        x = params[:N]
        y = params[N:2*N]
        
        # Boundary distances
        r_bound = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        
        # Pairwise distances
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)
        
        # Soft min radius for each circle
        # r_i = soft_min(0.5 * dist(i,j), r_bound_i)
        # We want to maximize sum(r_i), so minimize -sum(r_i)
        
        # Construct matrix of constraints for each i
        # c_ij = 0.5 * dist(i,j) for j != i
        # c_ii = r_bound_i
        c = 0.5 * dist
        c[np.arange(N), np.arange(N)] = r_bound
        
        # Soft min: -1/beta * log(sum(exp(-beta * c)))
        # To avoid overflow, subtract max
        neg_beta_c = -beta * c
        max_neg = np.max(neg_beta_c, axis=1, keepdims=True)
        exp_terms = np.exp(neg_beta_c - max_neg)
        sum_exp = np.sum(exp_terms, axis=1, keepdims=True)
        log_sum = np.log(sum_exp)
        r = -(max_neg + log_sum) / beta
        r = r.flatten()
        
        # Objective: -sum(r)
        return -np.sum(r)
        
    def _soft_gradient(params, beta=100.0):
        """
        Gradient of the soft objective.
        """
        x = params[:N]
        y = params[N:2*N]
        
        r_bound = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2 + 1e-12)
        np.fill_diagonal(dist, 1.0) # Avoid div by zero in grad calc, will mask later
        
        c = 0.5 * dist
        c[np.arange(N), np.arange(N)] = r_bound
        
        neg_beta_c = -beta * c
        max_neg = np.max(neg_beta_c, axis=1, keepdims=True)
        exp_terms = np.exp(neg_beta_c - max_neg)
        sum_exp = np.sum(exp_terms, axis=1, keepdims=True)
        w = exp_terms / sum_exp  # Weights for soft min
        
        # Gradient of c_ij w.r.t x_i
        # c_ij = 0.5 * dist(i,j)
        # d c_ij / d x_i = 0.5 * (x_i - x_j) / dist(i,j)
        
        dc_dx = 0.5 * dx / dist
        dc_dy = 0.5 * dy / dist
        
        # Diagonal terms: c_ii = r_bound_i
        # grad r_bound_i w.r.t x_i is +1 if x_i < 0.5, -1 if x_i > 0.5 (assuming x is the active bound)
        # We need to be careful with the min. 
        # For soft min gradient, the diagonal weight w_ii applies to the gradient of the diagonal element.
        
        # Approximate diagonal gradient with subgradient of min
        # If x_i is close to 0, grad is +1; if close to 1, grad is -1.
        # We can use a smooth approximation or just the hard subgradient.
        # Let's use the hard subgradient for the boundary part as it's often more stable in this context.
        
        grad_x = np.zeros(N)
        grad_y = np.zeros(N)
        
        # Sum over j for each i
        # grad_x[i] = sum_j w_ij * (d c_ij / d x_i)
        # Note: w is (N,N), dc_dx is (N,N) where row i corresponds to center i.
        
        # We need to sum contributions to grad_x[i] from all j.
        # Contribution from j: w[i,j] * dc_dx[i,j]
        # But wait, dc_dx[i,j] is d c_ij / d x_i.
        # Also, x_j appears in c_ij as -x_j.
        # So grad_x[i] gets contribution from w[i,j]*dc_dx[i,j] AND from w[j,i]*dc_dx[j,i] (where i is the second index).
        
        # Let's vectorize:
        # grad_x[i] = sum_j w[i,j] * (0.5 * (x_i - x_j) / dist) - sum_j w[j,i] * (0.5 * (x_j - x_i) / dist)
        # The second term is for the case where i is the 'j' index in pair (j,i).
        # d c_ji / d x_i = 0.5 * (x_j - x_i) / dist(j,i)
        
        # Diagonal: w[i,i] * d r_bound_i / d x_i
        
        # Off-diagonal x gradient:
        # term1 = np.sum(w * dc_dx, axis=1)
        # term2 = np.sum(w * dc_dx, axis=0) # This sums w[j,i] * dc_dx[j,i] for fixed i.
        # Note dc_dx[j,i] = 0.5 * (x_j - x_i) / dist = -dc_dx[i,j]
        # So grad_x_offdiag[i] = term1[i] - term2[i] ?
        # Let's check:
        # d(-sum r_k)/dx_i.
        # r_k depends on c_kj.
        # If k=i, term is w[i,j] * dc_ij/dx_i.
        # If k!=i, term is w[k,i] * dc_ki/dx_i.
        
        grad_x_off = np.sum(w * dc_dx, axis=1) - np.sum(w * dc_dx, axis=0)
        grad_y_off = np.sum(w * dc_dy, axis=1) - np.sum(w * dc_dy, axis=0)
        
        # Diagonal gradient
        # r_bound = min(x, 1-x, y, 1-y)
        # Use subgradient:
        grad_x_diag = np.zeros(N)
        grad_y_diag = np.zeros(N)
        for i in range(N):
            bounds = [x[i], 1.0-x[i], y[i], 1.0-y[i]]
            idx = np.argmin(bounds)
            if idx == 0: grad_x_diag[i] = 1.0
            elif idx == 1: grad_x_diag[i] = -1.0
            elif idx == 2: grad_y_diag[i] = 1.0
            elif idx == 3: grad_y_diag[i] = -1.0
            
        grad_x_diag = grad_x_diag * np.diag(w)
        grad_y_diag = grad_y_diag * np.diag(w)
        
        total_grad_x = grad_x_off + grad_x_diag
        total_grad_y = grad_y_off + grad_y_diag
        
        # Objective is -sum(r), so gradient is -total_grad
        return -np.concatenate([total_grad_x, total_grad_y])
        
    def _is_valid(x, y, r, tol=1e-9):
        if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
            return False
        if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
            return False
        if np.any(r < -tol):
            return False
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        min_dist = r[:, None] + r[None, :]
        np.fill_diagonal(dist, np.inf)
        np.fill_diagonal(min_dist, 0)
        if np.any(dist < min_dist - tol):
            return False
        return True
        
    def _compute_radii(x, y):
        r_bound = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)
        min_dist = np.min(dist, axis=1)
        r = np.minimum(r_bound, 0.5 * min_dist)
        r = np.maximum(r, 1e-6)
        return r
        
    def _score(x, y, r):
        if _is_valid(x, y, r):
            return np.sum(r)
        return -np.inf
        
    def _fix_and_validate(x, y, r):
        # Scale down radii to make valid if slightly invalid
        if not _is_valid(x, y, r):
            for scale in np.linspace(0.999, 0.95, 20):
                r_test = r * scale
                if _is_valid(x, y, r_test):
                    return x, y, r_test, np.sum(r_test)
            # If still invalid, shrink aggressively
            r = _compute_radii(x, y)
            if not _is_valid(x, y, r):
                r = r * 0.9
            return x, y, r, np.sum(r)
        return x, y, r, np.sum(r)
        
    def _grid_init(seed):
        rng = np.random.RandomState(seed)
        side = int(np.ceil(np.sqrt(N)))
        x = np.zeros(N)
        y = np.zeros(N)
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N: break
                x[idx] = (i + 0.5) / side
                y[idx] = (j + 0.5) / side
                idx += 1
        x += rng.uniform(-0.05, 0.05, N)
        y += rng.uniform(-0.05, 0.05, N)
        return np.clip(x, 0.05, 0.95), np.clip(y, 0.05, 0.95)
        
    def _hex_init(seed):
        rng = np.random.RandomState(seed)
        side = int(np.ceil(np.sqrt(N)))
        x = np.zeros(N)
        y = np.zeros(N)
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N: break
                x[idx] = (i + 0.5) / side + (0.5 / side) * (j % 2)
                y[idx] = (j + 0.5) / side
                idx += 1
        x += rng.uniform(-0.03, 0.03, N)
        y += rng.uniform(-0.03, 0.03, N)
        return np.clip(x, 0.05, 0.95), np.clip(y, 0.05, 0.95)
        
    def _random_init(seed):
        rng = np.random.RandomState(seed)
        x = rng.uniform(0.1, 0.9, N)
        y = rng.uniform(0.1, 0.9, N)
        return x, y
        
    def _force_directed_polish(x, y, max_iters=1000, step=0.01):
        """
        Uses repulsive forces to push circles apart and boundary forces to keep them in.
        Maximizes minimum distance, which helps in packing.
        """
        x = x.copy()
        y = y.copy()
        best_val = -np.inf
        best_x = x.copy()
        best_y = y.copy()
        
        for i in range(max_iters):
            if _remaining() < 5.0:
                break
                
            fx = np.zeros(N)
            fy = np.zeros(N)
            
            # Boundary forces (push in if too close to edge)
            fx += 5.0 * np.clip(0.05 - x, 0, None)
            fx -= 5.0 * np.clip(x - 0.95, 0, None)
            fy += 5.0 * np.clip(0.05 - y, 0, None)
            fy -= 5.0 * np.clip(y - 0.95, 0, None)
            
            # Repulsive forces
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(d, 1e10)
            
            f_mag = 1.0 / (d**2 + 0.001)
            np.fill_diagonal(f_mag, 0)
            
            fx += np.sum(-f_mag * dx / d, axis=1)
            fy += np.sum(-f_mag * dy / d, axis=1)
            
            # Move
            x_new = x + step * fx / (np.max(np.abs(fx)) + 1e-10)
            y_new = y + step * fy / (np.max(np.abs(fy)) + 1e-10)
            x_new = np.clip(x_new, 0.01, 0.99)
            y_new = np.clip(y_new, 0.01, 0.99)
            
            r_new = _compute_radii(x_new, y_new)
            val_new = np.sum(r_new)
            
            if val_new > best_val:
                best_val = val_new
                best_x = x_new.copy()
                best_y = y_new.copy()
                
            x, y = x_new, y_new
            
        return best_x, best_y, _compute_radii(best_x, best_y), best_val
        
    def _local_gradient_polish(x0, y0, beta_start=50, beta_end=500, max_iters=200):
        """
        Run L-BFGS-B on the smooth objective to refine positions.
        """
        params0 = np.concatenate([x0, y0])
        bounds = [(0.01, 0.99)] * (2 * N)
        
        current_params = params0.copy()
        best_val = -np.inf
        best_params = params0.copy()
        
        # Multi-stage beta
        betas = np.geomspace(beta_start, beta_end, 5)
        
        for beta in betas:
            if _remaining() < 2.0:
                break
            try:
                res = minimize(_soft_objective, current_params, method='L-BFGS-B', 
                               jac=lambda p: _soft_gradient(p, beta),
                               args=(beta,),
                               bounds=bounds,
                               options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-10})
                current_params = res.x
                
                x = current_params[:N]
                y = current_params[N:]
                r = _compute_radii(x, y)
                val = _score(x, y, r)
                
                if val > best_val:
                    best_val = val
                    best_params = current_params.copy()
            except Exception:
                pass
                
        x = best_params[:N]
        y = best_params[N:]
        r = _compute_radii(x, y)
        x, y, r, val = _fix_and_validate(x, y, r)
        return x, y, r, val
        
    best_x = np.zeros(N)
    best_y = np.zeros(N)
    best_r = np.zeros(N)
    best_val = 0.0
    
    # Initial best from grid
    x0, y0 = _grid_init(0)
    r0 = _compute_radii(x0, y0)
    best_x, best_y, best_r, best_val = _fix_and_validate(x0, y0, r0)
    
    # Multi-start loop
    modes = ['grid', 'hex', 'random']
    num_starts = 15
    
    for i in range(num_starts):
        if _remaining() < 10.0:
            break
            
        mode = modes[i % len(modes)]
        if mode == 'grid':
            x0, y0 = _grid_init(i)
        elif mode == 'hex':
            x0, y0 = _hex_init(i)
        else:
            x0, y0 = _random_init(i)
            
        # 1. Force directed polish to spread out
        x_fd, y_fd, r_fd, val_fd = _force_directed_polish(x0, y0, max_iters=500, step=0.005)
        
        # 2. Gradient polish
        x_gp, y_gp, r_gp, val_gp = _local_gradient_polish(x_fd, y_fd, beta_start=50, beta_end=200)
        
        # 3. Final force directed tweak with smaller step
        x_final, y_final, r_final, val_final = _force_directed_polish(x_gp, y_gp, max_iters=200, step=0.001)
        
        # 4. One more gradient polish
        x_final, y_final, r_final, val_final = _local_gradient_polish(x_final, y_final, beta_start=200, beta_end=1000)
        
        x_final, y_final, r_final, val_final = _fix_and_validate(x_final, y_final, r_final)
        
        if val_final > best_val:
            best_val = val_final
            best_x = x_final
            best_y = y_final
            best_r = r_final
            
    # Final aggressive polish on the best solution
    if _remaining() > 5.0:
        # Force directed
        x_f, y_f, r_f, v_f = _force_directed_polish(best_x, best_y, max_iters=1000, step=0.002)
        x_f, y_f, r_f, v_f = _local_gradient_polish(x_f, y_f, beta_start=500, beta_end=2000)
        x_f, y_f, r_f, v_f = _force_directed_polish(x_f, y_f, max_iters=500, step=0.0005)
        x_f, y_f, r_f, v_f = _local_gradient_polish(x_f, y_f, beta_start=2000, beta_end=5000)
        
        x_f, y_f, r_f, v_f = _fix_and_validate(x_f, y_f, r_f)
        
        if v_f > best_val:
            best_val = v_f
            best_x = x_f
            best_y = y_f
            best_r = r_f

    # Ensure validity
    best_x, best_y, best_r, best_val = _fix_and_validate(best_x, best_y, best_r)
    
    return np.column_stack((best_x, best_y, best_r))

if __name__ == "__main__":
    res = construct()
    print(res)
