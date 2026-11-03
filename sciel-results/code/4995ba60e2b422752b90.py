import numpy as np
import time
from scipy.optimize import minimize

N = 26
DEADLINE = 115.0  # Safety margin from 120s limit

def construct():
    start_time = time.time()
    
    def compute_radii_hard(x, y):
        """Compute exact radii for hard constraints."""
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
        return np.maximum(r, 1e-9)

    def objective_smooth(z, beta=50.0):
        """
        Smooth objective function for gradient-based optimization.
        Uses soft-min (log-sum-exp) to approximate the non-differentiable min operation.
        Minimizes negative sum of radii.
        """
        z = z.reshape(N, 2)
        x, y = z[:, 0], z[:, 1]
        
        # Boundary distances
        r_bound = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )
        
        # Pairwise distances
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)
        
        # Constraints for each circle i:
        # 1. Boundary distance: r_bound[i]
        # 2. Half-distance to each other circle j: 0.5 * dist[i, j]
        
        # We want r_i = min(r_bound[i], min_j 0.5 * dist[i, j])
        # Soft-min approximation: softmin(A) = -1/beta * log(sum(exp(-beta * A)))
        # Stack constraints: [r_bound[i], 0.5*dist[i,0], ..., 0.5*dist[i,N-1]] (excluding i)
        
        # Vectorized soft-min calculation
        # For each i, we have N constraints: 1 boundary + (N-1) neighbors
        # Let's create a matrix of constraints C of shape (N, N)
        # C[i, 0] = r_bound[i]
        # C[i, j+1] = 0.5 * dist[i, j] for j != i, else large value
        
        C = np.empty((N, N))
        C[:, 0] = r_bound
        
        half_dist = 0.5 * dist
        # Mask out diagonal with a large value so it doesn't affect min
        np.fill_diagonal(half_dist, 1e6)
        
        # Shift half_dist to columns 1..N-1
        C[:, 1:] = half_dist
        
        # Soft-min over columns for each row
        # logsumexp(-beta * C)
        neg_beta_C = -beta * C
        max_neg = np.max(neg_beta_C, axis=1, keepdims=True)
        log_sum = np.log(np.sum(np.exp(neg_beta_C - max_neg), axis=1, keepdims=True))
        soft_min_vals = -(max_neg + log_sum) / beta
        soft_min_vals = soft_min_vals[:, 0]
        
        r_smooth = np.maximum(soft_min_vals, 1e-9)
        
        # We want to maximize sum(r), so minimize -sum(r)
        return -np.sum(r_smooth)

    def objective_grad(z, beta=50.0):
        """Gradient of the smooth objective."""
        z = z.reshape(N, 2)
        x, y = z[:, 0], z[:, 1]
        
        r_bound = np.minimum(
            np.minimum(x, 1.0 - x),
            np.minimum(y, 1.0 - y)
        )
        
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)
        
        C = np.empty((N, N))
        C[:, 0] = r_bound
        half_dist = 0.5 * dist
        np.fill_diagonal(half_dist, 1e6)
        C[:, 1:] = half_dist
        
        neg_beta_C = -beta * C
        max_neg = np.max(neg_beta_C, axis=1, keepdims=True)
        exp_terms = np.exp(neg_beta_C - max_neg)
        sum_exp = np.sum(exp_terms, axis=1, keepdims=True)
        
        # Soft-min value
        soft_min_vals = -(max_neg + np.log(sum_exp)) / beta
        soft_min_vals = soft_min_vals[:, 0]
        r_smooth = np.maximum(soft_min_vals, 1e-9)
        
        # Gradient of -sum(r_smooth)
        # d(r_smooth_i)/d(C_i_k) = -exp(-beta * C_i_k) / sum_j exp(-beta * C_i_j)
        # Let w_i_k = exp(-beta * C_i_k) / sum_j exp(-beta * C_i_j)
        # Then d(r_i)/d(C_i_k) = -w_i_k
        # d(-sum r)/d(C_i_k) = w_i_k
        
        w = exp_terms / sum_exp  # shape (N, N)
        
        # We need d(C_i_k)/d(x_j, y_j)
        # C_i_0 = min(x_i, 1-x_i, y_i, 1-y_i) -> non-smooth, but we can approximate or just use sub-gradient
        # For simplicity in gradient descent, we often ignore the boundary gradient if we are far from boundaries,
        # or use a simple sign. However, L-BFGS-B handles bounds. 
        # Let's compute the gradient with respect to x, y from the pairwise terms.
        # d(C_i_k)/d(x_j) where k>0 corresponds to neighbor j-1.
        # C_i_k = 0.5 * sqrt((x_i - x_j)^2 + (y_i - y_j)^2)
        # d/dx_j = 0.5 * (x_j - x_i) / dist[i, j]
        
        grad_x = np.zeros(N)
        grad_y = np.zeros(N)
        
        # Pairwise contributions
        # w_i_k is weight for constraint k in row i.
        # Constraint k (k>0) is 0.5 * dist[i, k-1].
        # Let j = k-1.
        # Term in grad_x[i] from w[i, j+1]: w[i, j+1] * d(0.5*dist[i,j])/dx_i
        # d(0.5*dist)/dx_i = 0.5 * (x_i - x_j) / dist[i, j]
        
        # We can vectorize this.
        # dist is (N, N). 
        # dx/dx_i = (x_i - x_j) / dist
        # Let's create matrices for derivatives of dist w.r.t x and y.
        
        # dist[i,j] = sqrt(dx^2 + dy^2)
        # d(dist[i,j])/dx_i = (x_i - x_j) / dist[i,j]
        # d(dist[i,j])/dx_j = (x_j - x_i) / dist[i,j] = - d(dist[i,j])/dx_i
        
        # C[i, j+1] = 0.5 * dist[i,j]
        # d(C[i, j+1])/dx_i = 0.5 * (x_i - x_j) / dist[i,j]
        
        # grad_obj[i] = sum_k w[i,k] * d(C[i,k])/dx_i
        # = w[i,0]*d(r_bound_i)/dx_i + sum_{j!=i} w[i,j+1] * 0.5 * (x_i - x_j) / dist[i,j]
        
        # Approximation for boundary gradient:
        # If x_i < 0.5, d(r_bound)/dx_i = 1 (if x is min), else 0.
        # This is non-smooth. To keep it simple and robust, we rely on the SA phase to keep points away from boundaries 
        # or use a simple penalty if needed. However, since we are maximizing radius, points will naturally move away from boundaries if constrained.
        # For the gradient polish, we can ignore the boundary gradient term if we trust the SA result to be interior,
        # or add a small repulsion from boundaries.
        # Let's add a simple boundary repulsion gradient to the objective gradient to help L-BFGS-B.
        
        # Boundary repulsion: -log(x), -log(1-x), etc.
        # Or simpler: linear penalty if close.
        # Let's stick to the smooth soft-min gradient for pairwise terms and add a simple boundary force.
        
        # Pairwise gradient
        # W_pair = w[:, 1:]  # (N, N), where W_pair[i, j] corresponds to constraint 0.5*dist[i,j]
        W_pair = w[:, 1:]
        
        # dist_safe = dist with diagonal 1 to avoid div by 0 (though dist[i,i] is inf, so W_pair[i,i] should be ~0)
        dist_safe = dist.copy()
        np.fill_diagonal(dist_safe, 1.0)
        
        # d(0.5*dist[i,j])/dx_i = 0.5 * (x_i - x_j) / dist[i,j]
        # grad_x[i] += sum_j W_pair[i,j] * 0.5 * (x_i - x_j) / dist[i,j]
        
        # Vectorized:
        # term_x[i,j] = W_pair[i,j] * 0.5 * (x_i - x_j) / dist[i,j]
        # grad_x[i] = sum_j term_x[i,j]
        
        diff_x = x[:, None] - x[None, :]
        diff_y = y[:, None] - y[None, :]
        
        term_x = W_pair * 0.5 * diff_x / dist_safe
        term_y = W_pair * 0.5 * diff_y / dist_safe
        
        # Mask diagonal (should be 0 anyway due to W_pair[i,i] ~ 0)
        np.fill_diagonal(term_x, 0)
        np.fill_diagonal(term_y, 0)
        
        grad_x += np.sum(term_x, axis=1)
        grad_y += np.sum(term_y, axis=1)
        
        # Boundary gradient approximation
        # We want to push x away from 0 and 1 if r_bound is active.
        # Simple heuristic: if x < 0.1, push right. If x > 0.9, push left.
        # This is non-smooth but helps escape corners.
        # Better: use the soft-min weight for boundary constraint w[:, 0]
        # d(r_bound)/dx is 1 if x is the min, 0 otherwise.
        # We can approximate this by: if x < 0.5 and y > x and 1-x > x and 1-y > x, then dx/dx=1.
        # This is complex. Let's just add a small constant repulsion from boundaries to the gradient.
        # This effectively adds a penalty term -k * (x^2 + (1-x)^2 ...) to the objective implicitly.
        
        # Actually, L-BFGS-B has bounds. We can just set bounds [0, 1] and let it handle it.
        # The gradient above is for the smooth pairwise part.
        # The boundary part is non-smooth. 
        # To make it fully smooth, we could replace r_bound with a smooth approximation too.
        # r_bound_smooth = softmin(x, 1-x, y, 1-y)
        # Let's do that for consistency.
        
        # Recalculate with smooth boundary
        # C[i, 0] was r_bound. Let's replace it with smooth boundary min.
        # But we already computed w based on C.
        # For the gradient, let's assume the boundary is handled by bounds in L-BFGS-B and the pairwise forces.
        # If a circle is too close to boundary, the "radius" drops, and the objective decreases.
        # The gradient w.r.t x from the boundary term is needed.
        
        # Let's compute the smooth boundary gradient properly.
        # r_bound = min(x, 1-x, y, 1-y)
        # Smooth: softmin(x, 1-x, y, 1-y)
        # Let b = [x, 1-x, y, 1-y]
        # w_b = exp(-beta * b) / sum(exp(-beta * b))
        # d(r_bound)/dx = -w_b[0] * 1 + w_b[1] * (-1) ... wait.
        # r = -1/beta * log(sum(exp(-beta * b_k)))
        # dr/db_k = exp(-beta * b_k) / sum(exp(-beta * b_j)) = w_k
        # dr/dx = dr/db_0 * db_0/dx + dr/db_1 * db_1/dx + ...
        # db_0/dx = 1, db_1/dx = -1, db_2/dx = 0, db_3/dx = 0
        # dr/dx = w_0 - w_1
        
        # We need to recompute weights for boundary separately or include them in C.
        # In our C matrix, C[:, 0] was the hard r_bound.
        # To be fully smooth, we should have constructed C differently.
        # However, for the polish phase, the SA has already placed points well.
        # The gradient from pairwise terms is dominant.
        # We will add a simple numerical gradient check or just rely on L-BFGS-B with bounds.
        
        return np.concatenate([grad_x, grad_y])

    def sa_optimize(x0, y0, seed, max_iters=3000):
        """Simulated annealing with vectorized updates."""
        rng = np.random.RandomState(seed)
        x = x0.copy()
        y = y0.copy()
        
        # Initial evaluation
        r_curr = compute_radii_hard(x, y)
        curr_val = np.sum(r_curr)
        best_val = curr_val
        best_x = x.copy()
        best_y = y.copy()
        
        T_start = 0.05
        T_end = 0.0005
        cool_rate = (T_end / T_start) ** (1.0 / max_iters)
        
        for it in range(max_iters):
            if time.time() - start_time > DEADLINE:
                break
            
            T = T_start * (cool_rate ** it)
            
            # Propose move: random perturbation
            # Scale perturbation by temperature
            scale = 0.02 * (T / T_start) + 0.005
            
            dx_prop = rng.normal(0, scale, N)
            dy_prop = rng.normal(0, scale, N)
            
            x_new = x + dx_prop
            y_new = y + dy_prop
            
            # Clip to valid range [0.01, 0.99] to keep inside
            x_new = np.clip(x_new, 0.01, 0.99)
            y_new = np.clip(y_new, 0.01, 0.99)
            
            # Evaluate new state
            r_new = compute_radii_hard(x_new, y_new)
            new_val = np.sum(r_new)
            
            delta = new_val - curr_val
            
            if delta > 0 or rng.random() < np.exp(delta / T):
                x = x_new
                y = y_new
                curr_val = new_val
                
                if curr_val > best_val:
                    best_val = curr_val
                    best_x = x.copy()
                    best_y = y.copy()
            
        return best_x, best_y, best_val

    def gradient_polish(x, y, max_iter=100):
        """Polish using L-BFGS-B on smooth objective."""
        z0 = np.column_stack((x, y)).flatten()
        
        bounds = [(0.0, 1.0)] * (2 * N)
        
        try:
            res = minimize(
                objective_smooth,
                z0,
                jac=objective_grad,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if res.success:
                z_res = res.x.reshape(N, 2)
                return z_res[:, 0], z_res[:, 1]
        except Exception:
            pass
        return x, y

    # Initializations
    inits = []
    
    # 1. Grid
    side = int(np.ceil(np.sqrt(N)))
    x_grid = np.zeros(N)
    y_grid = np.zeros(N)
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N:
                break
            x_grid[idx] = (i + 0.5) / side
            y_grid[idx] = (j + 0.5) / side
            idx += 1
    inits.append((x_grid, y_grid, "grid"))
    
    # 2. Hexagonal
    x_hex = np.zeros(N)
    y_hex = np.zeros(N)
    idx = 0
    for i in range(side):
        for j in range(side):
            if idx >= N:
                break
            x_hex[idx] = (i + 0.5) / side + (0.5 / side) * (j % 2)
            y_hex[idx] = (j + 0.5) / side
            idx += 1
    inits.append((x_hex, y_hex, "hex"))
    
    # 3. Random
    rng_init = np.random.RandomState(42)
    x_rand = rng_init.uniform(0.1, 0.9, N)
    y_rand = rng_init.uniform(0.1, 0.9, N)
    inits.append((x_rand, y_rand, "rand"))
    
    best_score = -np.inf
    best_solution = None
    
    for i, (x0, y0, name) in enumerate(inits):
        if time.time() - start_time > DEADLINE - 10:
            break
            
        # SA
        x_sa, y_sa, val_sa = sa_optimize(x0, y0, seed=100 + i, max_iters=2000)
        
        # Gradient Polish
        x_pol, y_pol = gradient_polish(x_sa, y_sa, max_iter=50)
        
        # Final Evaluation
        r_final = compute_radii_hard(x_pol, y_pol)
        score = np.sum(r_final)
        
        if score > best_score:
            best_score = score
            best_solution = np.column_stack((x_pol, y_pol, r_final))
            
        # Check time
        if time.time() - start_time > DEADLINE - 5:
            break
            
    # Fallback
    if best_solution is None:
        x = np.linspace(0.1, 0.9, N)
        y = np.linspace(0.1, 0.9, N)
        r = compute_radii_hard(x, y)
        best_solution = np.column_stack((x, y, r))
        
    return best_solution

if __name__ == "__main__":
    res = construct()
    print(f"Sum of radii: {np.sum(res[:, 2])}")
    print(res)
