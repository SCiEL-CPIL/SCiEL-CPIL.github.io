import numpy as np
import time
from scipy.optimize import minimize

N = 26

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin for 150s limit
    
    best_score = -np.inf
    best_solution = None
    
    # Initial seed solution (grid) to ensure we have a valid fallback
    side = int(np.ceil(np.sqrt(N)))
    cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
    best_solution = np.array(cells[:N])
    best_score = np.sum(best_solution[:, 2])
    
    def loss_and_grad(z):
        # z is shape (N, 3) -> flatten to (3N,)
        z = z.reshape(N, 3)
        x = z[:, 0]
        y = z[:, 1]
        r = z[:, 2]
        
        # Objective: maximize sum of r -> minimize -sum(r)
        obj = -np.sum(r)
        grad_obj = np.zeros_like(z)
        grad_obj[:, 2] = -1.0
        
        # Overlap Penalty
        # d_ij = sqrt((x_i - x_j)^2 + (y_i - y_j)^2)
        # Constraint: d_ij >= r_i + r_j
        # Violation: max(0, r_i + r_j - d_ij)
        
        pen = 0.0
        grad_pen = np.zeros_like(z)
        
        # Compute pairwise distances
        # Using broadcasting for efficiency
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        
        # Avoid division by zero
        d_safe = np.where(d > 1e-8, d, 1e-8)
        
        # We only care about i < j
        # Sum over i < j
        # r_i + r_j - d_ij
        r_sum = r[:, None] + r[None, :]
        viol = r_sum - d_safe
        
        # Smooth max(0, viol)
        beta = 1.0
        soft_viol = beta * np.log1p(np.exp(viol / beta))
        
        # Only take upper triangle
        iu = np.triu_indices(N, k=1)
        sv = soft_viol[iu]
        
        pen += np.sum(sv**2) # Squaring to make it smoother and stronger
        
        # Gradient of penalty
        # d/dx_i ( (r_i + r_j - d_ij)^2 ) = 2 * (r_i + r_j - d_ij) * (-d/dx_i d_ij)
        # d/dx_i d_ij = (x_i - x_j) / d_ij
        # So grad term for x_i is 2 * viol_ij * (-(x_i - x_j)/d_ij)
        
        # We need to accumulate gradients for each i and j
        # Let's compute the gradient of the soft violation first
        # d(soft_viol)/d(viol) = sigmoid(viol/beta)
        sig = 1.0 / (1.0 + np.exp(-viol / beta))
        
        # grad of soft_viol**2 w.r.t viol is 2 * soft_viol * sig
        # But wait, soft_viol is an approximation of max(0, viol).
        # If we use soft_viol directly in the loss, the gradient is sig.
        # If we use soft_viol**2, the gradient is 2 * soft_viol * sig.
        
        # Let's stick to the simpler: Loss += sum(soft_viol)
        # This is smoother and easier to differentiate.
        pen = np.sum(sv)
        
        # Gradient of sum(soft_viol) w.r.t z
        # d pen / d r_i = sum_{j != i} sig_ij
        # d pen / d x_i = sum_{j != i} sig_ij * (-(x_i - x_j)/d_ij)
        
        # Compute sig for all pairs
        sig_all = 1.0 / (1.0 + np.exp(-viol / beta))
        
        # Zero out diagonal
        np.fill_diagonal(sig_all, 0.0)
        
        # Gradient w.r.t r
        grad_r_pen = np.sum(sig_all, axis=1)
        
        # Gradient w.r.t x, y
        # term_ij = sig_ij * (x_i - x_j) / d_ij
        # grad_x_i = - sum_j term_ij
        # grad_x_j = sum_j term_ij
        
        factor = sig_all / d_safe
        np.fill_diagonal(factor, 0.0)
        
        grad_x_pen = -np.sum(factor * dx, axis=1)
        grad_y_pen = -np.sum(factor * dy, axis=1)
        
        # Boundary Penalty
        # Constraints: x >= r, y >= r, 1-x >= r, 1-y >= r
        # viol_x1 = max(0, r - x)
        # viol_x2 = max(0, r - (1-x))
        # etc.
        
        beta_b = 1.0
        v_x1 = r - x
        v_x2 = r - (1 - x)
        v_y1 = r - y
        v_y2 = r - (1 - y)
        
        sv_x1 = beta_b * np.log1p(np.exp(v_x1 / beta_b))
        sv_x2 = beta_b * np.log1p(np.exp(v_x2 / beta_b))
        sv_y1 = beta_b * np.log1p(np.exp(v_y1 / beta_b))
        sv_y2 = beta_b * np.log1p(np.exp(v_y2 / beta_b))
        
        pen_b = np.sum(sv_x1) + np.sum(sv_x2) + np.sum(sv_y1) + np.sum(sv_y2)
        
        sig_x1 = 1.0 / (1.0 + np.exp(-v_x1 / beta_b))
        sig_x2 = 1.0 / (1.0 + np.exp(-v_x2 / beta_b))
        sig_y1 = 1.0 / (1.0 + np.exp(-v_y1 / beta_b))
        sig_y2 = 1.0 / (1.0 + np.exp(-v_y2 / beta_b))
        
        # Grads
        # d/dx sv_x1 = -sig_x1
        # d/dx sv_x2 = sig_x2
        # d/dr sv_x1 = sig_x1
        # d/dr sv_x2 = sig_x2
        
        grad_x_pen_b = -sig_x1 + sig_x2
        grad_y_pen_b = -sig_y1 + sig_y2
        grad_r_pen_b = sig_x1 + sig_x2 + sig_y1 + sig_y2
        
        # Total Loss and Gradient
        # We need to scale the penalty heavily
        lambda_pen = 50.0
        
        total_loss = obj + lambda_pen * (pen + pen_b)
        
        total_grad = grad_obj + lambda_pen * (
            np.column_stack((grad_x_pen + grad_x_pen_b, 
                             grad_y_pen + grad_y_pen_b, 
                             grad_r_pen + grad_r_pen_b))
        )
        
        return total_loss, total_grad.flatten()
    
    def run_optimization(z_init):
        bounds = [(0, 1), (0, 1), (0, 0.5)] * N
        res = minimize(
            loss_and_grad, 
            z_init, 
            method='L-BFGS-B', 
            jac=True, 
            bounds=bounds, 
            options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        return res.x, res.fun
    
    # Multi-start search
    rng = np.random.default_rng(42)
    
    while time.time() - start_time < time_limit:
        # Generate random initial solution
        # Random positions
        x = rng.uniform(0.1, 0.9, N)
        y = rng.uniform(0.1, 0.9, N)
        # Small initial radii
        r = rng.uniform(0.01, 0.05, N)
        
        z_init = np.column_stack((x, y, r)).flatten()
        
        try:
            z_opt, loss = run_optimization(z_init)
            z_opt = z_opt.reshape(N, 3)
            
            # Check validity and calculate score
            # Ensure no overlaps and within bounds
            valid = True
            for i in range(N):
                if z_opt[i, 0] < -1e-6 or z_opt[i, 0] > 1 + 1e-6: valid = False
                if z_opt[i, 1] < -1e-6 or z_opt[i, 1] > 1 + 1e-6: valid = False
                if z_opt[i, 2] < 0: valid = False
                
                # Boundary check
                if z_opt[i, 0] < z_opt[i, 2] - 1e-6: valid = False
                if z_opt[i, 1] < z_opt[i, 2] - 1e-6: valid = False
                if 1 - z_opt[i, 0] < z_opt[i, 2] - 1e-6: valid = False
                if 1 - z_opt[i, 1] < z_opt[i, 2] - 1e-6: valid = False
                
            for i in range(N):
                for j in range(i+1, N):
                    d = np.sqrt((z_opt[i,0]-z_opt[j,0])**2 + (z_opt[i,1]-z_opt[j,1])**2)
                    if d < z_opt[i,2] + z_opt[j,2] - 1e-6:
                        valid = False
            
            if valid:
                score = np.sum(z_opt[:, 2])
                if score > best_score:
                    best_score = score
                    best_solution = z_opt
        except Exception as e:
            pass # Ignore errors in optimization step
            
    return best_solution
