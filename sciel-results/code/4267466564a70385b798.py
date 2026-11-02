import numpy as np
import time

N = 21
DEADLINE = 140.0

def construct():
    start_time = time.time()
    best_score = -np.inf
    best_circles = None
    
    # Parameters
    lambda_overlap = 50.0
    mu_perim = 50.0
    alpha0 = 0.02
    beta = 0.005
    steps_per_restart = 200
    
    while time.time() - start_time < DEADLINE:
        # Initialize
        # Try a few different initialization strategies
        init_type = np.random.randint(3)
        if init_type == 0:
            # Random
            x = np.random.uniform(-0.4, 0.4, N)
            y = np.random.uniform(-0.4, 0.4, N)
            r = np.full(N, 0.05)
        elif init_type == 1:
            # Grid
            side = 5
            x, y = np.meshgrid(np.linspace(-0.4, 0.4, side), np.linspace(-0.4, 0.4, side))
            x = x.flatten()[:N]
            y = y.flatten()[:N]
            r = np.full(N, 0.08)
        else:
            # Hex-like
            x = np.random.uniform(-0.4, 0.4, N)
            y = np.random.uniform(-0.4, 0.4, N)
            r = np.full(N, 0.06)
            
        for step in range(steps_per_restart):
            if time.time() - start_time > DEADLINE:
                break
                
            # Compute overlaps
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(d, np.inf)
            
            overlap = np.maximum(0, r[:, None] + r[None, :] - d)
            factor = 2 * overlap / d
            np.fill_diagonal(factor, 0)
            
            grad_x = -np.sum(factor * dx, axis=1)
            grad_y = -np.sum(factor * dy, axis=1)
            grad_r = np.sum(factor, axis=1)
            
            # Perimeter penalty
            x_min = np.min(x - r)
            x_max = np.max(x + r)
            y_min = np.min(y - r)
            y_max = np.max(y + r)
            W_bb = x_max - x_min
            H_bb = y_max - y_min
            perim_val = W_bb + H_bb
            
            if perim_val > 2.0:
                ind_x_max = (x + r) == x_max
                ind_x_min = (x - r) == x_min
                ind_y_max = (y + r) == y_max
                ind_y_min = (y - r) == y_min
                
                dW_dx = ind_x_max.astype(float) - ind_x_min.astype(float)
                dW_dr = ind_x_max.astype(float) + ind_x_min.astype(float)
                
                dH_dy = ind_y_max.astype(float) - ind_y_min.astype(float)
                dH_dr = ind_y_max.astype(float) + ind_y_min.astype(float)
                
                dP_dx = 2 * mu_perim * (perim_val - 2) * dW_dx
                dP_dy = 2 * mu_perim * (perim_val - 2) * dH_dy
                dP_dr = 2 * mu_perim * (perim_val - 2) * (dW_dr + dH_dr)
            else:
                dP_dx = np.zeros(N)
                dP_dy = np.zeros(N)
                dP_dr = np.zeros(N)
                
            grad_x_total = -lambda_overlap * grad_x - dP_dx
            grad_y_total = -lambda_overlap * grad_y - dP_dy
            grad_r_total = 1.0 - lambda_overlap * grad_r - dP_dr
            
            alpha = alpha0 / (1 + step * beta)
            x += alpha * grad_x_total
            y += alpha * grad_y_total
            r += alpha * grad_r_total
            r = np.maximum(r, 1e-6)
            
        # Check validity and score
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(d, np.inf)
        overlap = np.maximum(0, r[:, None] + r[None, :] - d)
        max_overlap = np.max(overlap)
        
        x_min = np.min(x - r)
        x_max = np.max(x + r)
        y_min = np.min(y - r)
        y_max = np.max(y + r)
        W_bb = x_max - x_min
        H_bb = y_max - y_min
        perim = 2 * (W_bb + H_bb)
        
        if max_overlap < 1e-5 and perim <= 4.0 + 1e-5:
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_circles = np.column_stack((x, y, r))
                
    # Polish best solution if time permits
    if best_circles is not None and time.time() - start_time < DEADLINE - 10:
        x, y, r = best_circles.T
        for step in range(500):
            if time.time() - start_time > DEADLINE - 5:
                break
            # Same gradient logic but smaller step
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            d = np.sqrt(dx**2 + dy**2)
            np.fill_diagonal(d, np.inf)
            overlap = np.maximum(0, r[:, None] + r[None, :] - d)
            factor = 2 * overlap / d
            np.fill_diagonal(factor, 0)
            grad_x = -np.sum(factor * dx, axis=1)
            grad_y = -np.sum(factor * dy, axis=1)
            grad_r = np.sum(factor, axis=1)
            
            x_min = np.min(x - r)
            x_max = np.max(x + r)
            y_min = np.min(y - r)
            y_max = np.max(y + r)
            W_bb = x_max - x_min
            H_bb = y_max - y_min
            perim_val = W_bb + H_bb
            
            if perim_val > 2.0:
                ind_x_max = (x + r) == x_max
                ind_x_min = (x - r) == x_min
                ind_y_max = (y + r) == y_max
                ind_y_min = (y - r) == y_min
                dW_dx = ind_x_max.astype(float) - ind_x_min.astype(float)
                dW_dr = ind_x_max.astype(float) + ind_x_min.astype(float)
                dH_dy = ind_y_max.astype(float) - ind_y_min.astype(float)
                dH_dr = ind_y_max.astype(float) + ind_y_min.astype(float)
                dP_dx = 2 * mu_perim * (perim_val - 2) * dW_dx
                dP_dy = 2 * mu_perim * (perim_val - 2) * dH_dy
                dP_dr = 2 * mu_perim * (perim_val - 2) * (dW_dr + dH_dr)
            else:
                dP_dx = np.zeros(N)
                dP_dy = np.zeros(N)
                dP_dr = np.zeros(N)
                
            grad_x_total = -lambda_overlap * grad_x - dP_dx
            grad_y_total = -lambda_overlap * grad_y - dP_dy
            grad_r_total = 1.0 - lambda_overlap * grad_r - dP_dr
            
            alpha = 0.005
            x += alpha * grad_x_total
            y += alpha * grad_y_total
            r += alpha * grad_r_total
            r = np.maximum(r, 1e-6)
            
        # Re-check
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        d = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(d, np.inf)
        overlap = np.maximum(0, r[:, None] + r[None, :] - d)
        max_overlap = np.max(overlap)
        x_min = np.min(x - r)
        x_max = np.max(x + r)
        y_min = np.min(y - r)
        y_max = np.max(y + r)
        W_bb = x_max - x_min
        H_bb = y_max - y_min
        perim = 2 * (W_bb + H_bb)
        if max_overlap < 1e-5 and perim <= 4.0 + 1e-5:
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_circles = np.column_stack((x, y, r))

    if best_circles is None:
        side = int(np.ceil(np.sqrt(N)))
        cells = [((i + 0.5) / side, (j + 0.5) / side, 0.5 / side - 1e-3) for i in range(side) for j in range(side)]
        return np.array(cells[:N])
        
    return best_circles
