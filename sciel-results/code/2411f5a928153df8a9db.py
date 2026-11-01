"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 200

# EVOLVE-BLOCK-START
def construct():
    # Hexagonal lattice initialization
    # Estimate spacing
    # Area per point approx 1/N
    # Hex cell area approx sqrt(3)/2 * s^2
    # s = sqrt(2 * (1/N) / sqrt(3))
    s_base = np.sqrt(2.0 / (N * np.sqrt(3)))
    
    points = None
    # Try to find a good grid density
    for scale in np.linspace(0.85, 1.15, 10):
        s = s_base * scale
        row_height = s * np.sqrt(3) / 2
        num_rows = int(1.0 / row_height) + 2
        num_cols = int(1.0 / s) + 2
        
        pts = []
        for i in range(-num_rows//2, num_rows//2 + 1):
            y = i * row_height
            offset = (s / 2) if (i % 2 != 0) else 0
            for j in range(-num_cols//2, num_cols//2 + 1):
                x = j * s + offset
                if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                    pts.append([x, y])
                    
        if len(pts) >= N:
            points = np.array(pts)
            # Shuffle and take N
            idx = np.random.choice(len(points), N, replace=False)
            points = points[idx]
            break
            
    if points is None:
        # Fallback to grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        points = np.array([(x, y) for y in ticks for x in ticks])
        points = points[:N]
        
    # Clip to be strictly inside
    points = np.clip(points, -0.49, 0.49)
    
    def objective_and_grad(flat_x):
        x = flat_x.reshape(N, 2)
        
        # Pairwise distances
        dx = x[:, 0, None] - x[None, :, 0]
        dy = x[:, 1, None] - x[None, :, 1]
        d_sq = dx**2 + dy**2
        d = np.sqrt(d_sq + 1e-12)
        
        # Wall distances
        dw1 = 0.5 - x[:, 0]
        dw2 = 0.5 + x[:, 0]
        dw3 = 0.5 - x[:, 1]
        dw4 = 0.5 + x[:, 1]
        
        # Indices for pairs
        i_idx, j_idx = np.triu_indices(N, k=1)
        d_pairs = d[i_idx, j_idx]
        
        # Walls
        d_walls = np.concatenate([dw1, dw2, dw3, dw4])
        
        all_d = np.concatenate([d_pairs, d_walls])
        
        beta = 100.0
        
        # Soft min
        d_min = np.min(all_d)
        exp_term = np.exp(-beta * (all_d - d_min))
        sum_exp = np.sum(exp_term)
        weights = exp_term / sum_exp
        
        val = -1.0 / beta * (np.log(sum_exp) - beta * d_min)
        
        # Gradient
        grad_x = np.zeros_like(x)
        
        # Pair contributions
        W = np.zeros((N, N))
        W[i_idx, j_idx] = weights[:len(d_pairs)]
        W[j_idx, i_idx] = weights[:len(d_pairs)]
        
        safe_d = np.where(d > 1e-8, d, 1.0)
        grad_dx = dx / safe_d
        grad_dy = dy / safe_d
        
        np.fill_diagonal(grad_dx, 0)
        np.fill_diagonal(grad_dy, 0)
        
        grad_x[:, 0] = np.sum(W * grad_dx, axis=1)
        grad_x[:, 1] = np.sum(W * grad_dy, axis=1)
        
        # Wall contributions
        w_wall = weights[len(d_pairs):]
        w_dw1 = w_wall[0:N]
        w_dw2 = w_wall[N:2*N]
        w_dw3 = w_wall[2*N:3*N]
        w_dw4 = w_wall[3*N:4*N]
        
        grad_x[:, 0] += -w_dw1 + w_dw2
        grad_x[:, 1] += -w_dw3 + w_dw4
        
        return val, grad_x.flatten()
        
    res = minimize(objective_and_grad, points.flatten(), jac=True, method='L-BFGS-B', 
                   options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
    
    best_x = res.x.reshape(N, 2)
    best_x = np.clip(best_x, -0.5, 0.5)
    
    return best_x
# EVOLVE-BLOCK-END
