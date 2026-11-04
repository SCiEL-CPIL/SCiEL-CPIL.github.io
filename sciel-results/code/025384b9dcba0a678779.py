import numpy as np
import scipy.optimize as opt
import time

N = 100
DEADLINE = 140  # seconds

def construct():
    start_time = time.time()
    
    # Objective function to minimize
    # High power p makes it focus on the minimum distance
    p = 6.0
    lambda_wall = 1000.0
    
    def energy_and_grad(flat_x):
        x = flat_x.reshape(N, 3)
        
        # Pairwise distances
        # diff[i, j] = x[i] - x[j]
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        
        # Mask for i < j
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        d = dist[mask]
        
        # Avoid division by zero
        d_safe = np.maximum(d, 1e-8)
        
        # Term 1: Inverse distance power
        # E1 = sum (1/d^p)
        e1 = np.sum(1.0 / (d_safe ** p))
        
        # Gradient of E1
        # dE1/dx_i = sum_{j != i} -p * d_{ij}^{-p-2} * (x_i - x_j)
        # We can compute this using the full matrix
        # G[i, j] = -p * d_{ij}^{-p-2}
        # grad[i] = sum_j G[i, j] * (x[i] - x[j])
        
        # dist is symmetric. d_safe is symmetric.
        # We only need i < j for the sum, but the gradient formula works with full sum if we handle symmetry correctly?
        # Actually, sum_{j != i} includes both j and i.
        # Let's compute the full matrix of coefficients.
        coeff = np.zeros((N, N))
        coeff[mask] = -p / (d_safe ** (p + 2))
        # The term for j=i is 0.
        # The term for (i, j) and (j, i) are distinct in the sum over j for a fixed i.
        # Wait, the derivative of sum_{i<j} f(d_{ij}) w.r.t x_i involves only terms where i is the smaller index? No.
        # Let S = sum_{i<j} f(d_{ij}).
        # dS/dx_i = sum_{j < i} f'(d_{ji}) * (x_i - x_j)/d_{ji} + sum_{j > i} f'(d_{ij}) * (x_i - x_j)/d_{ij}.
        # Since f(d) = d^{-p}, f'(d) = -p d^{-p-1}.
        # Term is -p d^{-p-2} (x_i - x_j).
        
        # So we need the coefficient for each pair (i, j) where i != j.
        # For a fixed i, the contribution from j is -p d_{ij}^{-p-2} (x_i - x_j).
        # This is symmetric in the sense that the coefficient matrix C where C[i,j] = -p d_{ij}^{-p-2} (for i!=j) is symmetric.
        # Then grad[i] = sum_{j!=i} C[i,j] * (x[i] - x[j]).
        
        # Let's build C.
        # d_safe is 1D (mask applied). We need it in 2D for C.
        # Or just compute C from dist.
        dist_safe = np.maximum(dist, 1e-8)
        C = -p / (dist_safe ** (p + 2))
        np.fill_diagonal(C, 0.0)
        
        # grad_pair = C * diff  (element-wise)
        # grad[i] = sum_j grad_pair[i, j]
        grad_pair = C * diff
        grad_x = np.sum(grad_pair, axis=1)
        
        # Term 2: Wall penalty
        # P = sum_{i,k} max(0, |x_ik| - 0.5)^p
        # Let v_ik = x_ik
        # If |v| > 0.5, term is (|v| - 0.5)^p
        # Deriv w.r.t v: p * (|v| - 0.5)^{p-1} * sign(v)
        
        abs_x = np.abs(x)
        excess = np.maximum(0.0, abs_x - 0.5)
        e2 = np.sum(excess ** p)
        
        # Gradient
        # grad_wall[i, k] = lambda * p * excess^{p-1} * sign(x_ik)
        # If excess == 0, grad is 0.
        # Note: p=6 is even, so (|v|-0.5)^p is smooth?
        # f(v) = (max(0, |v|-0.5))^p.
        # If p is integer >= 1, it's differentiable everywhere?
        # At v=0.5, left deriv 0, right deriv 0. Yes.
        
        grad_wall = lambda_wall * p * (excess ** (p - 1)) * np.sign(x)
        # Where excess is 0, grad is 0.
        # But if p < 1, it's singular. p=6 is fine.
        
        total_e = e1 + e2
        total_grad = grad_x + grad_wall
        
        return total_e, total_grad

    def energy_only(flat_x):
        return energy_and_grad(flat_x)[0]

    best_radius = 0.0
    best_x = None
    
    # Initializations
    # 1. Grid
    # 2. Random
    
    def get_radius(x):
        x = x.reshape(N, 3)
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        d = dist[mask]
        r_center = np.min(d) / 2.0
        
        wall_dist = 0.5 - np.max(np.abs(x), axis=1)
        r_wall = np.min(wall_dist)
        
        return np.min([r_center, r_wall])

    # Try a few random starts
    # And the grid start
    
    # Grid start
    m = 5
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    x0_grid = grid[:N].flatten()
    
    candidates = [x0_grid]
    
    # Random starts
    for i in range(5):
        # Uniform random in [-0.4, 0.4] to start inside
        x0_rand = np.random.uniform(-0.4, 0.4, N * 3)
        candidates.append(x0_rand)
    
    for x0 in candidates:
        if time.time() - start_time > DEADLINE - 10:
            break
            
        try:
            res = opt.minimize(
                energy_only, 
                x0, 
                jac=energy_and_grad, 
                method='L-BFGS-B', 
                options={'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-6}
            )
            x_opt = res.x
            r = get_radius(x_opt)
            if r > best_radius:
                best_radius = r
                best_x = x_opt
        except Exception as e:
            continue
            
    if best_x is None:
        # Fallback to grid
        best_x = x0_grid
        best_radius = get_radius(best_x)
        
    return best_x.reshape(N, 3)
