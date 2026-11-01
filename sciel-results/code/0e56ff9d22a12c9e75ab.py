import numpy as np
import time
from scipy.optimize import minimize

N = 200

def construct():
    start_time = time.time()
    time_limit = 110.0  # 10s safety margin
    
    # 1. Initialization: Hexagonal grid
    # We want to place 200 circles.
    # Hex grid spacing: if radius is r, spacing is 2r * sqrt(3)/2 = r*sqrt(3) vertical, 2r horizontal?
    # Actually, let's just generate a grid of points and pick the first N.
    # Better: generate a hex lattice that fills the square.
    
    # Let's estimate a starting radius.
    # Area per circle ~ 1/200. r ~ 0.035.
    # Hex grid with spacing s = 2*r*sqrt(3)/2? No, distance between centers is 2r.
    # In hex packing, distance is 2r.
    # Let's generate points on a hex grid with spacing 0.08 (approx 2*0.04).
    
    # Generate hex grid
    points = []
    r_init = 0.04
    dx = 2 * r_init
    dy = r_init * np.sqrt(3)
    
    x = -0.5
    row = 0
    while x <= 0.5:
        y = -0.5
        offset = 0 if row % 2 == 0 else dx / 2
        while y <= 0.5:
            points.append((x + offset, y))
            y += dy
        x += dx
        row += 1
        
    points = np.array(points)
    # Filter points inside square
    mask = (points[:, 0] >= -0.5) & (points[:, 0] <= 0.5) & (points[:, 1] >= -0.5) & (points[:, 1] <= 0.5)
    points = points[mask]
    
    # If we have more than N, we need to select N.
    # If we have less, we need to add more.
    # For now, let's just take the first N if available.
    # A better init: sort by distance to center and take closest N?
    # Or just use a square grid if hex is messy.
    # Let's stick to a simple square grid init but with a better optimization.
    # Actually, the hex grid above might have irregular edges.
    # Let's use a square grid init for simplicity and robustness, but maybe shifted?
    # No, hex is better.
    # Let's just ensure we have exactly N points.
    
    if len(points) < N:
        # Fallback to square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        points = grid[:N]
    else:
        # Select N points. To maximize spread, maybe take the ones with largest min-distance to others?
        # That's expensive.
        # Just take the first N? They are generated in a pattern.
        # Let's try to center the grid.
        # The generated grid covers the square.
        # Let's just take points[:N].
        points = points[:N]
        
    # 2. Optimization
    # Flatten points to 1D array
    x0 = points.flatten()
    
    # Bounds
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    # Smooth objective and gradient
    lambda_val = 200.0  # Sharpness of soft-min
    
    def objective_and_gradient(x):
        x = np.array(x)
        c = x.reshape(N, 2)
        
        # Clip to bounds to avoid huge distances? No, bounds are handled by optimizer.
        # But for distance calc, we assume they are in bounds.
        
        # Pairwise distances
        # To save memory and time, we can compute in batches or just use broadcasting for N=200.
        # N=200, N^2=40000. Broadcasting is fine.
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        # Mask diagonal
        mask = ~np.eye(N, dtype=bool)
        
        # Wall distances
        # d_x_pos = 0.5 - c[:, 0]
        # d_x_neg = 0.5 + c[:, 0]
        # d_y_pos = 0.5 - c[:, 1]
        # d_y_neg = 0.5 + c[:, 1]
        
        # We want to include these in the sum.
        # Let's create a list of all distances.
        # Pair distances: dist[mask]
        # Wall distances:
        dw = np.concatenate([
            0.5 - c[:, 0],
            0.5 + c[:, 0],
            0.5 - c[:, 1],
            0.5 + c[:, 1]
        ])
        
        all_dists = np.concatenate([dist[mask], dw])
        
        # Soft min
        # r_soft = -1/lambda * log(sum(exp(-lambda * d)))
        # To avoid overflow, subtract max
        max_d = np.max(all_dists)
        exp_terms = np.exp(-lambda_val * (all_dists - max_d))
        S = np.sum(exp_terms)
        log_S = np.log(S)
        
        r_soft = -1.0 / lambda_val * (log_S - lambda_val * max_d)
        # Wait, log(sum(exp(-lambda(d-max)))) = log(sum(exp(-lambda(d)))) + lambda*max
        # So -1/lambda * (log_S - lambda*max_d) is correct?
        # Let's check:
        # sum(exp(-lambda d)) = exp(-lambda max) * sum(exp(-lambda(d-max)))
        # log(sum(exp(-lambda d))) = -lambda max + log(sum(exp(-lambda(d-max))))
        # r_soft = -1/lambda * (-lambda max + log_S) = max - 1/lambda log_S.
        
        val = -r_soft
        
        # Gradient
        # d(val)/dx = - d(r_soft)/dx
        # d(r_soft)/dx = -1/lambda * (1/S * sum(exp(-lambda d) * (-lambda * dd/dx)))
        #             = 1/S * sum(exp(-lambda d) * dd/dx)
        # So d(val)/dx = -1/S * sum(exp(-lambda d) * dd/dx)
        
        # We need to compute the sum of exp(-lambda d) * grad_d for each coordinate.
        # This is tricky with the concatenated array.
        
        # Let's compute gradients for pairs and walls separately.
        
        grad = np.zeros_like(x)
        
        # Pairs
        # weights = exp(-lambda * (dist - max_d)) / S
        # Note: max_d is global max.
        w_pairs = np.exp(-lambda_val * (dist - max_d)) / S
        w_pairs = np.where(mask, w_pairs, 0)
        
        # grad_dist for pairs:
        # d(dist_ij)/dx_i = (x_i - x_j) / dist_ij
        # We can compute this using broadcasting.
        # diff is (N, N, 2).
        # inv_dist = np.where(dist > 0, 1/dist, 0)
        # grad_dist = diff * inv_dist[:,:,np.newaxis]
        
        inv_dist = np.zeros_like(dist)
        nonzero = dist > 1e-8
        inv_dist[nonzero] = 1.0 / dist[nonzero]
        
        grad_dist = diff * inv_dist[:, :, np.newaxis]
        
        # Sum over j for each i: sum_j w_pairs[i,j] * grad_dist[i,j]
        # w_pairs is (N, N). grad_dist is (N, N, 2).
        # We want result (N, 2).
        # grad_pairs = np.sum(w_pairs[:, :, np.newaxis] * grad_dist, axis=1)
        
        grad_pairs = np.sum(w_pairs[:, :, np.newaxis] * grad_dist, axis=1)
        
        # Walls
        # dw are 4*N values.
        # exp_w = exp(-lambda * (dw - max_d)) / S
        exp_w = np.exp(-lambda_val * (dw - max_d)) / S
        
        # Gradients for walls
        # dw[0:N] = 0.5 - x_i -> grad_x = -1
        # dw[N:2N] = 0.5 + x_i -> grad_x = 1
        # dw[2N:3N] = 0.5 - y_i -> grad_y = -1
        # dw[3N:4N] = 0.5 + y_i -> grad_y = 1
        
        grad_walls = np.zeros((N, 2))
        grad_walls[:, 0] += -1.0 * exp_w[:N]
        grad_walls[:, 0] += 1.0 * exp_w[N:2*N]
        grad_walls[:, 1] += -1.0 * exp_w[2*N:3*N]
        grad_walls[:, 1] += 1.0 * exp_w[3*N:4*N]
        
        total_grad = grad_pairs + grad_walls
        
        # val = -r_soft
        # d(val)/dx = - d(r_soft)/dx = - (1/S * sum(exp * dd/dx)) = - total_grad
        # Wait, earlier I derived:
        # d(r_soft)/dx = 1/S * sum(exp * dd/dx)
        # So d(val)/dx = - total_grad
        
        return val, -total_grad.flatten()
    
    # Run optimization
    best_x = x0
    best_val = objective_and_gradient(x0)[0]
    
    try:
        res = minimize(
            objective_and_gradient,
            x0,
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if time.time() - start_time < time_limit:
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
                
    except Exception as e:
        pass
    
    # Polish with a few iterations of direct min-distance maximization?
    # Or just return.
    
    c = best_x.reshape(N, 2)
    # Clip to ensure validity
    c[:, 0] = np.clip(c[:, 0], -0.5, 0.5)
    c[:, 1] = np.clip(c[:, 1], -0.5, 0.5)
    
    return c
