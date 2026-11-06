import numpy as np
import time
import scipy.optimize as opt

N = 200

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # 1. Generate initial hexagonal lattice configuration
    d = 0.076
    points_list = []
    dy = d * np.sqrt(3) / 2
    n_rows = int(np.ceil(1.0 / dy)) + 2
    for j in range(-n_rows, n_rows + 1):
        y = j * dy
        if y < -0.55 or y > 0.55:
            continue
        x_offset = (j % 2) * d / 2
        n_cols = int(np.ceil(1.0 / d)) + 2
        for i in range(-n_cols, n_cols + 1):
            x = i * d + x_offset
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                points_list.append([x, y])
    
    points = np.array(points_list)
    
    if len(points) > N:
        np.random.shuffle(points)
        points = points[:N]
    elif len(points) < N:
        remaining = N - len(points)
        new_pts = np.random.uniform(-0.45, 0.45, size=(remaining, 2))
        points = np.vstack([points, new_pts])
        
    points[:, 0] = np.clip(points[:, 0], -0.49, 0.49)
    points[:, 1] = np.clip(points[:, 1], -0.49, 0.49)
    
    def get_exact_radius(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        min_c = np.min(dist) / 2.0
        
        dx = np.minimum(pts[:, 0] + 0.5, 0.5 - pts[:, 0])
        dy = np.minimum(pts[:, 1] + 0.5, 0.5 - pts[:, 1])
        min_b = np.min(np.minimum(dx, dy))
        
        return min(min_c, min_b)
    
    current_r = get_exact_radius(points)
    best_points = points.copy()
    best_r = current_r

    bounds = [(-0.5, 0.5)] * (2 * N)

    def soft_min_obj_and_grad(p_flat, beta):
        p = p_flat.reshape(N, 2)
        
        # Pairwise distances
        diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        d_pairs = dist / 2.0
        
        # Wall distances
        # d_x_pos: 0.5 - x
        # d_x_neg: 0.5 + x
        # d_y_pos: 0.5 - y
        # d_y_neg: 0.5 + y
        d_walls = np.stack([
            0.5 - p[:, 0],
            0.5 + p[:, 0],
            0.5 - p[:, 1],
            0.5 + p[:, 1]
        ], axis=1)
        
        # Combine all distances into a single vector for soft-min
        # To avoid huge memory allocations if N was larger, we process separately, 
        # but for N=200, M~20800 is small.
        # However, computing exp(-beta * d) for all at once is efficient.
        
        # Clip distances to prevent underflow/overflow in exp if beta is large
        # min distance can be 0. max distance ~ 1.
        # beta * d can be large.
        
        # Calculate weights for pairs
        # We need sum of exp(-beta * d) for all terms to normalize.
        # Let Z = sum(exp(-beta * d_i))
        # w_i = exp(-beta * d_i) / Z
        
        # To be numerically stable, subtract max d? No, min d dominates.
        # Actually, for soft-min, we want to emphasize the SMALL distances.
        # exp(-beta * d) is large for small d.
        
        # Let's compute log-sum-exp for stability if needed, but for beta < 1000 and d > 0, it's fine.
        # If d is very small, exp is large.
        
        # Pair weights
        # Mask diagonal
        mask = np.ones((N, N), dtype=bool)
        np.fill_diagonal(mask, False)
        
        # For stability, we can shift by min_d, but here we just compute directly.
        # If d is 0, exp(0)=1. If d is 0.01, beta=100, exp(-1) = 0.36.
        
        e_pairs = np.exp(-beta * d_pairs)
        e_pairs[~mask] = 0.0 # Zero out diagonal
        
        e_walls = np.exp(-beta * d_walls)
        
        Z = np.sum(e_pairs) + np.sum(e_walls)
        
        if Z == 0:
            return 0.0, np.zeros_like(p_flat)
            
        w_pairs = e_pairs / Z
        w_walls = e_walls / Z
        
        # Objective value: -1/beta * log(Z)
        val = -1.0 / beta * np.log(Z)
        
        # Gradient
        grad = np.zeros_like(p)
        
        # Pair gradient contribution
        # grad[i] += sum_j w_ij * 0.5 * (p_i - p_j) / ||p_i - p_j||
        # dist is ||p_i - p_j||
        # unit vector u_ij = (p_i - p_j) / dist
        # Handle dist=0 case (should not happen with init, but just in case)
        dist_safe = np.where(dist > 1e-10, dist, 1e-10)
        u = diff / dist_safe[:, :, np.newaxis]
        
        # w_pairs is (N, N). We want to sum over j for each i.
        # contribution to grad[i] is sum_j w_pairs[i,j] * 0.5 * u[i,j]
        # Since w_pairs is symmetric and u is antisymmetric (u[j,i] = -u[i,j]),
        # this is equivalent to summing over all j.
        
        # grad_pair[i] = 0.5 * sum_j w_pairs[i,j] * u[i,j]
        # Vectorized:
        grad_pair = 0.5 * np.sum(w_pairs[:, :, np.newaxis] * u, axis=1)
        
        # Wall gradient contribution
        # For each point i, we have 4 wall distances.
        # d0 = 0.5 - x. grad_x = -1.
        # d1 = 0.5 + x. grad_x = 1.
        # d2 = 0.5 - y. grad_y = -1.
        # d3 = 0.5 + y. grad_y = 1.
        
        grad_wall_x = -w_walls[:, 0] + w_walls[:, 1]
        grad_wall_y = -w_walls[:, 2] + w_walls[:, 3]
        
        grad[:, 0] += grad_pair[:, 0] + grad_wall_x
        grad[:, 1] += grad_pair[:, 1] + grad_wall_y
        
        return val, grad.flatten()

    # Annealing schedule
    betas = [10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
    
    current_points = points.copy()
    
    for beta in betas:
        if time.time() > deadline - 10.0:
            break
            
        try:
            res = opt.minimize(
                soft_min_obj_and_grad,
                current_points.flatten(),
                args=(beta,),
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            new_points = res.x.reshape(N, 2)
            new_r = get_exact_radius(new_points)
            
            if new_r > best_r:
                best_r = new_r
                best_points = new_points.copy()
                current_points = new_points.copy()
            else:
                # If no improvement, keep current_points for next beta?
                # Or revert? Usually keeping the optimized geometry is better even if exact radius dipped slightly due to noise?
                # But get_exact_radius is deterministic. If it dipped, the soft-min approximation led to a worse exact solution.
                # We should probably use the best_points for the next iteration to not drift away.
                current_points = best_points.copy()
                
        except Exception as e:
            # If optimization fails, skip to next beta
            continue

    # Final Polish with high beta
    if time.time() < deadline - 5.0:
        try:
            res = opt.minimize(
                soft_min_obj_and_grad,
                best_points.flatten(),
                args=(2000.0,),
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            new_points = res.x.reshape(N, 2)
            new_r = get_exact_radius(new_points)
            if new_r > best_r:
                best_r = new_r
                best_points = new_points.copy()
        except Exception:
            pass

    return best_points

if __name__ == "__main__":
    pts = construct()
    r = get_exact_radius(pts)
    print(f"Shape: {pts.shape}")
    print(f"Radius: {r}")
