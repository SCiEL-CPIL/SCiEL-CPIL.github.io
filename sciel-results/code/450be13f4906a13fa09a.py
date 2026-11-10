# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def kakeya_needle_3d8() -> np.ndarray:
    N = 8
    n_tubes = N * N
    
    # Deadline guard
    start_time = time.time()
    deadline = 600.0
    safety_margin = 30.0
    
    # Initial guess: Use the Keich construction as a good starting point
    # This ensures we start from a valid, reasonable configuration
    pos = int(np.log2(N))
    j_terms = np.arange(1, pos + 1)
    keich = np.empty(N, dtype=np.float64)
    for idx in range(N):
        eps = np.array(list(str(bin(idx))[2:]), dtype=int)
        eps = np.pad(eps, (pos - len(eps), 0), "constant")
        keich[idx] = np.sum((1 - j_terms) * eps * (2.0 ** -j_terms) / pos)
        
    i_grid, j_grid = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    i_flat = i_grid.flatten()
    j_flat = j_grid.flatten()
    
    base_x_init = keich[i_flat]
    base_y_init = keich[j_flat]
    
    # Flatten to 1D vector for optimizer
    x0 = np.concatenate([base_x_init, base_y_init])
    
    # We need to estimate the volume. 
    # The volume is the integral over [0,1]^3 of the union indicator.
    # Union indicator is 1 if ANY tube contains the point.
    # Indicator of union = 1 - prod(1 - I_k) where I_k is indicator of tube k.
    # Since I_k is 0 or 1, this is exact.
    # To make it differentiable, we approximate I_k with a sigmoid.
    # I_k(a,b,c) approx 1 / (1 + exp(-beta * margin_k))
    # where margin_k is the signed distance to the tube boundary (positive inside).
    # For a box [x_min, x_max] x [y_min, y_max] x [0,1]:
    # margin_x = min(a - x_min, x_max - a)
    # margin_y = min(b - y_min, y_max - b)
    # margin = min(margin_x, margin_y)
    # If margin > 0, inside. If margin < 0, outside.
    
    # Precompute i and j for each tube
    i_vals = i_flat.astype(np.float64)
    j_vals = j_flat.astype(np.float64)
    
    # We will use a fixed set of Monte Carlo points to estimate the volume.
    # Using a fixed set allows for consistent gradients (stochastic gradient is noisy, 
    # but with enough points it's okay. However, for L-BFGS-B, deterministic is better 
    # if possible, but 500k points is too many for full gradient. 
    # We can use a smaller subset for the optimization loop, or use the full set 
    # if we vectorize efficiently. 
    # Let's try a subset of 10000 points for speed during optimization, 
    # then maybe refine? Or just use 50000.
    # The evaluator uses 500,000 points. 
    # If we optimize for a subset, we might overfit to that subset.
    # But the structure is simple. 
    # Let's use 20,000 points for the optimization to balance speed and accuracy.
    
    rng = np.random.default_rng(42)
    # We need the box boundaries X0, X1, Y0, Y1 which depend on the positions.
    # This makes the objective function non-stationary (the domain changes).
    # This is tricky for standard optimizers.
    
    # Alternative: Fix the domain? No, the task says the box is defined by the positions.
    # V = (fraction inside) * (X1-X0) * (Y1-Y0)
    
    # Let's define the objective function
    def objective(params):
        # Check time
        if time.time() > deadline - safety_margin:
            # Return a large value to stop, or just return current best?
            # L-BFGS-B doesn't have a clean "abort" that returns best so far easily without custom callback.
            # We'll rely on the callback to stop.
            pass
            
        x = params[:n_tubes]
        y = params[n_tubes:]
        
        # Calculate Box Bounds
        # X0 = min(x_k), X1 = max(x_k + (i_k+1)/N)
        # Y0 = min(y_k), Y1 = max(y_k + (j_k+1)/N)
        
        # (i+1)/N is the width of the square in x direction?
        # Task: "square [x_k + c*i/8, x_k + c*i/8 + 1/8]"
        # So width is 1/8 = 1/N.
        # At c=1, x range is [x_k + i/N, x_k + i/N + 1/N]
        # So max x extent for tube k is x_k + (i_k+1)/N
        # Min x extent is x_k (at c=0)
        
        x_extents_max = x + (i_vals + 1) / N
        y_extents_max = y + (j_vals + 1) / N
        
        X0 = np.min(x)
        X1 = np.max(x_extents_max)
        Y0 = np.min(y)
        Y1 = np.max(y_extents_max)
        
        if X1 - X0 < 1e-9 or Y1 - Y0 < 1e-9:
            return 1e10
        
        # Generate sample points
        # To be consistent with the evaluator's logic but faster, we use a fixed set.
        # However, the evaluator uses rng.uniform(X0, X1, ...).
        # If we use fixed points in [0,1] and scale them, the "fraction inside"
        # is invariant to scaling, but the volume term (X1-X0)(Y1-Y0) changes.
        # Let's generate points in the unit cube [0,1]^3 and map them.
        
        # a = X0 + u * (X1-X0)
        # b = Y0 + v * (Y1-Y0)
        # c = w
        
        # We need a fixed set of (u,v,w) in [0,1]^3.
        # Let's use 20,000 points.
        # We can't define them inside objective if they change, but they don't.
        # They are just coordinates in the normalized space.
        
        if not hasattr(objective, 'sample_points'):
            # Generate once
            # Use a deterministic set for reproducibility in optimization
            # The evaluator uses a specific seed, but for optimization, any fixed set
            # that represents the volume well is fine. 
            # Actually, to avoid overfitting to a specific noise pattern, 
            # using a large enough deterministic set (like Sobol or just random) is good.
            # Let's use random with a fixed seed.
            obj_rng = np.random.default_rng(123)
            u = obj_rng.uniform(0, 1, 20000)
            v = obj_rng.uniform(0, 1, 20000)
            w = obj_rng.uniform(0, 1, 20000)
            objective.sample_points = (u, v, w)
            
        u, v, w = objective.sample_points
        
        a = X0 + u * (X1 - X0)
        b = Y0 + v * (Y1 - Y0)
        c = w
        
        # Calculate margin for each tube
        # Tube k at height c:
        # x_range: [x_k + c*i_k/N, x_k + c*i_k/N + 1/N]
        # y_range: [y_k + c*j_k/N, y_k + c*j_k/N + 1/N]
        
        # Shifts
        shift_x = c[None, :] * i_vals[:, None] / N  # (n_tubes, n_points)
        shift_y = c[None, :] * j_vals[:, None] / N
        
        # Intervals
        x_min = x[:, None] + shift_x
        x_max = x_min + 1/N
        y_min = y[:, None] + shift_y
        y_max = y_min + 1/N
        
        # Point coordinates
        a_flat = a[None, :]
        b_flat = b[None, :]
        
        # Margin in x
        # dist to left: a - x_min
        # dist to right: x_max - a
        margin_x = np.minimum(a_flat - x_min, x_max - a_flat)
        
        # Margin in y
        margin_y = np.minimum(b_flat - y_min, y_max - y_min) # Wait, y_max - b_flat
        margin_y = np.minimum(b_flat - y_min, y_max - b_flat)
        
        # Combined margin (distance to boundary of the square)
        margin = np.minimum(margin_x, margin_y)
        
        # Sigmoid approximation
        # beta controls sharpness
        beta = 50.0
        # I_k approx 1 / (1 + exp(-beta * margin))
        # To avoid overflow in exp, clip margin
        margin_clipped = np.clip(margin, -10, 10)
        I_k = 1.0 / (1.0 + np.exp(-beta * margin_clipped))
        
        # Union Indicator
        # P(not in any) = prod(1 - I_k)
        # P(in union) = 1 - prod(1 - I_k)
        
        # Log-sum-exp trick for stability? 
        # log(1 - I_k) is -log(1 + exp(beta*margin))
        # This can be unstable.
        # Let's just compute prod. With 64 tubes, product of 64 numbers < 1 is fine.
        
        one_minus_I = 1.0 - I_k
        # If I_k is 1, one_minus_I is 0.
        # Product might underflow to 0, which is good (means union is 1).
        
        # Sum over tubes for each point
        # We want mean over points of (1 - prod_k one_minus_I_k(point))
        
        # To save memory, we can process in batches or just let numpy do it.
        # 20000 points * 64 tubes = 1.28 million floats. ~10MB. Fine.
        
        prod_om_i = np.prod(one_minus_I, axis=0)
        union_indicator = 1.0 - prod_om_i
        
        fraction_inside = np.mean(union_indicator)
        
        volume = fraction_inside * (X1 - X0) * (Y1 - Y0)
        
        return volume

    # Gradient
    # Numerical gradient is slow but robust.
    # Analytical gradient:
    # V = (1/M) * sum_p [ 1 - prod_k (1 - I_{k,p}) ] * Area
    # dV/dx_m = (Area/M) * sum_p [ - prod_k (1 - I_{k,p}) * sum_{k} d(1-I_{k,p})/dx_m / (1-I_{m,p}) ]
    #            * (d Area / dx_m) + ...
    # This is complex due to the moving boundaries X0, X1, Y0, Y1.
    # Let's use finite differences or just let L-BFGS-B use numerical gradients?
    # L-BFGS-B with numerical gradients on 128 variables with 20k point evaluation might be slow.
    # Each gradient step requires ~128 function evaluations.
    # 128 * 20k * 64 ops = ~160M ops per gradient step.
    # 100 steps = 16B ops. Might be too slow in Python/NumPy if not vectorized well.
    # The vectorization above is quite efficient.
    
    # Let's try with numerical gradient first, but limit iterations.
    # Or implement a simplified gradient?
    # Given the time constraint (600s), let's be conservative.
    
    # Callback to stop if time is up
    stop_flag = {'stop': False}
    
    def callback(xk):
        if time.time() > deadline - safety_margin:
            stop_flag['stop'] = True
            return True
        return False

    # Run optimization
    # Bounds: None, positions can be anywhere.
    # However, to keep the box small, we might want to constrain? No, let it find it.
    
    try:
        res = minimize(
            objective, 
            x0, 
            method='L-BFGS-B', 
            options={
                'maxiter': 50, 
                'ftol': 1e-8, 
                'gtol': 1e-5,
                'maxfun': 200
            },
            callback=callback
        )
        
        if res.success or stop_flag['stop']:
            best_params = res.x
        else:
            # If it didn't converge, check if it improved
            if res.fun < objective(x0):
                best_params = res.x
            else:
                best_params = x0
                
    except Exception as e:
        # Fallback to initial guess
        best_params = x0
        print(f"Optimization failed: {e}")

    # If time is running out, return current best
    if time.time() > deadline - 10:
        pass # Just return what we have

    best_x = best_params[:n_tubes]
    best_y = best_params[n_tubes:]
    
    positions = np.stack([best_x, best_y], axis=1)
    
    return positions

# EVOLVE-BLOCK-END
