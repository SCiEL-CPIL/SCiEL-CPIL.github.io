import numpy as np
import time
import math

N = 100

def _energy_and_grad(x):
    """
    Calculate log energy and Riemannian gradient.
    x: (N, 3) array of points on unit sphere.
    """
    # diff[i, j] = x_i - x_j
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    # dist_sq[i, j] = ||x_i - x_j||^2
    dist_sq = np.sum(diff**2, axis=2)
    
    # Avoid division by zero for i==j
    # We set diagonal to 1, then mask it out later
    np.fill_diagonal(dist_sq, 1.0)
    
    # Gradient of E w.r.t x_i:
    # d/dx_i ( - log ||x_i - x_j|| ) = - (x_i - x_j) / ||x_i - x_j||^2
    # Sum over j != i.
    # g_i = - sum_{j != i} (x_i - x_j) / dist_sq_ij
    
    weights = 1.0 / dist_sq
    np.fill_diagonal(weights, 0.0)
    
    grad = -np.sum(weights[:, :, np.newaxis] * diff, axis=1)
    
    # Project gradient to tangent space
    # grad_proj = grad - (grad . x) x
    proj_coeff = np.sum(grad * x, axis=1)
    grad_tangent = grad - proj_coeff[:, np.newaxis] * x
    
    # Calculate energy using math.fsum for precision
    mask = ~np.eye(N, dtype=bool)
    log_dist_sq = np.log(dist_sq)
    log_dist_sq[~mask] = 0.0
    # Use math.fsum for more accurate summation
    energy = -0.25 * math.fsum(log_dist_sq.flatten().tolist())
    
    return energy, grad_tangent

def _retract(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms < 1e-15, 1.0, norms)
    return x / norms

def _optimize_single(start_x, max_iter=2000, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8,
                     time_limit=None, start_time=None):
    if start_time is None:
        start_time = time.time()
    if time_limit is None:
        time_limit = 110.0
    
    x = start_x.copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    
    best_x = x.copy()
    best_e, _ = _energy_and_grad(x)
    
    # Adaptive learning rate
    current_lr = lr
    no_improve_count = 0
    patience = 200
    
    for t in range(1, max_iter + 1):
        # Check time limit
        if time.time() > time_limit:
            break
        
        e, g = _energy_and_grad(x)
        
        if e < best_e:
            best_e = e
            best_x = x.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            # Reduce learning rate if no improvement for a while
            if no_improve_count > patience and current_lr > 1e-6:
                current_lr *= 0.5
                no_improve_count = 0
        
        # Adam update
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)
        
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        
        # Step in Euclidean space
        x_new = x - current_lr * m_hat / (np.sqrt(v_hat) + eps)
        
        # Retract to sphere
        x = _retract(x_new)
        
        # Check convergence more frequently
        if t % 50 == 0 and np.linalg.norm(g) < 1e-7:
            break
            
    return best_x, best_e

def construct():
    start_time = time.time()
    deadline = start_time + 105.0  # 15s safety margin
    
    best_x = None
    best_e = float('inf')
    
    # Initialization strategies
    inits = []
    
    # 1. Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    inits.append(fib)
    
    # 2. Random
    np.random.seed(42)
    rand = np.random.randn(N, 3)
    rand /= np.linalg.norm(rand, axis=1, keepdims=True)
    inits.append(rand)
    
    # 3. Random with different seed
    np.random.seed(123)
    rand2 = np.random.randn(N, 3)
    rand2 /= np.linalg.norm(rand2, axis=1, keepdims=True)
    inits.append(rand2)
    
    # 4. Another random with different seed
    np.random.seed(456)
    rand3 = np.random.randn(N, 3)
    rand3 /= np.linalg.norm(rand3, axis=1, keepdims=True)
    inits.append(rand3)
    
    # 5. Grid-like initialization
    # Create a rough grid on the sphere
    n_theta = 10
    n_phi = 10
    theta_vals = np.linspace(0, np.pi, n_theta + 2)[1:-1]
    phi_vals = np.linspace(0, 2*np.pi, n_phi + 1)[:-1]
    grid_points = []
    for theta in theta_vals:
        for phi in phi_vals:
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            grid_points.append([x, y, z])
    # Add poles
    grid_points.append([0, 0, 1])
    grid_points.append([0, 0, -1])
    grid_arr = np.array(grid_points)
    # Normalize
    grid_arr /= np.linalg.norm(grid_arr, axis=1, keepdims=True)
    # If we have more than N points, randomly select N
    if len(grid_arr) > N:
        idx = np.random.choice(len(grid_arr), N, replace=False)
        grid_arr = grid_arr[idx]
    elif len(grid_arr) < N:
        # Pad with random points
        extra = np.random.randn(N - len(grid_arr), 3)
        extra /= np.linalg.norm(extra, axis=1, keepdims=True)
        grid_arr = np.vstack([grid_arr, extra])
    inits.append(grid_arr)
    
    # Run optimization for each initialization
    for init in inits:
        if time.time() > deadline - 15:
            break
            
        # Run optimization with adaptive learning rate
        x_opt, e_opt = _optimize_single(init, max_iter=4000, lr=0.03,
                                        time_limit=deadline - 15, start_time=start_time)
        
        if e_opt < best_e:
            best_e = e_opt
            best_x = x_opt
    
    # If we have time, do a final polish on the best solution with smaller LR
    if best_x is not None and time.time() < deadline - 5:
        x_polished, e_polished = _optimize_single(best_x, max_iter=8000, lr=0.005,
                                                  time_limit=deadline - 3, start_time=start_time)
        if e_polished < best_e:
            best_x = x_polished
            best_e = e_polished
            
    return best_x
