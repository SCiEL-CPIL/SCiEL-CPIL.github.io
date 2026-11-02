import numpy as np
import time
from scipy.optimize import minimize

N = 50
START_TIME = time.time()
TIME_LIMIT = 110.0  # Safety margin for 120s limit

def soft_min(x, alpha=100.0):
    """
    Compute the soft minimum of array x.
    x: 1D array of values.
    alpha: sharpness parameter.
    """
    # For numerical stability, subtract max
    x_max = np.max(x)
    # If all are very negative, exp will underflow, but that's fine, result will be close to min
    # We want -1/alpha * log(sum(exp(-alpha * x)))
    # = -1/alpha * log(sum(exp(-alpha * (x - x_max) * -1))) ? No.
    # Let y = -alpha * x.
    # soft_min = -1/alpha * log(sum(exp(y)))
    # = -1/alpha * (y_max + log(sum(exp(y - y_max))))
    # = x_max - 1/alpha * log(sum(exp(-alpha * (x - x_max))))
    
    # Actually, standard log-sum-exp trick:
    # log(sum(exp(-alpha * x))) = -alpha * x_max + log(sum(exp(-alpha * (x - x_max))))
    # So soft_min = -1/alpha * (-alpha * x_max + log(sum(exp(-alpha * (x - x_max)))))
    #             = x_max - 1/alpha * log(sum(exp(-alpha * (x - x_max))))
    
    shifted = -alpha * (x - x_max)
    # Clip shifted to prevent overflow in exp (though x - x_max <= 0, so shifted >= 0)
    # Wait, x - x_max is negative or zero. -alpha * (x - x_max) is positive or zero.
    # exp(positive) can overflow if alpha is large and x is far from x_max.
    # But x is distances ~ 0.1 to 0.5. x_max ~ 0.5. x - x_max ~ -0.4.
    # -100 * (-0.4) = 40. exp(40) is ~ 2e17. Sum of 1000 such terms is ~ 1e20. Log is fine.
    # However, if alpha is very large, we might get issues.
    
    sum_exp = np.sum(np.exp(shifted))
    return x_max - (1.0 / alpha) * np.log(sum_exp)

def get_radius(centers, alpha=100.0):
    """
    Calculate the smooth radius for a set of centers.
    centers: (N, 3) array.
    """
    # Pairwise distances
    # centers shape (N, 3)
    # diff shape (N, N, 3)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # We only need i < j
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    r_pairs = 0.5 * pair_dists
    
    # Boundary distances
    # For each center, min(0.5 - |c_x|, 0.5 - |c_y|, 0.5 - |c_z|)
    # Actually, the constraint is min over all dimensions and all spheres.
    # So we can just flatten all boundary terms.
    abs_c = np.abs(centers)
    bounds = 0.5 - abs_c
    r_bounds = bounds.flatten()
    
    # Combine all candidate radii
    all_radii = np.concatenate([r_pairs, r_bounds])
    
    return soft_min(all_radii, alpha)

def objective(centers_flat, alpha=100.0):
    """
    Objective function for minimization.
    We want to maximize radius, so minimize -radius.
    """
    centers = centers_flat.reshape(N, 3)
    r = get_radius(centers, alpha)
    return -r

def optimize_from_init(init_centers, alpha=100.0, max_iter=200):
    """
    Run L-BFGS-B optimization starting from init_centers.
    """
    bounds = [(-0.5, 0.5)] * (N * 3)
    x0 = init_centers.flatten()
    
    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                   options={'maxiter': max_iter, 'ftol': 1e-8, 'gtol': 1e-6, 'alpha': alpha})
    
    best_centers = res.x.reshape(N, 3)
    best_r = get_radius(best_centers, alpha)
    
    return best_centers, best_r

def init_fcc():
    """
    Generate FCC lattice points inside the cube.
    """
    # FCC points in a unit cell:
    # (0,0,0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)
    # We need to scale this to fit the cube [-0.5, 0.5].
    # A grid of FCC cells.
    # Let's use a 4x4x4 grid of cells?
    # Cell size 0.25.
    # Points per cell: 4.
    # Total points approx 4*4*4*4 = 256.
    # We will generate them, filter to cube, and take first 50.
    
    points = []
    n_cells = 4
    cell_size = 1.0 / n_cells
    offset = -0.5
    
    # FCC basis in cell coordinates (0..1)
    basis = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0]
    ])
    
    for i in range(n_cells):
        for j in range(n_cells):
            for k in range(n_cells):
                for b in basis:
                    x = offset + (i + b[0]) * cell_size
                    y = offset + (j + b[1]) * cell_size
                    z = offset + (k + b[2]) * cell_size
                    # Check if inside cube
                    if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5 and -0.5 <= z <= 0.5:
                        points.append([x, y, z])
                        
    points = np.array(points)
    # Shuffle to avoid bias towards corner
    np.random.shuffle(points)
    
    if len(points) >= N:
        return points[:N]
    else:
        # Pad with random points if not enough (should not happen)
        random_pts = np.random.uniform(-0.5, 0.5, (N - len(points), 3))
        return np.vstack([points, random_pts])

def init_bcc():
    """
    Generate BCC lattice points inside the cube.
    """
    points = []
    n_cells = 4
    cell_size = 1.0 / n_cells
    offset = -0.5
    
    # BCC basis: corners and center
    basis = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.5, 0.5, 0.5]
    ])
    
    for i in range(n_cells):
        for j in range(n_cells):
            for k in range(n_cells):
                for b in basis:
                    x = offset + (i + b[0]) * cell_size
                    y = offset + (j + b[1]) * cell_size
                    z = offset + (k + b[2]) * cell_size
                    if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5 and -0.5 <= z <= 0.5:
                        points.append([x, y, z])
                        
    points = np.array(points)
    np.random.shuffle(points)
    
    if len(points) >= N:
        return points[:N]
    else:
        random_pts = np.random.uniform(-0.5, 0.5, (N - len(points), 3))
        return np.vstack([points, random_pts])

def init_repulsion():
    """
    Initialize with random points and apply repulsion forces.
    """
    centers = np.random.uniform(-0.5, 0.5, (N, 3))
    eta = 0.01
    for step in range(100):
        forces = np.zeros_like(centers)
        for i in range(N):
            for j in range(N):
                if i == j: continue
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                if dist < 1e-6:
                    diff = np.random.uniform(-1, 1, 3)
                    dist = np.linalg.norm(diff)
                # Repulsive force
                f = diff / (dist**3)
                forces[i] += f
        
        # Boundary repulsion
        for i in range(N):
            for k in range(3):
                c = centers[i][k]
                if c > 0.4:
                    forces[i][k] -= 10.0
                elif c < -0.4:
                    forces[i][k] += 10.0
                    
        centers += eta * forces
        centers = np.clip(centers, -0.5, 0.5)
        
    return centers

def construct():
    global START_TIME, TIME_LIMIT
    
    best_r = -np.inf
    best_centers = None
    
    # Initializations
    inits = [
        ("fcc", init_fcc()),
        ("bcc", init_bcc()),
        ("rep", init_repulsion())
    ]
    
    # Add a few random restarts if time permits
    # We will try to run optimization for each init
    
    for name, init_c in inits:
        if time.time() - START_TIME > TIME_LIMIT - 10:
            break
            
        try:
            c_opt, r_opt = optimize_from_init(init_c, alpha=100.0, max_iter=100)
            if r_opt > best_r:
                best_r = r_opt
                best_centers = c_opt
        except Exception as e:
            pass
            
    # If time remains, do a second pass with higher alpha (sharper min) on the best result
    if best_centers is not None and time.time() - START_TIME < TIME_LIMIT - 20:
        try:
            c_opt2, r_opt2 = optimize_from_init(best_centers, alpha=500.0, max_iter=50)
            if r_opt2 > best_r:
                best_r = r_opt2
                best_centers = c_opt2
        except Exception as e:
            pass
            
    # Final polish with even higher alpha if time permits
    if best_centers is not None and time.time() - START_TIME < TIME_LIMIT - 5:
        try:
            c_opt3, r_opt3 = optimize_from_init(best_centers, alpha=2000.0, max_iter=20)
            if r_opt3 > best_r:
                best_r = r_opt3
                best_centers = c_opt3
        except Exception as e:
            pass
            
    if best_centers is None:
        # Fallback to grid
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        best_centers = grid[:N]
        
    return best_centers

# EVOLVE-BLOCK-END
