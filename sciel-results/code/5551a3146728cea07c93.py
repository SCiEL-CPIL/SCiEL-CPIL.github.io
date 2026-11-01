import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    def compute_radius(c):
        if not np.all(np.isfinite(c)):
            return 0.0
        if np.any(c > 0.5) or np.any(c < -0.5):
            return 0.0
        wall_dists = 0.5 - np.abs(c)
        r_wall = np.min(wall_dists)
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        r_pair = min_pair_dist / 2.0
        
        return min(r_wall, r_pair)

    def relax_step(c, r_target, alpha, noise_level):
        c_new = c.copy()
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        dists_safe = np.where(dists > 1e-10, dists, 1e-10)
        
        viol = np.maximum(0, 2 * r_target - dists)
        factors = viol / dists_safe
        np.fill_diagonal(factors, 0.0)
        
        row_sums = np.sum(factors, axis=1)
        force_pair = c * row_sums[:, np.newaxis] - (factors @ c)
        
        abs_c = np.abs(c)
        dist_to_wall = 0.5 - abs_c
        viol_wall = np.maximum(0, r_target - dist_to_wall)
        signs = np.sign(c)
        force_wall = -signs * viol_wall
        
        force_total = force_pair + force_wall
        
        c_new = c + alpha * force_total
        
        if noise_level > 0:
            c_new += np.random.normal(0, noise_level, c.shape)
            
        c_new = np.clip(c_new, -0.5, 0.5)
        
        return c_new

    # Initialization
    nx, ny, nz = 5, 5, 4
    xs = np.linspace(-0.4, 0.4, nx)
    ys = np.linspace(-0.4, 0.4, ny)
    zs = np.linspace(-0.4, 0.4, nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    c = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    current_r = compute_radius(c)
    best_r = current_r
    best_c = c.copy()
    
    r_target = current_r
    alpha = 0.1
    noise = 0.0
    stagnant = 0
    
    while time.time() < deadline:
        c = relax_step(c, r_target, alpha, noise)
        achieved_r = compute_radius(c)
        
        if achieved_r > best_r:
            best_r = achieved_r
            best_c = c.copy()
            stagnant = 0
        else:
            stagnant += 1
            
        if achieved_r >= r_target - 1e-5:
            r_target = achieved_r + 0.0005
            noise = max(0.0, noise * 0.8)
        else:
            if stagnant > 100:
                noise = 0.01
                stagnant = 0
                if r_target > current_r:
                    r_target = current_r + 0.001
            if noise < 0.01:
                noise = 0.001

    best_c = np.clip(best_c, -0.5, 0.5)
    if not np.all(np.isfinite(best_c)):
        xs = np.linspace(-0.4, 0.4, nx)
        ys = np.linspace(-0.4, 0.4, ny)
        zs = np.linspace(-0.4, 0.4, nz)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
        best_c = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
        
    return best_c
