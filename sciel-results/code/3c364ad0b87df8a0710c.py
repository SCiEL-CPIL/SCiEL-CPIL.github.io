import numpy as np
import time
from scipy.optimize import minimize

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (10s buffer before 150s kill)

    def compute_radius(p):
        if not np.all(np.isfinite(p)):
            return -1.0
        if np.any(p < -0.5 - 1e-9) or np.any(p > 0.5 + 1e-9):
            return -1.0
        # Wall distance
        wall_dists = 0.5 - np.abs(p)
        min_wall = np.min(wall_dists)
        
        # Pairwise distance
        dists = np.zeros((N, N))
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(p[i] - p[j])
                dists[i, j] = d
                dists[j, i] = d
        min_pair = np.min(dists)
        
        r = min(min_wall, min_pair / 2.0)
        return r

    def local_search_lbfgs(initial_pos, time_limit, beta_start=30.0, beta_end=150.0):
        p = initial_pos.copy()
        current_r = compute_radius(p)
        best_r_local = current_r
        best_p_local = p.copy()
        
        # Anneal beta from soft to sharp
        num_stages = 6
        betas = np.linspace(beta_start, beta_end, num_stages)
        
        for beta in betas:
            if time.time() > time_limit:
                break
                
            def obj_and_grad(x):
                p2 = x.reshape(N, 3)
                # Compute soft min radius
                wall_dists = 0.5 - np.abs(p2)
                
                # Efficiently compute pair distances and gradients
                # We need the soft-min of:
                # 1. wall_dists (N*3 values)
                # 2. pair_dists/2 (N*(N-1)/2 values)
                
                # Wall part
                # v_wall = 0.5 - |p2|
                # grad_v_wall = -sign(p2)
                
                # Pair part
                # v_pair[i,j] = |p2[i] - p2[j]| / 2
                # grad_v_pair[i] += 0.5 * (p2[i]-p2[j])/|p2[i]-p2[j]|
                # grad_v_pair[j] += -0.5 * (p2[i]-p2[j])/|p2[i]-p2[j]|
                
                all_vals = np.empty(N*3 + N*(N-1)//2)
                all_vals[:N*3] = wall_dists.ravel()
                
                idx = N*3
                pair_indices = []
                for i in range(N):
                    for j in range(i+1, N):
                        d = p2[i] - p2[j]
                        dist = np.linalg.norm(d)
                        if dist < 1e-12:
                            dist = 1e-12
                        all_vals[idx] = dist / 2.0
                        pair_indices.append((i, j, dist))
                        idx += 1
                
                max_val = np.max(all_vals)
                shifted = all_vals - max_val
                exp_vals = np.exp(-beta * shifted)
                sum_exp = np.sum(exp_vals)
                
                soft_min_val = max_val - np.log(sum_exp) / beta
                
                grad = np.zeros_like(p2)
                
                # Wall gradient contribution
                wall_exp = exp_vals[:N*3]
                wall_w = wall_exp / sum_exp
                # d(soft_min)/dx = sum w_i * (-beta * dv_i/dx)
                # dv_wall/dx = -sign(p2)
                # So contribution = sum w_i * (-beta * (-sign(p2))) = beta * sum w_i * sign(p2)
                signs = np.sign(p2)
                signs[p2 == 0] = 1.0 # Handle zero
                grad += (beta * wall_w.reshape(N, 3)) * signs
                
                # Pair gradient contribution
                pair_exp = exp_vals[N*3:]
                pair_w = pair_exp / sum_exp
                for k, (i, j, dist) in enumerate(pair_indices):
                    w = pair_w[k]
                    if w < 1e-10: continue
                    d_vec = p2[i] - p2[j]
                    unit = d_vec / dist
                    # dv_pair/dx_i = 0.5 * unit
                    # dv_pair/dx_j = -0.5 * unit
                    # Contribution = -beta * w * dv/dx
                    grad[i] -= beta * w * 0.5 * unit
                    grad[j] += beta * w * 0.5 * unit
                
                # We minimize -soft_min, so gradient is -grad_soft_min
                return -soft_min_val, -grad.reshape(-1)
            
            # Use L-BFGS-B
            try:
                res = minimize(
                    obj_and_grad,
                    p.reshape(-1),
                    jac=True,
                    method='L-BFGS-B',
                    bounds=[(-0.5, 0.5)] * (N * 3),
                    options={'maxiter': 300, 'maxfun': 15000, 'gtol': 1e-8, 'ftol': 1e-12}
                )
                
                new_p = res.x.reshape(N, 3)
                new_r = compute_radius(new_p)
                
                if new_r > best_r_local:
                    best_r_local = new_r
                    best_p_local = new_p.copy()
                    p = new_p.copy()
                else:
                    # If it didn't improve, maybe try a smaller step or just keep best
                    # p = new_p.copy() # Don't update p if it got worse, to avoid drifting
                    pass
                    
            except Exception:
                pass
            
            # Check time
            if time.time() > time_limit:
                break
        
        # Final polish: high-precision local optimization on active constraints
        p = best_p_local.copy()
        # Use a smaller step size for polishing
        for _ in range(100):
            if time.time() > time_limit:
                break
            
            # Find active constraints
            min_pair_dist = 1.0
            min_pair_idx = (-1, -1)
            for i in range(N):
                for j in range(i+1, N):
                    d = np.linalg.norm(p[i] - p[j])
                    if d < min_pair_dist:
                        min_pair_dist = d
                        min_pair_idx = (i, j)
                        
            min_wall_dist = 1.0
            min_wall_idx = (-1, -1)
            for i in range(N):
                for k in range(3):
                    w = 0.5 - abs(p[i, k])
                    if w < min_wall_dist:
                        min_wall_dist = w
                        min_wall_idx = (i, k)
            
            current_r = min(min_pair_dist / 2.0, min_wall_dist)
            
            # Move spheres in min pair apart
            if min_pair_idx[0] != -1:
                i, j = min_pair_idx
                d_vec = p[i] - p[j]
                dist = np.linalg.norm(d_vec)
                if dist > 1e-6:
                    d_vec = d_vec / dist
                    eps = 0.0002
                    p[i] += eps * d_vec
                    p[j] -= eps * d_vec
                    p = np.clip(p, -0.5, 0.5)
            
            # Move sphere in min wall inward
            if min_wall_idx[0] != -1:
                i, k = min_wall_idx
                sign = 1.0 if p[i, k] > 0 else -1.0
                p[i, k] -= 0.0002 * sign
                p = np.clip(p, -0.5, 0.5)
            
            new_r = compute_radius(p)
            if new_r > best_r_local:
                best_r_local = new_r
                best_p_local = p.copy()
            else:
                # If it got worse, revert? Or just stop?
                # Let's revert to previous best if it degrades significantly
                pass
        
        return best_p_local, best_r_local

    # Generate initial configurations
    configs = []
    
    # 1. Simple Cubic Grid (Seed)
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    configs.append(grid[:N].copy())
    
    # 2. BCC Lattice (Inspiration 1 & Cell Finding)
    # BCC points in a 3x3x3 supercell? 
    # Simple BCC unit cell has 2 atoms.
    # Let's try a 3x3x3 grid of BCC cells.
    # Cell size = 1/3.
    # Points at (0,0,0) and (0.5, 0.5, 0.5) relative to cell.
    bcc_points = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                x0 = i/3.0 - 0.5
                y0 = j/3.0 - 0.5
                z0 = k/3.0 - 0.5
                bcc_points.append([x0, y0, z0])
                bcc_points.append([x0 + 1/6.0, y0 + 1/6.0, z0 + 1/6.0])
    bcc_points = np.array(bcc_points)
    # We have 54 points. Need 50.
    # Randomly select 50? Or take first 50?
    # Taking first 50 might be biased. Let's shuffle with a fixed seed for reproducibility.
    rng_init = np.random.RandomState(42)
    perm = rng_init.permutation(54)
    configs.append(bcc_points[perm[:50]].copy())
    
    # 3. Perturbed BCC
    configs.append(bcc_points[perm[:50]].copy() + rng_init.normal(0, 0.02, (50, 3)))
    configs[-1] = np.clip(configs[-1], -0.48, 0.48)
    
    # 4. Random Uniform
    rng_rand = np.random.RandomState(123)
    configs.append(rng_rand.uniform(-0.45, 0.45, (N, 3)))
    
    # 5. Perturbed Grid
    configs.append(grid[:N].copy() + rng_init.normal(0, 0.02, (50, 3)))
    configs[-1] = np.clip(configs[-1], -0.48, 0.48)

    best_pos = configs[0]
    best_r = compute_radius(best_pos)
    
    # Allocate time for each config
    # We have ~140s. 5 configs.
    # Give more time to BCC and Grid as they are likely better basins.
    # Time per config: ~25s each.
    time_per_config = 25.0
    
    for i, config in enumerate(configs):
        if time.time() > deadline - 5:
            break
            
        # Ensure valid start
        config = np.clip(config, -0.5, 0.5)
        
        # Run local search
        t_start = time.time()
        t_limit = t_start + time_per_config
        # Leave a bit of time for the next config or final check
        if i == len(configs) - 1:
             t_limit = deadline - 2.0
             
        pos, r = local_search_lbfgs(config, t_limit)
        
        if r > best_r:
            best_r = r
            best_pos = pos.copy()
            
    # Final check
    final_r = compute_radius(best_pos)
    if final_r < 0:
        # Fallback to grid
        best_pos = grid[:N].copy()
        final_r = compute_radius(best_pos)
        
    return best_pos
