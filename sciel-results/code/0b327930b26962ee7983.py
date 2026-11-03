import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 100
HALF = 0.5

def compute_min_dist(pos):
    """Compute the minimum radius for a given configuration."""
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    min_pair_val = np.min(dists)
    min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
    
    d_walls = HALF - np.abs(pos)
    min_wall_val = np.min(d_walls)
    min_wall_idx = np.unravel_index(np.argmin(d_walls), d_walls.shape)
    
    r_pair = min_pair_val / 2.0
    r_wall = min_wall_val
    
    if r_pair < r_wall:
        return r_pair, 'pair', min_pair_idx, r_pair, r_wall
    else:
        return r_wall, 'wall', min_wall_idx, r_pair, r_wall

def _fast_repulsion_step(pos, r_target, n_iters=50, lr=0.01, wall_margin=0.001):
    """
    Fast vectorized repulsion-based local search using KD-tree for neighbor queries.
    Each sphere is pushed away from nearby spheres and pulled away from walls.
    This is a simple gradient-free method that can escape shallow local minima.
    """
    pos = pos.copy()
    for it in range(n_iters):
        # Build KD-tree for efficient neighbor queries
        tree = cKDTree(pos)
        
        # For each sphere, find neighbors within 2*r_target + small buffer
        search_r = 2.0 * r_target + 0.02
        neighbors = tree.query_ball_point(pos, r=search_r, workers=-1)
        
        forces = np.zeros_like(pos)
        for i in range(N):
            for j in neighbors[i]:
                if i == j:
                    continue
                d = pos[i] - pos[j]
                dist = np.linalg.norm(d)
                if dist < 1e-12:
                    # Random small push
                    forces[i] += rng_global.normal(0, 1e-4, size=3)
                    continue
                # Repulsion: stronger when closer than 2*r_target
                if dist < 2.0 * r_target:
                    strength = (2.0 * r_target - dist) / (2.0 * r_target)
                    forces[i] += strength * d / dist
                else:
                    # Mild repulsion to maintain spacing
                    strength = 0.1 * (search_r - dist) / search_r
                    forces[i] += strength * d / dist
        
        # Wall forces: push spheres away from walls
        for i in range(N):
            for k in range(3):
                if pos[i, k] > HALF - wall_margin:
                    forces[i, k] -= (pos[i, k] - (HALF - wall_margin)) / wall_margin
                elif pos[i, k] < -HALF + wall_margin:
                    forces[i, k] += ((-HALF + wall_margin) - pos[i, k]) / wall_margin
        
        # Apply forces with adaptive learning rate
        pos += lr * forces
        pos = np.clip(pos, -HALF + 1e-6, HALF - 1e-6)
    
    return pos

def soft_min_objective(flat_pos, beta=50.0):
    """
    Smooth surrogate objective using soft-min.
    We want to maximize r = min(r_pair, r_wall).
    Instead, we minimize -soft_min(r_pair, r_wall) where soft_min is a smooth approximation.
    """
    pos = flat_pos.reshape(N, 3)
    
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Wall distances
    d_walls = HALF - np.abs(pos)
    
    r_pair_min = np.min(dists) / 2.0
    r_wall_min = np.min(d_walls)
    
    # Soft-min of two values (stable form)
    m = min(r_pair_min, r_wall_min)
    if m < -1000:
        return r_pair_min, None
    
    exp_pair = np.exp(-beta * (r_pair_min - m))
    exp_wall = np.exp(-beta * (r_wall_min - m))
    
    soft_r = m - np.log(exp_pair + exp_wall) / beta
    
    # Gradient of r_pair_min
    min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
    i, j = min_pair_idx
    d = dists[i, j]
    
    grad_r_pair = np.zeros_like(pos)
    if d > 1e-12:
        vec = (pos[i] - pos[j]) / d
        grad_r_pair[i] += 0.5 * vec
        grad_r_pair[j] += -0.5 * vec
    
    # Gradient of r_wall_min
    min_wall_idx = np.unravel_index(np.argmin(d_walls), d_walls.shape)
    wi, wk = min_wall_idx
    
    grad_r_wall = np.zeros_like(pos)
    s = np.sign(pos[wi, wk])
    if s == 0:
        s = 1.0
    grad_r_wall[wi, wk] = -s
    
    total_exp = exp_pair + exp_wall
    if total_exp < 1e-300:
        return soft_r, None
    
    grad_soft_r = -(exp_pair * grad_r_pair + exp_wall * grad_r_wall) / total_exp
    
    obj = -soft_r
    grad = -grad_soft_r.flatten()
    
    return obj, grad

def objective_and_grad(flat_pos):
    """Original hard-min objective with analytical gradient."""
    pos = flat_pos.reshape(N, 3)
    
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    min_pair_val = np.min(dists)
    min_pair_idx = np.unravel_index(np.argmin(dists), dists.shape)
    i, j = min_pair_idx
    
    d_walls = HALF - np.abs(pos)
    min_wall_val = np.min(d_walls)
    min_wall_idx = np.unravel_index(np.argmin(d_walls), d_walls.shape)
    wi, wk = min_wall_idx
    
    r_pair = min_pair_val / 2.0
    r_wall = min_wall_val
    
    grad = np.zeros_like(pos)
    
    if r_pair <= r_wall:
        d = min_pair_val
        if d > 1e-12:
            vec = (pos[i] - pos[j]) / d
            grad[i] += 0.5 * vec
            grad[j] += -0.5 * vec
        return -r_pair, -grad.flatten()
    else:
        s = np.sign(pos[wi, wk])
        if s == 0:
            s = 1.0
        grad[wi, wk] = -s
        return -r_wall, -grad.flatten()

def _polish(pos, max_iter=500, ftol=1e-12, gtol=1e-10):
    """High-precision polish pass using the hard-min objective."""
    bounds = [(-0.5, 0.5)] * (N * 3)
    try:
        res = minimize(
            objective_and_grad,
            pos.flatten(),
            jac=True,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
        )
        if res.success or res.fun < 0:
            new_pos = res.x.reshape(N, 3)
            if np.all(np.isfinite(new_pos)):
                return new_pos
    except Exception:
        pass
    return pos

def construct():
    global rng_global
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s limit
    
    rng_global = np.random.default_rng(42)
    
    # Initial guess: Better grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    best_pos = grid[:N].copy()
    
    # Evaluate initial
    val, _, _, _, _ = compute_min_dist(best_pos)
    best_r = val
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    # Phase 1: Stochastic local search with repulsion steps
    # This helps escape local minima that pure gradient descent gets stuck in
    phase1_deadline = start_time + 60.0
    attempt = 0
    
    while time.time() < phase1_deadline:
        attempt += 1
        
        # Choose perturbation strategy
        if attempt % 7 == 0:
            # Random restart
            start_pos = rng_global.uniform(-0.45, 0.45, size=(N, 3))
        elif attempt % 7 == 1:
            # Small perturbation
            scale = 0.01
            start_pos = best_pos + rng_global.normal(0, scale, size=(N, 3))
        elif attempt % 7 == 2:
            # Medium perturbation
            scale = 0.04
            start_pos = best_pos + rng_global.normal(0, scale, size=(N, 3))
        elif attempt % 7 == 3:
            # Large perturbation
            scale = 0.08
            start_pos = best_pos + rng_global.normal(0, scale, size=(N, 3))
        elif attempt % 7 == 4:
            # Very small perturbation
            scale = 0.003
            start_pos = best_pos + rng_global.normal(0, scale, size=(N, 3))
        elif attempt % 7 == 5:
            # Swap-based perturbation: swap two random spheres
            start_pos = best_pos.copy()
            i, j = rng_global.choice(N, size=2, replace=False)
            start_pos[[i, j]] = start_pos[[j, i]]
            start_pos += rng_global.normal(0, 0.01, size=(N, 3))
        else:
            # Kick: randomly displace a few spheres
            start_pos = best_pos.copy()
            n_kick = rng_global.integers(3, 10)
            kick_idx = rng_global.choice(N, size=n_kick, replace=False)
            start_pos[kick_idx] += rng_global.normal(0, 0.05, size=(n_kick, 3))
        
        start_pos = np.clip(start_pos, -0.5, 0.5)
        
        # Apply repulsion-based local search
        current_r = best_r * 0.95  # Target slightly below best to allow exploration
        new_pos = _fast_repulsion_step(start_pos, current_r, n_iters=30, lr=0.008)
        
        # Evaluate
        val, _, _, _, _ = compute_min_dist(new_pos)
        
        # If improved, try gradient refinement
        if val > best_r:
            best_r = val
            best_pos = new_pos.copy()
            
            # Polish with gradient method
            if time.time() < phase1_deadline - 5.0:
                try:
                    res = minimize(
                        objective_and_grad,
                        new_pos.flatten(),
                        jac=True,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
                    )
                    if res.fun < -best_r:
                        refined_pos = res.x.reshape(N, 3)
                        if np.all(np.isfinite(refined_pos)):
                            val2, _, _, _, _ = compute_min_dist(refined_pos)
                            if val2 > best_r:
                                best_r = val2
                                best_pos = refined_pos
                except Exception:
                    pass
        
        # Also try direct gradient-based optimization from perturbed start
        if time.time() < phase1_deadline - 10.0 and attempt % 3 == 0:
            try:
                use_soft = (attempt % 2 == 0)
                if use_soft:
                    res = minimize(
                        lambda x: soft_min_objective(x, beta=50.0)[0],
                        start_pos.flatten(),
                        jac=lambda x: soft_min_objective(x, beta=50.0)[1],
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 80, 'ftol': 1e-10, 'gtol': 1e-8}
                    )
                else:
                    res = minimize(
                        objective_and_grad,
                        start_pos.flatten(),
                        jac=True,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 80, 'ftol': 1e-10, 'gtol': 1e-8}
                    )
                
                if res.fun < -best_r:
                    cand_pos = res.x.reshape(N, 3)
                    if np.all(np.isfinite(cand_pos)):
                        val, _, _, _, _ = compute_min_dist(cand_pos)
                        if val > best_r:
                            best_r = val
                            best_pos = cand_pos
            except Exception:
                pass
    
    # Phase 2: Intensive gradient-based refinement
    phase2_deadline = deadline - 15.0
    while time.time() < phase2_deadline:
        # Small perturbations around best
        scale = 0.005
        start_pos = best_pos + rng_global.normal(0, scale, size=(N, 3))
        start_pos = np.clip(start_pos, -0.5, 0.5)
        
        try:
            res = minimize(
                objective_and_grad,
                start_pos.flatten(),
                jac=True,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'ftol': 1e-11, 'gtol': 1e-9}
            )
            
            if res.fun < -best_r:
                cand_pos = res.x.reshape(N, 3)
                if np.all(np.isfinite(cand_pos)):
                    val, _, _, _, _ = compute_min_dist(cand_pos)
                    if val > best_r:
                        best_r = val
                        best_pos = cand_pos
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() < deadline - 5.0:
        polished_pos = _polish(best_pos, max_iter=500, ftol=1e-13, gtol=1e-11)
        val, _, _, _, _ = compute_min_dist(polished_pos)
        if val > best_r:
            best_r = val
            best_pos = polished_pos
    
    # Ensure valid output
    best_pos = np.clip(best_pos, -HALF, HALF)
    
    return best_pos

# EVOLVE-BLOCK-END
