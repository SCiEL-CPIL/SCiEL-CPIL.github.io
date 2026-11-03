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

def _energy_objective_and_grad(flat_pos, r_target):
    """
    Smooth energy objective: penalize pair overlaps and wall violations.
    E = 0.5 * sum_{i<j} max(0, 2*r_target - d_ij)^2 + 0.5 * sum_{i,k} max(0, r_target - (HALF - |x_ik|))^2
    This is C^1 smooth and aggregates all violations.
    """
    pos = flat_pos.reshape(N, 3)
    
    # Pairwise distances
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Wall distances
    d_walls = HALF - np.abs(pos)
    
    # Compute violations
    pair_viol = np.maximum(0.0, 2.0 * r_target - dists)
    wall_viol = np.maximum(0.0, r_target - d_walls)
    
    energy = 0.5 * np.sum(pair_viol**2) + 0.5 * np.sum(wall_viol**2)
    
    # Gradient
    grad_pos = np.zeros_like(pos)
    
    # Pair gradient: for each violating pair, push apart
    for i in range(N):
        for j in range(i+1, N):
            d = dists[i, j]
            if d < 2.0 * r_target and d > 1e-14:
                # dE/dd_ij = -(2*r_target - d), dE/dx_i = -(2*r_target - d) * (x_i - x_j)/d
                coeff = (2.0 * r_target - d) / d
                vec = pos[i] - pos[j]
                grad_pos[i] -= coeff * vec
                grad_pos[j] += coeff * vec
    
    # Wall gradient
    for i in range(N):
        for k in range(3):
            if d_walls[i, k] < r_target:
                s = np.sign(pos[i, k])
                if s == 0:
                    s = 1.0
                grad_pos[i, k] += (r_target - d_walls[i, k]) * s
    
    return energy, -grad_pos.flatten()

def _energy_minimize(pos, r_target, max_iter=200, ftol=1e-12, gtol=1e-10):
    """Minimize energy for a fixed target radius."""
    bounds = [(-HALF, HALF)] * (N * 3)
    try:
        res = minimize(
            lambda x: _energy_objective_and_grad(x, r_target)[0],
            pos.flatten(),
            jac=lambda x: _energy_objective_and_grad(x, r_target)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol}
        )
        if res.success or res.fun < 1e-8:
            new_pos = res.x.reshape(N, 3)
            if np.all(np.isfinite(new_pos)):
                return new_pos
    except Exception:
        pass
    return pos

def _radius_increase_search(pos, r_start, r_end, n_steps=50, deadline=None):
    """
    Iteratively increase target radius and minimize energy at each step.
    This escapes local minima of the hard-min objective.
    """
    r_values = np.linspace(r_start, r_end, n_steps)
    current_pos = pos.copy()
    
    for r in r_values:
        if deadline is not None and time.time() > deadline:
            break
        current_pos = _energy_minimize(current_pos, r, max_iter=100, ftol=1e-12, gtol=1e-10)
    
    return current_pos

def _polish(pos, max_iter=500, ftol=1e-12, gtol=1e-10):
    """High-precision polish pass using the hard-min objective."""
    bounds = [(-0.5, 0.5)] * (N * 3)
    try:
        # Use energy-based polish with current best radius as target
        val, _, _, _, _ = compute_min_dist(pos)
        r_target = val * 1.001
        polished = _energy_minimize(pos, r_target, max_iter=max_iter, ftol=ftol, gtol=gtol)
        if np.all(np.isfinite(polished)):
            return polished
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
    
    # Phase 1: Energy-based radius increase from multiple starting points
    phase1_deadline = start_time + 100.0
    attempt = 0
    
    while time.time() < phase1_deadline:
        attempt += 1
        
        # Choose perturbation strategy
        if attempt % 10 == 0:
            # Random restart
            start_pos = rng_global.uniform(-0.45, 0.45, size=(N, 3))
        elif attempt % 10 == 1:
            # Small perturbation
            scale = 0.01
            start_pos = best_pos + rng_global.normal(0, scale, size=(N, 3))
        elif attempt % 10 == 2:
            # Medium perturbation
            scale = 0.04
            start_pos = best_pos + rng_global.normal(0, scale, size=(N, 3))
        elif attempt % 10 == 3:
            # Large perturbation
            scale = 0.08
            start_pos = best_pos + rng_global.normal(0, scale, size=(N, 3))
        elif attempt % 10 == 4:
            # Very small perturbation
            scale = 0.003
            start_pos = best_pos + rng_global.normal(0, scale, size=(N, 3))
        elif attempt % 10 == 5:
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
        
        # Energy-based radius increase
        # Start from a radius below current best to allow exploration
        r_start = best_r * 0.85
        r_end = best_r * 1.05
        
        new_pos = _radius_increase_search(start_pos, r_start, r_end, n_steps=30, deadline=phase1_deadline)
        
        # Evaluate
        val, _, _, _, _ = compute_min_dist(new_pos)
        
        # If improved, try gradient refinement
        if val > best_r:
            best_r = val
            best_pos = new_pos.copy()
            
            # Polish with gradient method
            if time.time() < phase1_deadline - 5.0:
                polished = _polish(best_pos, max_iter=200, ftol=1e-13, gtol=1e-11)
                val2, _, _, _, _ = compute_min_dist(polished)
                if val2 > best_r:
                    best_r = val2
                    best_pos = polished
    
    # Phase 2: Intensive energy-based refinement around best
    phase2_deadline = deadline - 15.0
    while time.time() < phase2_deadline:
        # Small perturbations around best
        scale = 0.005
        start_pos = best_pos + rng_global.normal(0, scale, size=(N, 3))
        start_pos = np.clip(start_pos, -0.5, 0.5)
        
        # Energy-based radius increase with tighter range
        r_start = best_r * 0.98
        r_end = best_r * 1.02
        
        new_pos = _radius_increase_search(start_pos, r_start, r_end, n_steps=20, deadline=phase2_deadline)
        
        val, _, _, _, _ = compute_min_dist(new_pos)
        if val > best_r:
            best_r = val
            best_pos = new_pos.copy()
    
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
