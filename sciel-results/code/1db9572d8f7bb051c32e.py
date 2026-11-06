"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _radius(centers):
    """Compute max common radius (vectorized)."""
    n = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, np.inf)
    min_pair_dist = np.sqrt(np.min(d2))
    wall = 0.5 - np.max(np.abs(centers), axis=1)
    min_wall = np.min(wall)
    return min(min_pair_dist / 2.0, min_wall)


def _bottleneck_pairs(centers, top_k=5):
    """Return the top_k closest pairs and wall-constrained indices."""
    n = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, np.inf)
    # Get indices of smallest distances
    flat_idx = np.argsort(d2.ravel())[:top_k * 2]
    pairs = set()
    for idx in flat_idx:
        i, j = divmod(idx, n)
        if i < j:
            pairs.add((i, j))
        elif j < i:
            pairs.add((j, i))
    pair_list = list(pairs)[:top_k]
    
    wall = 0.5 - np.max(np.abs(centers), axis=1)
    wall_idx = np.argsort(wall)[:top_k]
    return pair_list, list(wall_idx)


def _relieve_pair(centers, i, j, step=0.01):
    """Move spheres i and j apart along their connecting vector."""
    c = centers.copy()
    diff = c[i] - c[j]
    d = np.linalg.norm(diff)
    if d < 1e-12:
        diff = np.random.randn(3) * 1e-3
        d = np.linalg.norm(diff)
    unit = diff / d
    c[i] += step * unit
    c[j] -= step * unit
    return c


def _relieve_wall(centers, idx, step=0.01):
    """Move sphere idx away from the nearest wall."""
    c = centers.copy()
    # Find which wall is closest
    for k in range(3):
        if c[idx, k] > 0:
            # Near +x or +y or +z wall
            if 0.5 - c[idx, k] == min(0.5 - abs(c[idx, m]) for m in range(3)):
                c[idx, k] -= step
        else:
            if 0.5 + c[idx, k] == min(0.5 - abs(c[idx, m]) for m in range(3)):
                c[idx, k] += step
    return c


def _targeted_refine(centers, time_budget, seed=42):
    """Iteratively relieve bottleneck pairs and wall constraints."""
    start = time.time()
    c = centers.copy()
    best_r = _radius(c)
    best = c.copy()
    rng = np.random.RandomState(seed)
    
    step = 0.02
    patience = 0
    
    while time.time() - start < time_budget:
        pairs, wall_idx = _bottleneck_pairs(c, top_k=3)
        
        improved = False
        for i, j in pairs:
            new_c = _relieve_pair(c, i, j, step)
            new_c = np.clip(new_c, -0.499, 0.499)
            r = _radius(new_c)
            if r > best_r + 1e-10:
                c = new_c
                best_r = r
                best = c.copy()
                improved = True
                break
            elif r > _radius(c) + 1e-10:
                c = new_c
                improved = True
                break
        
        if not improved:
            for idx in wall_idx:
                new_c = _relieve_wall(c, idx, step)
                new_c = np.clip(new_c, -0.499, 0.499)
                r = _radius(new_c)
                if r > best_r + 1e-10:
                    c = new_c
                    best_r = r
                    best = c.copy()
                    improved = True
                    break
                elif r > _radius(c) + 1e-10:
                    c = new_c
                    improved = True
                    break
        
        if not improved:
            patience += 1
            step *= 0.9
            if patience > 50 or step < 1e-5:
                break
        else:
            patience = 0
    
    return best, best_r


def _nelder_mead_polish(centers, time_budget):
    """Light Nelder-Mead polish on the full configuration."""
    from scipy.optimize import minimize
    start = time.time()
    
    def objective(x):
        c = x.reshape(N, 3)
        if np.any(np.abs(c) > 0.499) or not np.all(np.isfinite(c)):
            return 1e10
        return -_radius(c)
    
    x0 = centers.flatten()
    try:
        result = minimize(
            objective, x0, method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        c = result.x.reshape(N, 3)
        if np.all(np.abs(c) <= 0.499) and np.all(np.isfinite(c)):
            r = _radius(c)
            if r > 0:
                return c, r
    except Exception:
        pass
    return centers, _radius(centers)


def _relax_repulsion(centers, time_budget, seed=42):
    """
    Relax centers using repulsive forces to maximize minimum distance.
    Uses a simple velocity-free update with clipping.
    """
    start = time.time()
    c = centers.copy()
    rng = np.random.RandomState(seed)
    step = 0.05
    best_r = _radius(c)
    best_c = c.copy()
    
    # We want to push spheres apart.
    # Force is proportional to 1/d^2 for pairs, and 1/d^2 for walls.
    
    while time.time() - start < time_budget:
        # Compute pair forces
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        d2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(d2, 1e10) # Avoid div by zero
        d = np.sqrt(d2)
        
        # Force magnitude: 1/d^2, but only if d is small? 
        # To maximize min distance, we should push ALL pairs apart, but strongly if they are close.
        # A common trick: F = 1/d^2 for d < threshold, else 0.
        # Or just always repel.
        forces = np.zeros_like(c)
        
        # Vectorized force accumulation is tricky with 1/d^2.
        # Let's use a simpler approach: 
        # For each pair, if d < current_min_dist * 1.5, push apart.
        # But we don't know current_min_dist easily without full scan.
        
        # Alternative: Gradient ascent on min distance is hard.
        # Let's use a potential energy function: E = -sum(1/d) or similar?
        # Minimizing E = sum(1/d^2) maximizes distances.
        
        # Let's compute gradient of E = sum_{i<j} 1/||c_i - c_j||^2
        # dE/dc_i = sum_{j!=i} 2 * (c_i - c_j) / ||c_i - c_j||^4
        
        inv_d2 = 1.0 / d2
        inv_d4 = inv_d2 * inv_d2
        # Multiply diff by inv_d4
        force_terms = diff * inv_d4[:, :, np.newaxis]
        # Sum over j (axis 1)
        pair_forces = np.sum(force_terms, axis=1)
        
        # Wall forces: Push away from walls if close
        # dist to wall: 0.5 - |c|
        # Force: if 0.5 - |c| < threshold, push inwards.
        # Let's use a quadratic penalty for being close to walls.
        # E_wall = sum (max(0, 0.5 - |c| - r_target)^2)? 
        # Simpler: Just add a constant inward force if close to wall.
        
        # Let's add a gentle centering force to prevent clustering at edges?
        # No, we want them to spread.
        
        # Normalize forces?
        # The scale of pair_forces depends on distance.
        
        c_new = c + step * pair_forces
        
        # Wall constraint: clip
        c_new = np.clip(c_new, -0.499, 0.499)
        
        # Check if improved
        r_new = _radius(c_new)
        if r_new > best_r:
            best_r = r_new
            best_c = c_new.copy()
            # Decay step size slowly if improving?
            # step *= 0.99 
        else:
            # If not improving, reduce step size
            step *= 0.9
            if step < 1e-5:
                break
            # Try a smaller step or random jitter
            # c_new = c + rng.normal(0, step, c.shape)
            # c_new = np.clip(c_new, -0.499, 0.499)
            # r_new = _radius(c_new)
            # if r_new > best_r:
            #    ...
        
        c = c_new
        
    return best_c, best_r

def _polish_powell(centers, time_budget):
    """Polish using Powell's method."""
    from scipy.optimize import minimize
    start = time.time()
    
    def objective(x):
        c = x.reshape(N, 3)
        if np.any(np.abs(c) > 0.499) or not np.all(np.isfinite(c)):
            return 1e10
        return -_radius(c)
    
    x0 = centers.flatten()
    try:
        result = minimize(
            objective, x0, method='Powell',
            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        c = result.x.reshape(N, 3)
        if np.all(np.abs(c) <= 0.499) and np.all(np.isfinite(c)):
            r = _radius(c)
            if r > 0:
                return c, r
    except Exception:
        pass
    return centers, _radius(centers)

def construct():
    """Construct 50 sphere centers in a unit cube for maximum radius."""
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    # Initialization: Random subset of 4x4x4 grid
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    # Randomly select 50 points from the 64 grid points
    np.random.seed(42)
    indices = np.random.choice(len(grid), N, replace=False)
    centers = grid[indices].copy()
    
    # Add small noise to break symmetry
    centers += np.random.normal(0, 0.001, centers.shape)
    centers = np.clip(centers, -0.49, 0.49)
    
    best_centers = centers.copy()
    best_r = _radius(best_centers)
    
    # Phase 1: Repulsion Relaxation
    if time.time() - start_time < deadline - 30:
        budget = min(20.0, deadline - 30 - (time.time() - start_time))
        refined, r = _relax_repulsion(best_centers, budget, seed=42)
        if r > best_r:
            best_r = r
            best_centers = refined.copy()
            
    # Phase 2: Targeted Refine (Local search on bottlenecks)
    if time.time() - start_time < deadline - 15:
        budget = min(10.0, deadline - 15 - (time.time() - start_time))
        refined, r = _targeted_refine(best_centers, budget, seed=123)
        if r > best_r:
            best_r = r
            best_centers = refined.copy()
            
    # Phase 3: Powell Polish
    if time.time() - start_time < deadline - 5:
        budget = min(15.0, deadline - 5 - (time.time() - start_time))
        polished, r = _polish_powell(best_centers, budget)
        if r > best_r:
            best_r = r
            best_centers = polished.copy()
            
    # Final validity check
    if not np.all(np.isfinite(best_centers)) or np.any(np.abs(best_centers) > 0.499):
        # Fallback to grid
        best_centers = grid[indices].copy()
        best_centers = np.clip(best_centers, -0.499, 0.499)
    
    return best_centers
# EVOLVE-BLOCK-END
