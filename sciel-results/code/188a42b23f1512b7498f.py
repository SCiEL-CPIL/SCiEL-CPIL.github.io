"""Improved packing for 50 equal circles in a unit square."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_DEADLINE = time.time() + 110  # safety margin before 120s kill


def _radius_for(centers):
    """Return the common radius for a set of centers."""
    n = len(centers)
    if n < 2:
        return 0.5
    # Wall distances
    wall = np.minimum(0.5 - np.abs(centers), axis=1)
    r_wall = np.min(wall)
    # Pairwise distances / 2
    if n > 1:
        # Use broadcasting for pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        r_pair = np.min(dists) / 2.0
    else:
        r_pair = np.inf
    return min(r_wall, r_pair)


def _neg_radius(centers_flat):
    """Objective: negative radius (to maximize)."""
    centers = centers_flat.reshape(-1, 2)
    r = _radius_for(centers)
    return -r


def _hexagonal_init():
    """Create a hexagonal-ish packing initial configuration."""
    # 50 circles: try a 7x7 grid with some removed, or rows of 7,7,7,7,7,7,7,1
    # Better: use rows with staggered positions
    # 50 = 7*7 + 1, but let's do rows of about 7
    # Actually, let's use a proper hexagonal pattern
    
    # Row-based hexagonal packing
    # r_hex = 0.07137... for 50 circles
    # Let's just create a reasonable initial guess
    
    # Use a grid with jitter
    m = 7  # 7x7 = 49, plus 1
    # Place 49 in a 7x7 grid, 1 extra
    
    # Better approach: start with the known good structure
    # Boll et al. packing is complex, so let's use a good heuristic init
    
    # Create rows with hexagonal offset
    rows = [7, 7, 7, 7, 7, 7, 7, 1]  # 49 + 1 = 50
    # Actually that's 8 rows. Let me use 5 rows of 10 or 7 rows.
    
    # Simpler: 7 columns, ~7 rows
    # Let's use a staggered grid
    n_per_row = 7
    n_rows = (N + n_per_row - 1) // n_per_row  # 8 rows
    
    centers = []
    # Use radius estimate for spacing
    r_est = 0.07
    dx = 2 * r_est
    dy = 2 * r_est * np.sqrt(3) / 2  # hexagonal row spacing
    
    # Center the pattern
    for i in range(n_rows):
        n_in_row = n_per_row
        if i == n_rows - 1:
            n_in_row = N - i * n_per_row
        if n_in_row <= 0:
            continue
        # Stagger odd rows
        offset = (i % 2) * dx / 2
        for j in range(n_in_row):
            x = (j - (n_per_row - 1) / 2.0) * dx + offset
            y = (i - (n_rows - 1) / 2.0) * dy
            centers.append([x, y])
    
    centers = np.array(centers[:N])
    return centers


def _optimize_from(centers0, time_budget):
    """Run local optimization starting from centers0."""
    flat0 = centers0.flatten()
    
    # Use Nelder-Mead which doesn't need gradients
    try:
        res = minimize(
            _neg_radius,
            flat0,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10,
                     'adaptive': True}
        )
        if res.success or res.fun < 0:
            best = res.x.reshape(-1, 2)
            return best
    except Exception:
        pass
    
    # Fallback to coordinate ascent / simple gradient-like method
    best = centers0.copy()
    best_r = _radius_for(best)
    
    step = 0.01
    for iteration in range(3000):
        if time.time() > time_budget:
            break
        improved = False
        for i in range(N):
            for dim in range(2):
                # Try small steps
                for delta in [step, -step]:
                    new_c = best.copy()
                    new_c[i, dim] += delta
                    # Clamp to square
                    new_c = np.clip(new_c, -0.5 + 0.001, 0.5 - 0.001)
                    new_r = _radius_for(new_c)
                    if new_r > best_r + 1e-12:
                        best = new_c
                        best_r = new_r
                        improved = True
                        break
                if improved:
                    break
        if not improved:
            # Reduce step size
            step *= 0.7
            if step < 1e-9:
                break
    
    return best


def construct():
    start_time = time.time()
    
    # Initial configuration
    centers0 = _hexagonal_init()
    
    # Run optimization with time budget
    time_budget = start_time + 100  # leave 10s margin
    
    best = _optimize_from(centers0, time_budget)
    best_r = _radius_for(best)
    
    # Check deadline
    if time.time() > _DEADLINE:
        return best
    
    # Try a few random restarts if time permits
    rng = np.random.default_rng(42)
    for restart in range(5):
        if time.time() > time_budget - 15:
            break
        
        # Perturb initial config
        noise = rng.normal(0, 0.02, (N, 2))
        perturbed = np.clip(centers0 + noise, -0.49, 0.49)
        
        candidate = _optimize_from(perturbed, time_budget)
        cand_r = _radius_for(candidate)
        
        if cand_r > best_r:
            best = candidate
            best_r = cand_r
    
    return best
# EVOLVE-BLOCK-END
