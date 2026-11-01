"""Hybrid Global-Local Packing: Simulated Annealing + L-BFGS-B Polish."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _score(cents):
    """Compute the packing radius (the score)."""
    n = len(cents)
    # Wall distances
    wall_min = np.min(np.minimum(0.5 - np.abs(cents[:, 0]), 0.5 - np.abs(cents[:, 1])))
    # Pairwise distances
    if n > 1:
        diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_min = np.min(dists) / 2.0
    else:
        pair_min = 0.5
    return min(wall_min, pair_min)

def _hex_seed():
    """Create a hexagonal packing arrangement for 50 circles."""
    best_cents = None
    best_r = 0
    for n_rows in [5, 6, 7, 8]:
        base = N // n_rows
        extra = N % n_rows
        row_counts = [base + (1 if i < extra else 0) for i in range(n_rows)]
        max_rc = max(row_counts)
        r_horiz = 1.0 / (2.0 * max_rc)
        r_vert = 1.0 / (2.0 + (n_rows - 1) * np.sqrt(3.0))
        r = min(r_horiz, r_vert)
        if r <= best_r:
            continue
        centers = []
        total_y = (n_rows - 1) * 2 * r * np.sqrt(3.0) / 2.0
        y_start = -0.5 + r + total_y / 2.0
        for row_idx in range(n_rows):
            rc = row_counts[row_idx]
            y = y_start + row_idx * 2 * r * np.sqrt(3.0) / 2.0
            x_span = (rc - 1) * 2 * r
            x_start = -0.5 + r + x_span / 2.0
            offset = (2 * r * np.sqrt(3.0) / 2.0) * 0.5 * (row_idx % 2)
            for col in range(rc):
                x = x_start + col * 2 * r + offset
                centers.append([x, y])
        centers = np.array(centers[:N])
        r_actual = _score(centers)
        if r_actual > best_r:
            best_r = r_actual
            best_cents = centers.copy()
    if best_cents is None:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_cents = grid[:N]
    return best_cents

def _energy(cents):
    """
    Smooth energy function for Simulated Annealing.
    Minimizing this encourages large minimum distances.
    E = sum(1/d^2) + sum(1/w^2)
    """
    diff = cents[:, np.newaxis, :] - cents[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dists_sq, 1.0) # Avoid div by zero, set to 1 so 1/1=1 (small contribution)
    
    # Pairwise repulsion
    # Use a soft minimum or just sum of inverse squares
    pair_energy = np.sum(1.0 / dists_sq)
    
    # Wall repulsion
    wall_dist_x = 0.5 - np.abs(cents[:, 0])
    wall_dist_y = 0.5 - np.abs(cents[:, 1])
    # If wall_dist is small or negative, penalty is huge
    # Clip to avoid inf
    wall_dist_x = np.maximum(wall_dist_x, 1e-6)
    wall_dist_y = np.maximum(wall_dist_y, 1e-6)
    wall_energy = np.sum(1.0 / (wall_dist_x**2)) + np.sum(1.0 / (wall_dist_y**2))
    
    return pair_energy + wall_energy

def _sa_optimize(init_cents, time_limit=5.0):
    """
    Run Simulated Annealing to minimize energy.
    """
    cents = init_cents.copy()
    current_energy = _energy(cents)
    best_cents = cents.copy()
    best_energy = current_energy
    
    start_time = time.time()
    T = 0.01  # Initial temperature
    alpha = 0.9999  # Cooling rate
    step_size = 0.05
    
    while (time.time() - start_time) < time_limit:
        # Generate move: pick a random circle and move it
        idx = np.random.randint(N)
        delta = np.random.normal(0, step_size, 2)
        new_cents = cents.copy()
        new_cents[idx] += delta
        
        # Keep within bounds (hard constraint for SA steps to avoid huge penalties)
        new_cents = np.clip(new_cents, -0.5, 0.5)
        
        new_energy = _energy(new_cents)
        delta_e = new_energy - current_energy
        
        if delta_e < 0 or np.random.random() < np.exp(-delta_e / T):
            cents = new_cents
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_cents = cents.copy()
        
        T *= alpha
        if T < 1e-8:
            T = 1e-8
            
    return best_cents

def _polish(cents, time_limit=5.0):
    """
    Polish the configuration using L-BFGS-B on violation loss for a specific radius.
    We estimate the radius from the current config and try to maximize it.
    """
    r_est = _score(cents)
    # Try to improve r_est slightly
    r_target = r_est * 1.001
    
    x0 = cents.reshape(-1).copy()
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    def _violation_loss(x, r):
        c = x.reshape(N, 2)
        wall_pen_x = np.maximum(0.0, np.abs(c[:, 0]) - (0.5 - r))
        wall_pen_y = np.maximum(0.0, np.abs(c[:, 1]) - (0.5 - r))
        wall_loss = np.sum(wall_pen_x**2) + np.sum(wall_pen_y**2)
        
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        overlaps = np.maximum(0.0, 2.0 * r - dists)
        pair_loss = np.sum(overlaps**2)
        
        return wall_loss + pair_loss

    start = time.time()
    try:
        result = minimize(
            _violation_loss, x0, args=(r_target,),
            method='L-BFGS-B', bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50, 'maxfun': 500}
        )
        if time.time() - start > time_limit:
            return cents # Timeout, return original
        
        x_best = result.x
        loss_best = _violation_loss(x_best, r_target)
        
        if loss_best < 1e-8:
            return x_best.reshape(N, 2)
        else:
            # Check if the polished config has a better score than original
            r_new = _score(x_best.reshape(N, 2))
            if r_new > r_est:
                return x_best.reshape(N, 2)
            else:
                return cents
    except Exception:
        return cents

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    # 1. Generate initial hexagonal seed
    cents = _hex_seed()
    best_r = _score(cents)
    best_cents = cents.copy()
    
    # 2. Global Search with Simulated Annealing
    # Run multiple SA restarts if time permits
    n_restarts = 0
    max_restarts = 5
    sa_time_per_restart = 10.0
    
    while (time.time() - start_time) < deadline - 20 and n_restarts < max_restarts:
        # Use current best as init or random perturb
        init = best_cents.copy()
        # Add small noise to escape local minima
        init += np.random.normal(0, 0.01, init.shape)
        init = np.clip(init, -0.5, 0.5)
        
        sa_cents = _sa_optimize(init, time_limit=sa_time_per_restart)
        sa_r = _score(sa_cents)
        
        if sa_r > best_r:
            best_r = sa_r
            best_cents = sa_cents.copy()
        
        n_restarts += 1
        
        # If we have time, polish the current best
        remaining = deadline - (time.time() - start_time)
        if remaining > 10:
            polished = _polish(best_cents, time_limit=min(remaining - 5, 15))
            pol_r = _score(polished)
            if pol_r > best_r:
                best_r = pol_r
                best_cents = polished.copy()
        
        # Reduce SA time for subsequent restarts if running out of time
        if remaining < 30:
            sa_time_per_restart = 5.0
        if remaining < 15:
            break

    # Final polish
    remaining = deadline - (time.time() - start_time)
    if remaining > 5:
        final_polish = _polish(best_cents, time_limit=min(remaining - 2, 10))
        final_r = _score(final_polish)
        if final_r > best_r:
            best_cents = final_polish
    
    best_cents = np.clip(best_cents, -0.5, 0.5)
    return best_cents
# EVOLVE-BLOCK-END
