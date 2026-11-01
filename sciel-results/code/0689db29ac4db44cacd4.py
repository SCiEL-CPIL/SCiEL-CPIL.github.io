"""Force-directed packing: hexagonal initialization + Langevin dynamics."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

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

def _construct_hexagonal():
    """Create a hexagonal packing arrangement for 50 circles."""
    best_cents = None
    best_r = 0
    
    # Try multiple row configurations
    for n_rows in [5, 6, 7, 8]:
        base = N // n_rows
        extra = N % n_rows
        row_counts = [base + (1 if i < extra else 0) for i in range(n_rows)]
        
        max_rc = max(row_counts)
        # Estimate r to fit
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
        best_r = _score(best_cents)
    
    return best_cents, best_r

def _force_optimize(cents, time_limit=100.0):
    """Optimize using force-directed dynamics (Langevin)."""
    start_time = time.time()
    best_cents = cents.copy()
    best_r = _score(best_cents)
    n = len(cents)
    
    # Parameters
    dt = 0.001
    k_repel = 100.0
    k_wall = 500.0
    damping = 0.90
    temp = 0.01
    rng = np.random.default_rng(42)
    
    # Initial velocities
    v = np.zeros_like(best_cents)
    
    while time.time() - start_time < time_limit:
        # Calculate forces
        forces = np.zeros_like(best_cents)
        
        # Pairwise repulsion
        diff = best_cents[:, np.newaxis, :] - best_cents[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, 1e-6) # Avoid div by 0
        
        # Force = k / d^2 * direction
        # Only add force if d < 2*r (overlapping or close)
        # Actually, for packing, we want to maximize min distance.
        # A common heuristic is to push apart any pair closer than some threshold.
        # Let's use a threshold slightly larger than current best_r to expand the packing.
        # But we don't know the target r. 
        # Standard soft-body: F = k * (d_target - d) if d < d_target.
        # Let's try repulsion for all pairs, scaled by 1/d^2.
        
        inv_d2 = 1.0 / (dists**2)
        np.fill_diagonal(inv_d2, 0)
        
        # Force on i from j: k * (r_i - r_j) / d^3 * d  => k * diff / d^3
        # diff is r_i - r_j. So force is k * diff / d^3.
        # Wait, diff = r_i - r_j. If i is to the right of j, diff is positive.
        # Force should push i to the right (away from j). So +diff is correct.
        
        # We want to avoid huge forces when d is small.
        # Let's use F = k * diff / d^3.
        
        force_mag = k_repel * inv_d2 # This is k/d^2
        # Force vector = force_mag * (diff / d) = k/d^2 * diff/d = k * diff / d^3
        # diff is (n, n, 2). dists is (n, n).
        # We need diff / d^3.
        # d^3 = dists^3.
        
        # Efficient computation:
        # F_i = sum_j k * (r_i - r_j) / |r_i - r_j|^3
        # This is the standard repulsive force in molecular dynamics (Lennard-Jones repulsive part).
        
        # To prevent explosion, we can cap the force or use a softer potential.
        # Let's use F = k * diff / (d^2 + eps)^(3/2) ? No, d^3 is fine if we handle d=0.
        
        # Let's compute:
        # factor = k_repel / (dists**3)
        # factor shape (n, n)
        # forces += sum_j factor[i, j] * diff[i, j]
        
        # But this is O(N^2) per step. N=50, so 2500 ops. Very fast.
        
        # Optimization:
        # dists_sq = np.sum(diff**2, axis=2)
        # np.fill_diagonal(dists_sq, 1e-6)
        # inv_d3 = 1.0 / (dists_sq * np.sqrt(dists_sq))
        # np.fill_diagonal(inv_d3, 0)
        # forces = np.sum(k_repel * diff * inv_d3[:, :, np.newaxis], axis=1)
        
        # Let's do this.
        dists_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dists_sq, 1e-6)
        inv_d3 = 1.0 / (dists_sq * np.sqrt(dists_sq))
        np.fill_diagonal(inv_d3, 0)
        
        f_pairs = np.sum(k_repel * diff * inv_d3[:, :, np.newaxis], axis=1)
        
        # Wall forces
        # Push back if outside [-0.5, 0.5]
        # x < -0.5: push +x. x > 0.5: push -x.
        # Linear spring: F = k_wall * (boundary - pos)
        
        f_walls = np.zeros_like(best_cents)
        
        # Left/Right
        f_walls[:, 0] += k_wall * (-0.5 - best_cents[:, 0]) * (best_cents[:, 0] < -0.5)
        f_walls[:, 0] += k_wall * (0.5 - best_cents[:, 0]) * (best_cents[:, 0] > 0.5)
        
        # Bottom/Top
        f_walls[:, 1] += k_wall * (-0.5 - best_cents[:, 1]) * (best_cents[:, 1] < -0.5)
        f_walls[:, 1] += k_wall * (0.5 - best_cents[:, 1]) * (best_cents[:, 1] > 0.5)
        
        total_forces = f_pairs + f_walls
        
        # Update velocity
        v = damping * v + total_forces * dt + rng.normal(0, temp, size=best_cents.shape)
        
        # Update position
        new_cents = best_cents + v * dt
        
        # Hard constraint: if outside, clamp? 
        # The wall force should keep them in. But if dt is too large, they might jump out.
        # Clamping is safer for validity.
        new_cents = np.clip(new_cents, -0.5 + 1e-9, 0.5 - 1e-9)
        
        # Score
        new_r = _score(new_cents)
        if new_r > best_r:
            best_r = new_r
            best_cents = new_cents.copy()
        else:
            # If score dropped, revert position but keep velocity?
            # Or just revert everything to best?
            # Let's revert position to best, but keep velocity for momentum?
            # No, if we moved to a worse place, the force calculation was based on the worse place.
            # It's safer to just keep the best position and update velocity based on best position forces?
            # Actually, standard MD: if we reject a move, we usually keep the old velocity or re-calc.
            # Here, let's just keep best_cents and re-calc forces in next iteration.
            # But we already updated v.
            # Let's just reset position to best_cents.
            pass # new_cents is discarded. best_cents remains.
            
        # Anneal temperature
        if temp > 0.001:
            temp *= 0.999
        
        # Adaptive dt?
        # If we have many overlaps, reduce dt?
        # For now, fixed dt.
        
    return best_cents, best_r

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Step 1: Create hexagonal initial arrangement
    cents, r0 = _construct_hexagonal()
    best_cents = cents.copy()
    best_r = r0
    
    # Step 2: Force optimization
    remaining = deadline - (time.time() - start_time)
    if remaining > 5:
        cents, r1 = _force_optimize(best_cents, time_limit=min(remaining - 2, 100))
        if r1 > best_r:
            best_r = r1
            best_cents = cents
            
    return best_cents
# EVOLVE-BLOCK-END
