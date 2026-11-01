"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def construct():
    """
    Multi-start simulated annealing to maximize the packing radius.
    """
    rng = np.random.default_rng(42)
    start_time = time.time()
    max_time = 115.0

    def full_score(centers):
        """Compute packing radius: min(min_pair_dist/2, min_wall_dist)."""
        wall = 0.5 - np.abs(centers)
        min_wall = np.min(wall)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        d2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(d2, np.inf)
        min_pair = np.sqrt(np.min(d2))
        return min(min_pair / 2.0, min_wall)

    def sa_run(init_centers, rng, time_budget):
        """Run SA with incremental distance tracking."""
        t0 = time.time()
        centers = init_centers.copy()
        
        # Maintain distance matrix and wall distances
        wall_dists = 0.5 - np.abs(centers)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        d2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(d2, np.inf)
        
        cur_score = min(np.sqrt(np.min(d2)) / 2.0, np.min(wall_dists))
        best_score = cur_score
        best_centers = centers.copy()
        
        # SA parameters
        n_iters = 1000000
        T_start = 0.02
        T_end = 1e-10
        alpha = (T_end / T_start) ** (1.0 / n_iters)
        
        step = 0.025
        base_step = 0.025
        no_improve = 0
        max_no_improve = 3000
        
        for it in range(n_iters):
            if it % 200 == 0 and time.time() - t0 > time_budget:
                break
            
            T = T_start * (alpha ** it)
            i = rng.integers(N)
            old_pos = centers[i].copy()
            
            # Perturb
            new_pos = old_pos + rng.normal(0, step, 2)
            new_pos = np.clip(new_pos, -0.4999, 0.4999)
            
            # Compute new distances from i to all others
            new_d2_i = np.sum((centers - new_pos) ** 2, axis=1)
            new_d2_i[i] = np.inf
            new_wall_i = np.min(0.5 - np.abs(new_pos))
            
            # Determine if move can improve
            # New min pair = min(min d2 excluding row/col i, new_d2_i)
            # New min wall = min(min wall excluding i, new_wall_i)
            
            # Get min d2 not involving i
            mask = np.ones((N, N), dtype=bool)
            mask[i, :] = False
            mask[:, i] = False
            min_d2_excl_i = np.min(d2[mask]) if np.any(mask) else np.inf
            
            min_wall_excl_i = np.min(np.delete(wall_dists, i))
            
            new_min_d2 = min(min_d2_excl_i, np.min(new_d2_i))
            new_min_wall = min(min_wall_excl_i, new_wall_i)
            new_score = min(np.sqrt(new_min_d2) / 2.0, new_min_wall)
            
            delta = new_score - cur_score
            
            # Acceptance criterion
            accept = False
            if delta >= 0:
                accept = True
            else:
                # Metropolis criterion with cap on negative delta
                if delta > -0.05 and rng.random() < np.exp(delta / T):
                    accept = True
            
            if accept:
                # Apply move
                centers[i] = new_pos
                wall_dists[i] = new_wall_i
                diff_i = centers - new_pos
                d2[i, :] = np.sum(diff_i ** 2, axis=1)
                d2[:, i] = d2[i, :]
                d2[i, i] = np.inf
                cur_score = new_score
                no_improve = 0
                if cur_score > best_score:
                    best_score = cur_score
                    best_centers = centers.copy()
            else:
                no_improve += 1
            
            # Adaptive step
            if no_improve > max_no_improve:
                step *= 0.5
                no_improve = 0
            elif it % 500 == 0 and it > 0:
                step *= 0.999
            
            # Periodic restart of step
            if it % 50000 == 0 and it > 0:
                step = base_step * (0.5 + 0.5 * rng.random())
                no_improve = 0
        
        # Final local refinement
        remaining = time_budget - (time.time() - t0)
        if remaining > 2.0:
            centers = best_centers.copy()
            ref_step = 0.008
            cur_s = full_score(centers)
            ref_t0 = time.time()
            while time.time() - ref_t0 < remaining - 0.5:
                improved = False
                for _ in range(300):
                    i = rng.integers(N)
                    old_p = centers[i].copy()
                    new_p = old_p + rng.normal(0, ref_step, 2)
                    new_p = np.clip(new_p, -0.4999, 0.4999)
                    centers[i] = new_p
                    ns = full_score(centers)
                    if ns > cur_s:
                        cur_s = ns
                        improved = True
                        if cur_s > best_score:
                            best_score = cur_s
                            best_centers = centers.copy()
                    else:
                        centers[i] = old_p
                if not improved:
                    ref_step *= 0.5
                    if ref_step < 1e-7:
                        break
        
        return best_centers, best_score

    # Generate diverse initializations
    inits = []
    
    # Grid patterns
    for rows, cols in [(7, 8), (8, 7), (5, 10), (10, 5), (6, 9), (9, 6)]:
        cx = np.linspace(-0.48, 0.48, cols)
        cy = np.linspace(-0.48, 0.48, rows)
        c = np.zeros((N, 2))
        idx = 0
        for r in range(rows):
            for cc in range(cols):
                if idx < N:
                    c[idx] = [cx[cc], cy[r]]
                    idx += 1
                else:
                    break
            if idx >= N:
                break
        c += rng.normal(0, 0.002, (N, 2))
        c = np.clip(c, -0.4999, 0.4999)
        inits.append(c)
    
    # Hexagonal
    hex_sp = 0.19
    hex_h = hex_sp * np.sqrt(3) / 2
    hc = np.zeros((N, 2))
    idx = 0
    row = 0
    while idx < N and row < 12:
        off = 0 if row % 2 == 0 else hex_sp / 2
        col = 0
        while idx < N and col < 10:
            x = -0.48 + off + col * hex_sp
            y = -0.48 + row * hex_h
            if -0.4999 <= x <= 0.4999 and -0.4999 <= y <= 0.4999:
                hc[idx] = [x, y]
                idx += 1
            col += 1
        row += 1
    if idx == N:
        hc += rng.normal(0, 0.002, (N, 2))
        hc = np.clip(hc, -0.4999, 0.4999)
        inits.append(hc)
    
    # Random
    inits.append(rng.uniform(-0.45, 0.45, (N, 2)))
    
    # Run SA from each initialization
    best_centers = None
    best_score = -np.inf
    
    for idx, init in enumerate(inits):
        rem = max_time - (time.time() - start_time)
        if rem < 5.0:
            break
        budget = min(rem, max(12.0, rem / (len(inits) - idx)))
        c_local, s_local = sa_run(init, rng, budget)
        if s_local > best_score:
            best_score = s_local
            best_centers = c_local
    
    if best_centers is None:
        cx = np.linspace(-0.48, 0.48, 8)
        cy = np.linspace(-0.48, 0.48, 7)
        best_centers = np.zeros((N, 2))
        idx = 0
        for r in range(7):
            for cc in range(8):
                if idx < N:
                    best_centers[idx] = [cx[cc], cy[r]]
                    idx += 1
                else:
                    break
            if idx >= N:
                break
    
    return best_centers
# EVOLVE-BLOCK-END
