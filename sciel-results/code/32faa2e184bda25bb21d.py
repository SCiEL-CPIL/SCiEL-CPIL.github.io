import random
import time
import numpy as np

def _sum_counts_vec(mask, N):
    """Number of representations a+b (a<=b) for each sum, using a numpy matrix."""
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        return np.zeros(2 * N + 1, dtype=np.int32)
    A = idx[:, None]
    B = idx[None, :]
    sums = A + B
    cnt = np.zeros(2 * N + 1, dtype=np.int32)
    np.add.at(cnt, sums, 1)
    return cnt

def _add_vec(mask, cnt, x, N):
    idx = np.nonzero(mask)[0]
    cnt[2 * x] += 1
    if idx.size:
        cnt[idx + x] += 1
    mask[x] = True

def _remove_vec(mask, cnt, x, N):
    idx = np.nonzero(mask)[0]
    cnt[2 * x] -= 1
    if idx.size:
        other = idx[idx != x]
        cnt[other + x] -= 1
    mask[x] = False

def _saturating_greedy(N, rng, skip_p=0.0):
    """
    Greedy construction.
    To improve quality, we don't just take the first feasible element.
    We can look ahead or use randomization.
    For simplicity and speed, we stick to random skip-noise but ensure we
    scan all elements.
    """
    mask = np.zeros(N + 1, dtype=bool)
    cnt = np.zeros(2 * N + 1, dtype=np.int32)
    for x in range(1, N + 1):
        if mask[x]:
            continue
        # Check if x can be added
        idx = np.nonzero(mask)[0]
        ok = True
        if cnt[2 * x] == 2:
            ok = False
        else:
            if idx.size:
                s = idx + x
                if (cnt[s] == 2).any():
                    ok = False
        if ok:
            if rng.random() < skip_p:
                continue
            idx = np.nonzero(mask)[0]
            cnt[2 * x] += 1
            if idx.size:
                cnt[idx + x] += 1
            mask[x] = True
    return mask, cnt

def _simulated_annealing(mask, cnt, N, rng, time_limit, best_mask, best_size, T_start=50.0, T_end=0.1, cooling=0.9999):
    """
    Simulated Annealing with 1-swap, 2-swap, and 3-swap moves.
    Objective: Maximize size.
    Constraint: No sum count > 2.
    Penalty: Large penalty for violations.
    """
    cur_size = int(mask.sum())
    best_size_local = best_size
    best_mask_local = best_mask.copy()
    
    # Energy function: -size + lambda * violations
    # We want to minimize Energy.
    # If valid, Energy = -size.
    # If invalid, Energy = -size + lambda * sum(max(0, cnt-2))
    
    lambda_pen = 100.0
    
    def energy(cnt, size):
        viol = int(np.maximum(0, cnt - 2).sum())
        return -size + lambda_pen * viol

    cur_energy = energy(cnt, cur_size)
    
    T = T_start
    start_time = time.time()
    
    # Pre-compute some structures? 
    # For swaps, we need to be fast.
    
    while time.time() < time_limit:
        # Move type selection
        r = rng.random()
        if r < 0.4:
            move_type = 1
        elif r < 0.8:
            move_type = 2
        else:
            move_type = 3
            
        in_idx = np.nonzero(mask)[0]
        out_idx = np.nonzero(~mask)[0]
        
        if in_idx.size < move_type or out_idx.size < move_type:
            # Not enough elements for this move type, try smaller
            if in_idx.size < 1 or out_idx.size < 1:
                break
            move_type = 1

        # Select elements to remove and add
        # To make it more effective, we can bias selection.
        # For now, random selection from available sets.
        
        # Optimization: If move_type > 1, ensure we don't pick the same element
        # We use rng.sample which ensures distinctness
        
        try:
            rem = rng.sample(list(in_idx), move_type)
            add = rng.sample(list(out_idx), move_type)
        except ValueError:
            continue
            
        # Apply move
        for x in rem:
            _remove_vec(mask, cnt, x, N)
        for y in add:
            _add_vec(mask, cnt, y, N)
            
        new_size = cur_size # Size doesn't change for swaps
        new_energy = energy(cnt, new_size)
        
        delta_E = new_energy - cur_energy
        
        # Acceptance criterion
        if delta_E < 0 or rng.random() < np.exp(-delta_E / max(T, 1e-9)):
            cur_energy = new_energy
            if cur_energy < -best_size_local: # Valid solution with larger size
                # Check if valid
                if np.any(cnt > 2):
                    pass # Should not happen if delta_E < 0 and lambda is high, but check
                else:
                    best_size_local = new_size
                    best_mask_local = mask.copy()
        else:
            # Revert
            for y in add:
                _remove_vec(mask, cnt, y, N)
            for x in rem:
                _add_vec(mask, cnt, x, N)
                
        # Cooling
        T *= cooling
        if T < T_end:
            T = T_end
            
        # Periodic restart of temperature if stuck?
        # Or just let it run.
        
    return best_mask_local, best_size_local

def construct():
    N = 1000
    start_time = time.time()
    time_limit = 145.0  # Safety margin

    rng = random.Random(12345)
    
    best_mask = None
    best_size = 0
    best_cnt = None
    
    # Phase 1: Greedy Seeds
    # Try a few greedy seeds with different skip probabilities
    seed_time = 10.0
    while time.time() < start_time + seed_time:
        skip_p = rng.random() * 0.3
        mask, cnt = _saturating_greedy(N, rng, skip_p=skip_p)
        s = int(mask.sum())
        if s > best_size:
            best_size = s
            best_mask = mask
            best_cnt = cnt

    if best_mask is None:
        best_mask = np.zeros(N + 1, dtype=bool)
        best_cnt = np.zeros(2 * N + 1, dtype=np.int32)
        best_size = 0

    # Phase 2: Simulated Annealing
    # We will run SA for the remaining time, restarting with new temperatures if we hit a plateau
    # Or just one long SA run with a slow cooling schedule.
    
    # Let's do multiple SA runs with restarts
    sa_time_left = time_limit - (time.time() - start_time) - 5.0 # Leave 5s for cleanup
    
    if sa_time_left > 10.0:
        # Run SA
        # We pass a copy to SA, and SA returns the best found
        # We might want to run SA multiple times if time permits?
        # A single long SA run with good parameters is usually better than many short ones.
        
        # Let's use a cooling schedule that adapts to time.
        # T_start = 50, T_end = 0.1
        # Number of steps? We don't know how many steps fit.
        # The loop in SA is time-based.
        
        best_mask, best_size = _simulated_annealing(
            best_mask.copy(), best_cnt.copy(), N, rng,
            sa_time_left, best_mask, best_size, 
            T_start=50.0, T_end=0.05, cooling=0.99995
        )

    # Phase 3: Final Greedy Polish
    # Try to add any remaining feasible elements
    if time.time() < time_limit - 1.0:
        cnt3 = _sum_counts_vec(best_mask, N)
        improved = True
        while improved and time.time() < time_limit - 0.5:
            improved = False
            # Find all feasible candidates
            feas = np.zeros(N + 1, dtype=bool)
            for x in range(1, N + 1):
                if best_mask[x]:
                    continue
                idx = np.nonzero(best_mask)[0]
                ok = True
                if cnt3[2 * x] == 2:
                    ok = False
                else:
                    if idx.size:
                        s = idx + x
                        if (cnt3[s] == 2).any():
                            ok = False
                if ok:
                    feas[x] = True
            
            if feas.any():
                cand = np.nonzero(feas)[0]
                # Add them in increasing order to be deterministic and stable
                for x in sorted(cand):
                    if time.time() > time_limit - 0.5:
                        break
                    _add_vec(best_mask, cnt3, int(x), N)
                    improved = True
                cnt3 = _sum_counts_vec(best_mask, N)
        best_size = int(best_mask.sum())

    result = [int(i) for i in np.nonzero(best_mask)[0]]
    return result
