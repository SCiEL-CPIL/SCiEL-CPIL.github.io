import time
import numpy as np
import random
import math

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s/150s kill
    n = 256

    # Precompute all 4-term APs as index tuples (0-indexed)
    # a, a+d, a+2d, a+3d with 1<=a, d>=1, a+3d<=256
    aps = []
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            aps.append((i0, i1, i2, i3))

    aps_arr = np.array(aps, dtype=np.int32)  # shape (M, 4)
    M = len(aps_arr)

    # Build per-position list of AP indices for delta computation
    pos_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for p in ap:
            pos_aps[p].append(idx)
    pos_aps_arr = [np.array(lst, dtype=np.int32) for lst in pos_aps]

    # Precompute for each position, which column (0-3) it occupies in each affected AP
    pos_col = []
    for pos in range(n):
        cols = []
        ap_indices = pos_aps_arr[pos]
        if len(ap_indices) > 0:
            affected = aps_arr[ap_indices]  # shape (K, 4)
            # Find column where each row equals pos
            col_arr = np.argmax(affected == pos, axis=1)
            cols.append(col_arr)
        pos_col.append(cols[0] if cols else np.array([], dtype=np.int32))

    def count_mono(col):
        a0 = col[aps_arr[:, 0]]
        a1 = col[aps_arr[:, 1]]
        a2 = col[aps_arr[:, 2]]
        a3 = col[aps_arr[:, 3]]
        mono = (a0 == a1) & (a1 == a2) & (a2 == a3)
        return int(np.sum(mono))

    def get_delta(pos, col):
        """Vectorized delta computation for flipping position pos."""
        ap_indices = pos_aps_arr[pos]
        K = len(ap_indices)
        if K == 0:
            return 0
        affected = aps_arr[ap_indices]  # shape (K, 4)
        colors = col[affected]  # shape (K, 4)
        col_idx = pos_col[pos]  # shape (K,)
        # Current mono status
        cur_mono = (colors[:, 0] == colors[:, 1]) & (colors[:, 1] == colors[:, 2]) & (colors[:, 2] == colors[:, 3])
        cur_mono_count = int(np.sum(cur_mono))
        # After flip: flip the specific column in each row
        flipped = colors.copy()
        rows = np.arange(K)
        flipped[rows, col_idx] = 1 - flipped[rows, col_idx]
        new_mono = (flipped[:, 0] == flipped[:, 1]) & (flipped[:, 1] == flipped[:, 2]) & (flipped[:, 2] == flipped[:, 3])
        new_mono_count = int(np.sum(new_mono))
        return new_mono_count - cur_mono_count

    def greedy_polish(col, max_iters=200):
        """Run greedy local search until no improvement or max_iters reached."""
        col = col.copy()
        cur_count = count_mono(col)
        improved = True
        iters = 0
        while improved and iters < max_iters and (time.time() - start_time) < deadline - 5.0:
            improved = False
            iters += 1
            # Compute deltas for all positions in vectorized fashion
            # For each position, we need to check all APs it's part of
            # We'll do it in chunks for memory efficiency
            order = np.random.permutation(n)
            for pos in order:
                if (time.time() - start_time) > deadline - 5.0:
                    break
                d = get_delta(pos, col)
                if d < 0:
                    col[pos] = 1 - col[pos]
                    cur_count += d
                    improved = True
            # After a full pass, do a second pass without shuffling to catch remaining
            if improved:
                for pos in range(n):
                    if (time.time() - start_time) > deadline - 5.0:
                        break
                    d = get_delta(pos, col)
                    if d < 0:
                        col[pos] = 1 - col[pos]
                        cur_count += d
                        improved = True
        return col, cur_count

    def sa_polish(col, max_iters=500):
        """Simulated annealing polish: accept worse moves with probability exp(-delta/T)."""
        col = col.copy()
        cur_count = count_mono(col)
        best_col = col.copy()
        best_count = cur_count
        T = 8.0
        T_min = 0.005
        cooling = 0.998
        iters = 0
        no_improve_streak = 0
        while iters < max_iters and (time.time() - start_time) < deadline - 5.0:
            iters += 1
            T *= cooling
            if T < T_min:
                T = T_min
            order = np.random.permutation(n)
            any_change = False
            for pos in order:
                if (time.time() - start_time) > deadline - 5.0:
                    break
                d = get_delta(pos, col)
                if d < 0:
                    col[pos] = 1 - col[pos]
                    cur_count += d
                    any_change = True
                    if cur_count < best_count:
                        best_count = cur_count
                        best_col = col.copy()
                        no_improve_streak = 0
                    else:
                        no_improve_streak += 1
                elif d == 0:
                    # Accept neutral moves to explore
                    if random.random() < 0.3:
                        col[pos] = 1 - col[pos]
                        any_change = True
                elif d > 0 and random.random() < math.exp(-d / T):
                    col[pos] = 1 - col[pos]
                    cur_count += d
                    any_change = True
                    no_improve_streak += 1
            if not any_change:
                break
            # If stuck for many iterations without improvement, do a greedy sweep
            if no_improve_streak > 100:
                for pos in range(n):
                    if (time.time() - start_time) > deadline - 5.0:
                        break
                    d = get_delta(pos, col)
                    if d < 0:
                        col[pos] = 1 - col[pos]
                        cur_count += d
                        if cur_count < best_count:
                            best_count = cur_count
                            best_col = col.copy()
                no_improve_streak = 0
        return best_col, best_count

    # Phase 1: Generate diverse initial solutions and polish them
    best_c = None
    best_count = float('inf')

    # 1. Thue-Morse
    c_tm = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
    c_polished, count = greedy_polish(c_tm)
    if count < best_count:
        best_count = count
        best_c = c_polished.copy()

    # 2. Random
    for _ in range(8):
        c_rand = np.random.randint(0, 2, n).astype(np.int8)
        c_polished, count = greedy_polish(c_rand)
        if count < best_count:
            best_count = count
            best_c = c_polished.copy()

    # 3. Block patterns
    for block_size in [2, 3, 4, 5, 8, 16, 32, 64]:
        c_block = np.zeros(n, dtype=np.int8)
        for i in range(n):
            c_block[i] = (i // block_size) % 2
        c_polished, count = greedy_polish(c_block)
        if count < best_count:
            best_count = count
            best_c = c_polished.copy()

    # 4. Rademacher-like: based on parity of floor(log2(i))
    c_rad = np.zeros(n, dtype=np.int8)
    for i in range(n):
        c_rad[i] = int(math.log2(i + 1)) % 2
    c_polished, count = greedy_polish(c_rad)
    if count < best_count:
        best_count = count
        best_c = c_polished.copy()

    # 5. Walsh-Hadamard row (first non-trivial)
    c_walsh = np.array([1 if (i & 1) else 0 for i in range(n)], dtype=np.int8)
    c_polished, count = greedy_polish(c_walsh)
    if count < best_count:
        best_count = count
        best_c = c_polished.copy()

    # 6. Quadratic pattern: color based on (i*i) % 2
    c_quad = np.array([(i * i) % 2 for i in range(n)], dtype=np.int8)
    c_polished, count = greedy_polish(c_quad)
    if count < best_count:
        best_count = count
        best_c = c_polished.copy()

    # 7. Sinusoidal pattern
    c_sin = np.array([1 if math.sin(i * 0.1) > 0 else 0 for i in range(n)], dtype=np.int8)
    c_polished, count = greedy_polish(c_sin)
    if count < best_count:
        best_count = count
        best_c = c_polished.copy()

    # 8. Multi-scale Thue-Morse
    for scale in [2, 4, 8]:
        c_mt = np.array([bin(i // scale).count('1') % 2 for i in range(n)], dtype=np.int8)
        c_polished, count = greedy_polish(c_mt)
        if count < best_count:
            best_count = count
            best_c = c_polished.copy()

    # Phase 2: Iterated Local Search (ILS) with SA
    # We will repeatedly perturb the best solution and polish it.
    
    base_perturb = 2
    max_perturb = 12
    stuck_count = 0
    last_improvement = best_count
    sa_max_iters = 300
    
    while (time.time() - start_time) < deadline - 15.0:
        # Perturb the best solution
        c_current = best_c.copy()
        
        # Adaptive perturbation: more flips when stuck
        time_elapsed = time.time() - start_time
        time_remaining = deadline - 15.0 - time_elapsed
        
        if best_count == last_improvement:
            stuck_count += 1
        else:
            stuck_count = 0
            last_improvement = best_count
        
        if stuck_count > 15 or time_remaining < 15.0:
            num_flips = random.randint(10, 18)
            sa_max_iters = 200
        elif stuck_count > 8:
            num_flips = random.randint(6, 12)
            sa_max_iters = 250
        elif stuck_count > 3:
            num_flips = random.randint(4, 8)
            sa_max_iters = 300
        else:
            num_flips = random.randint(base_perturb, max_perturb)
            sa_max_iters = 350
        
        # Flip random positions
        positions_to_flip = np.random.choice(n, num_flips, replace=False)
        for p in positions_to_flip:
            c_current[p] = 1 - c_current[p]
        
        # Polish the perturbed solution - use SA for better escape
        c_polished, count = sa_polish(c_current, max_iters=sa_max_iters)
        
        if count < best_count:
            best_count = count
            best_c = c_polished.copy()
            stuck_count = 0
            last_improvement = best_count
            # Do a final greedy polish on the improved solution
            c_polished2, count2 = greedy_polish(c_polished, max_iters=400)
            if count2 < best_count:
                best_count = count2
                best_c = c_polished2.copy()
        
        # Also try a pure greedy polish from the perturbed state sometimes
        if random.random() < 0.3 and (time.time() - start_time) < deadline - 15.0:
            c_greedy, count_g = greedy_polish(c_current, max_iters=300)
            if count_g < best_count:
                best_count = count_g
                best_c = c_greedy.copy()

    # Phase 3: Final Polish with both methods - deep polish
    if (time.time() - start_time) < deadline - 4.0:
        final_c, final_count = greedy_polish(best_c, max_iters=500)
        if final_count < best_count:
            best_c = final_c
            best_count = final_count

    if (time.time() - start_time) < deadline - 3.0:
        final_c2, final_count2 = sa_polish(best_c, max_iters=400)
        if final_count2 < best_count:
            best_c = final_c2
            best_count = final_count2

    # One more greedy pass after SA
    if (time.time() - start_time) < deadline - 2.0:
        final_c3, final_count3 = greedy_polish(best_c, max_iters=300)
        if final_count3 < best_count:
            best_c = final_c3
            best_count = final_count3

    # Final SA with very low temperature for fine-tuning
    if (time.time() - start_time) < deadline - 1.0:
        final_c4, final_count4 = sa_polish(best_c, max_iters=200)
        if final_count4 < best_count:
            best_c = final_c4
            best_count = final_count4

    # Verify and return
    actual_count = count_mono(best_c)
    if actual_count < best_count:
        best_count = actual_count

    return best_c.tolist()
