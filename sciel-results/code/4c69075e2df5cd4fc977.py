import time
import random
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin

    # 1. Precompute 4-APs
    ap_list = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            # 0-based indices
            ap_list.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    total_aps = len(ap_list)
    aps = np.array(ap_list, dtype=np.int32)
    
    # 2. Precompute position to AP mapping and "others"
    # pos_ap_info[p] = list of (ap_idx, position_in_ap)
    pos_ap_info = [[] for _ in range(N)]
    for i in range(total_aps):
        for j in range(4):
            pos_ap_info[aps[i, j]].append((i, j))
    
    # Precompute "others" for fast delta calculation
    # others[ap_idx, pos_in_ap] = (idx1, idx2, idx3)
    others = np.zeros((total_aps, 4, 3), dtype=np.int32)
    for i in range(total_aps):
        for j in range(4):
            for k in range(4):
                if k != j:
                    # We need to store the 3 others.
                    # Let's just fill a temp list
                    pass
            
    # More efficient precomputation of others
    others_list = []
    for i in range(total_aps):
        row = []
        for j in range(4):
            others_in_ap = [aps[i, k] for k in range(4) if k != j]
            row.append(tuple(others_in_ap))
        others_list.append(row)
    
    # 3. Helper functions
    
    def count_mono(colors):
        c = colors[aps]
        mono = (c.max(axis=1) == c.min(axis=1))
        return int(mono.sum())
    
    def delta_flip(colors, p):
        old_c = colors[p]
        new_c = 1 - old_c
        d = 0
        for a_idx, pos_in_ap in pos_ap_info[p]:
            o1, o2, o3 = others_list[a_idx][pos_in_ap]
            c1, c2, c3 = colors[o1], colors[o2], colors[o3]
            if c1 == c2 == c3 == new_c:
                d += 1
            elif c1 == c2 == c3 == old_c:
                d -= 1
        return d
    
    def local_search(colors, max_time=1.0):
        """Hill climbing: flip bits that reduce mono count."""
        current_count = count_mono(colors)
        improved = True
        start_ls = time.time()
        while improved and time.time() < start_ls + max_time:
            improved = False
            # Randomize order to avoid cycles? Or just iterate 0..N-1
            for p in range(N):
                if time.time() > start_ls + max_time:
                    break
                delta = delta_flip(colors, p)
                if delta < 0:
                    colors[p] ^= 1
                    current_count += delta
                    improved = True
        return current_count

    def simulated_annealing(colors, max_time=5.0):
        """SA to escape local minima."""
        current_count = count_mono(colors)
        best_count = current_count
        best_colors = colors.copy()
        
        # Cooling schedule
        # We don't know the number of steps in advance, so we use time-based steps
        # Estimate steps: ~10k steps/sec? Let's say 50k steps in 5s.
        # T_start = 10, T_end = 0.1
        # alpha = (T_end / T_start) ** (1 / num_steps)
        
        start_time_sa = time.time()
        T = 5.0 # Initial temperature
        
        while time.time() < start_time_sa + max_time:
            # Pick random position
            p = random.randint(0, N - 1)
            delta = delta_flip(colors, p)
            
            # Acceptance criterion
            if delta <= 0:
                # Always accept improvements or neutral
                colors[p] ^= 1
                current_count += delta
                if current_count < best_count:
                    best_count = current_count
                    best_colors = colors.copy()
            else:
                # Accept worse solutions with probability exp(-delta / T)
                # delta is positive integer (1, 2, ...)
                # If T is small, exp(-delta/T) is small
                r = random.random()
                if r < np.exp(-delta / T):
                    colors[p] ^= 1
                    current_count += delta
            
            # Cool down
            # Simple linear cooling based on time fraction
            elapsed = time.time() - start_time_sa
            fraction = elapsed / max_time
            T = 5.0 * (1.0 - fraction) + 0.01
            if T < 0.01:
                T = 0.01

        return best_colors, best_count

    # 4. Main Search Loop
    
    best_global_count = float('inf')
    best_global_colors = None
    
    # LFSR Generator for structured initialization
    def lfsr_init(seed=1):
        # 8-bit LFSR
        state = seed & 0xFF
        colors = np.zeros(N, dtype=np.int8)
        for i in range(N):
            colors[i] = state & 1
            # Tap positions for 8-bit Fibonacci LFSR: 0x12 (bits 4 and 2)
            feedback = ((state >> 4) ^ (state >> 2)) & 1
            state = (state >> 1) | (feedback << 7)
        return colors

    # Generate initial candidates
    candidates = []
    # 1. LFSR variants
    for s in range(1, 10):
        candidates.append(lfsr_init(s))
    # 2. Random
    for _ in range(10):
        candidates.append(np.random.randint(0, 2, size=N).astype(np.int8))
    # 3. Alternating
    candidates.append(np.array([(i // 2) % 2 for i in range(N)], dtype=np.int8))
    
    # Process candidates
    for i, cand in enumerate(candidates):
        if time.time() > deadline - 10:
            break
            
        # Run SA
        sa_colors, sa_count = simulated_annealing(cand.copy(), max_time=3.0)
        
        # Run Local Search on the result of SA
        ls_count = local_search(sa_colors, max_time=1.0)
        
        if ls_count < best_global_count:
            best_global_count = ls_count
            best_global_colors = sa_colors.copy()
            
        # If time allows, try a different seed/strategy
        # We can just continue to next candidate

    # If we still have time, run a final deep local search on the best found
    if best_global_colors is not None and time.time() < deadline - 5:
        # Try to improve best_global_colors further with a longer local search
        # local_search already does full passes, but we can run it again if it improved
        # Or run SA on the best_global_colors to see if it can escape a local minimum
        # Let's do a short SA on the best
        sa_colors, sa_count = simulated_annealing(best_global_colors.copy(), max_time=5.0)
        ls_count = local_search(sa_colors, max_time=2.0)
        if ls_count < best_global_count:
            best_global_count = ls_count
            best_global_colors = sa_colors.copy()

    if best_global_colors is None:
        # Fallback: random
        best_global_colors = np.random.randint(0, 2, size=N).astype(np.int8)
        local_search(best_global_colors, max_time=1.0)

    return [int(c) for c in best_global_colors]
