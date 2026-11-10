import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    N = 256
    start_time = time.time()
    # Use 118 seconds to be safe within 120s limit
    deadline = start_time + 118.0 

    # 1. Precompute all 4-term arithmetic progressions
    aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            i1 = a - 1
            i2 = a + d - 1
            i3 = a + 2 * d - 1
            i4 = a + 3 * d - 1
            aps.append((i1, i2, i3, i4))
    
    aps_arr = np.array(aps, dtype=np.int32)
    
    # 2. Precompute neighbors for fast delta calculation
    # neighbors[i] is an array of shape (count, 3) containing the other 3 indices
    neighbors = [np.empty((0, 3), dtype=np.int32) for _ in range(N)]
    counts = [0] * N
    for i1, i2, i3, i4 in aps:
        counts[i1] += 1
        counts[i2] += 1
        counts[i3] += 1
        counts[i4] += 1
        
    # Allocate exact sizes
    for i in range(N):
        if counts[i] > 0:
            neighbors[i] = np.empty((counts[i], 3), dtype=np.int32)
            
    # Fill neighbors
    indices = [0] * N
    for i1, i2, i3, i4 in aps:
        neighbors[i1][indices[i1]] = (i2, i3, i4)
        indices[i1] += 1
        neighbors[i2][indices[i2]] = (i1, i3, i4)
        indices[i2] += 1
        neighbors[i3][indices[i3]] = (i1, i2, i4)
        indices[i3] += 1
        neighbors[i4][indices[i4]] = (i1, i2, i3)
        indices[i4] += 1

    def count_mono(colors):
        """Count monochromatic APs using numpy vectorization"""
        c0 = colors[aps_arr[:, 0]]
        c1 = colors[aps_arr[:, 1]]
        c2 = colors[aps_arr[:, 2]]
        c3 = colors[aps_arr[:, 3]]
        return int(np.sum((c0 == c1) & (c0 == c2) & (c0 == c3)))

    # 3. Generate diverse initializations
    def generate_initializations():
        inits = []
        
        # Thue-Morse sequence
        tm = np.array([bin(i).count('1') % 2 for i in range(N)], dtype=np.int32)
        inits.append(tm)
        inits.append(1 - tm) # Inverted Thue-Morse
        
        # Shifted Thue-Morse
        for shift in [1, 2, 3]:
            shifted = np.roll(tm, shift)
            inits.append(shifted)
            
        # LFSR based patterns (pseudo-random with structure)
        # Simple LFSR generator
        def lfsr_pattern(seed, poly_mask, length):
            state = seed
            out = np.zeros(length, dtype=np.int32)
            for i in range(length):
                # Simple tap for 8-bit LFSR, extended to 256
                # Using a simple feedback polynomial
                bit = (state >> 0) ^ (state >> 3) ^ (state >> 5) ^ (state >> 7)
                state = (state >> 1) | (bit << 7)
                out[i] = state & 1
            return out
        
        inits.append(lfsr_pattern(1, 0, N))
        inits.append(lfsr_pattern(2, 0, N))
        inits.append(lfsr_pattern(3, 0, N))
        
        # Greedy Initialization
        def greedy_init():
            colors = np.zeros(N, dtype=np.int32)
            # For each position, choose color that minimizes new mono APs
            # This is O(N * APs_per_pos) which is fine for N=256
            for i in range(N):
                best_c = 0
                best_delta = float('inf')
                for c in [0, 1]:
                    # Temporarily set color
                    colors[i] = c
                    # Calculate cost increase
                    # Only need to check APs containing i
                    # Since we are building sequentially, APs with index > i are not fully determined yet?
                    # Actually, an AP (a, a+d, a+2d, a+3d) is valid if all indices are < N.
                    # If we fill 0..N-1, when we fill i, we only affect APs where i is the max index?
                    # No, i can be any position in the AP.
                    # However, if we fill sequentially, when we set color[i], the other 3 indices might be < i or > i.
                    # If > i, their color is currently 0 (default), which is a guess.
                    # This simple greedy is flawed if it doesn't look ahead.
                    # Let's stick to random and structured patterns for init, 
                    # and rely on SA to fix them.
                    pass
                # Fallback: random
                colors[i] = random.randint(0, 1)
            return colors
        inits.append(greedy_init())
        
        # Random
        inits.append(np.random.randint(0, 2, size=N))
        inits.append(np.random.randint(0, 2, size=N))
        
        return inits

    init_list = generate_initializations()
    
    # Evaluate and sort initializations
    scored_inits = []
    for init in init_list:
        score = count_mono(init)
        scored_inits.append((score, init))
        
    scored_inits.sort(key=lambda x: x[0])
    
    # Take top 5 for multi-start SA
    top_inits = [x[1] for x in scored_inits[:5]]
    
    best_score = scored_inits[0][0]
    best_coloring = top_inits[0].copy()
    
    # 4. Multi-Start Simulated Annealing
    # Allocate time equally to each start, with a small buffer
    time_per_start = (deadline - time.time()) / len(top_inits)
    
    for k, initial_colors in enumerate(top_inits):
        if time.time() > deadline - 5.0:
            break
            
        colors = initial_colors.copy()
        current_score = count_mono(colors)
        
        if current_score < best_score:
            best_score = current_score
            best_coloring = colors.copy()
            
        # SA Parameters
        T = 5.0
        # Estimate number of steps we can do
        # Each step takes ~10-20 microseconds with numpy? Let's assume 20us.
        # time_per_start is in seconds.
        # max_steps = time_per_start / 20e-6
        # We don't know exact speed, so we use a time-based loop with a dynamic cooling rate.
        
        # Let's run for a fixed fraction of the allotted time
        end_time_for_this_start = time.time() + time_per_start * 0.95
        
        # Adaptive cooling: we want T to drop to 0.1 by end_time_for_this_start
        # T(t) = T0 * r^t
        # We don't know t (number of steps), so we just use a small cooling rate per step.
        # If steps are fast, we need smaller cooling rate.
        # Let's try a fixed cooling rate that works well for ~100k-500k steps.
        cooling_rate = 0.99995
        
        while time.time() < end_time_for_this_start:
            # Pick random index
            i = random.randint(0, N - 1)
            old_c = colors[i]
            new_c = 1 - old_c
            
            # Calculate delta
            nbrs = neighbors[i]
            if len(nbrs) == 0:
                delta = 0
            else:
                # nbrs is (M, 3)
                cols = colors[nbrs]
                c0, c1, c2 = cols[:, 0], cols[:, 1], cols[:, 2]
                all_eq = (c0 == c1) & (c1 == c2)
                was_mono = all_eq & (c0 == old_c)
                becomes_mono = all_eq & (c0 == new_c)
                delta = int(np.sum(becomes_mono) - np.sum(was_mono))
                
            # Acceptance
            if delta < 0:
                colors[i] = new_c
                current_score += delta
            else:
                # Metropolis criterion
                if T > 0.01:
                    # Use log for numerical stability if delta/T is large
                    # ln(P) = -delta/T
                    # If -delta/T > ln(random), accept
                    # random.random() is in (0,1), ln is negative
                    # So if random() > exp(-delta/T), accept
                    # Or: log(random()) > -delta/T
                    # log(random()) is slow. Let's use exp.
                    if random.random() < np.exp(-delta / T):
                        colors[i] = new_c
                        current_score += delta
            
            # Update best
            if current_score < best_score:
                best_score = current_score
                best_coloring = colors.copy()
                
            # Cool
            T *= cooling_rate
            if T < 0.01:
                # Reheat or stop? 
                # If T drops too low, we are stuck. 
                # For multi-start, we just stop this start.
                break
                
    return best_coloring.tolist()
# EVOLVE-BLOCK-END
