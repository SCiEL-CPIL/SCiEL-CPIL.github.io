import numpy as np
import random
import time

def count_mono_4ap(colors, n=256):
    """Count monochromatic 4-term arithmetic progressions in colors[0..n-1]."""
    count = 0
    # Using numpy for speed if possible, but loop is fine for n=256
    # Actually, a pure python loop with precomputed lists is fast enough
    # But let's use the precomputed structure if available, or just standard loop for initial count
    # For the initial count in construct, we can use a helper that uses the same logic
    # Or just the standard loop since it's done once per candidate.
    
    # Standard loop is O(N^2) ~ 65k ops, very fast.
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            if colors[a] == colors[a + d] == colors[a + 2 * d] == colors[a + 3 * d]:
                count += 1
    return count

def construct():
    n = 256
    start_time = time.time()
    time_limit = 115  # seconds
    
    # 1. Precompute AP neighbors for fast delta
    # ap_neighbors[i] is a list of lists, each inner list contains the 3 other indices
    ap_neighbors = [[] for _ in range(n)]
    
    # We also need to know the APs to count them initially, but we can just use the standard count.
    # For delta, we need: for each pos i, and for each AP containing i, the other 3 positions.
    
    # Let's generate all APs first to build the neighbor lists
    all_aps = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            all_aps.append((a, a + d, a + 2 * d, a + 3 * d))
            
    for ap in all_aps:
        a, b, c, d_pos = ap
        ap_neighbors[a].append([b, c, d_pos])
        ap_neighbors[b].append([a, c, d_pos])
        ap_neighbors[c].append([a, b, d_pos])
        ap_neighbors[d_pos].append([a, b, c])

    def make_thue_morse():
        colors = np.zeros(n, dtype=np.uint8)
        for i in range(n):
            colors[i] = bin(i).count('1') % 2
        return colors

    def make_random():
        return np.random.randint(0, 2, n, dtype=np.uint8)

    # Generate candidates
    candidates = []
    
    # Thue-Morse and variants
    tm = make_thue_morse()
    candidates.append(tm)
    candidates.append(1 - tm)
    
    # Shifted Thue-Morse (cyclic shifts)
    for shift in [1, 2, 3, 5, 7, 10]:
        candidates.append(np.roll(tm, shift))
        
    # Randoms
    for _ in range(20):
        candidates.append(make_random())
        
    # Block patterns
    for bs in [2, 4, 8, 16, 32]:
        blocks = np.repeat(np.array([0, 1], dtype=np.uint8), bs)[:n]
        candidates.append(blocks)
        candidates.append(1 - blocks)

    # Find best initial
    best_colors = None
    best_count = float('inf')
    
    for cand in candidates:
        c = count_mono_4ap(cand, n)
        if c < best_count:
            best_count = c
            best_colors = cand.copy()
            
    global_best_colors = best_colors.copy()
    global_best_count = best_count
    
    # Multi-start SA
    num_starts = 15
    time_per_start = (time_limit - 5) / num_starts
    
    for start_idx in range(num_starts):
        if time.time() - start_time > time_limit - 2:
            break
            
        # Initialize current state
        if start_idx == 0:
            colors = global_best_colors.copy()
        else:
            # Perturb global best
            colors = global_best_colors.copy()
            num_flips = random.randint(5, 20)
            for _ in range(num_flips):
                pos = random.randint(0, n - 1)
                colors[pos] = 1 - colors[pos]
                
        current_count = count_mono_4ap(colors, n)
        
        # SA Parameters
        T = 10.0
        T_min = 0.01
        max_iters = 200000
        cooling = (T_min / T) ** (1.0 / max_iters)
        
        for i in range(max_iters):
            if time.time() - start_time > time_limit - 1:
                break
                
            pos = random.randint(0, n - 1)
            old_val = colors[pos]
            
            # Calculate delta
            delta = 0
            for others in ap_neighbors[pos]:
                # Count matches with old_val
                # old_val is 0 or 1
                # We need to check colors[others[0]], colors[others[1]], colors[others[2]]
                
                # Optimization: 
                # If old_val == 0:
                #   mono_before = (colors[o0]==0 and colors[o1]==0 and colors[o2]==0)
                #   mono_after  = (colors[o0]==1 and colors[o1]==1 and colors[o2]==1)
                # If old_val == 1:
                #   mono_before = (colors[o0]==1 and colors[o1]==1 and colors[o2]==1)
                #   mono_after  = (colors[o0]==0 and colors[o1]==0 and colors[o2]==0)
                
                c0 = colors[others[0]]
                c1 = colors[others[1]]
                c2 = colors[others[2]]
                
                if old_val == 0:
                    if c0 == 0 and c1 == 0 and c2 == 0:
                        delta -= 1 # was mono, becomes not mono (unless all are 1, which is impossible if they were 0)
                    if c0 == 1 and c1 == 1 and c2 == 1:
                        delta += 1 # was not mono, becomes mono
                else: # old_val == 1
                    if c0 == 1 and c1 == 1 and c2 == 1:
                        delta -= 1
                    if c0 == 0 and c1 == 0 and c2 == 0:
                        delta += 1
            
            if delta <= 0:
                colors[pos] = 1 - old_val
                current_count += delta
                if current_count < global_best_count:
                    global_best_count = current_count
                    global_best_colors = colors.copy()
            else:
                if random.random() < np.exp(-delta / T):
                    colors[pos] = 1 - old_val
                    current_count += delta
            
            T *= cooling
            if T < T_min:
                T = T_min

        # Polishing step: Greedy Hill Climb
        # Try to improve further by only accepting improvements
        improved = True
        polish_iters = 0
        while improved and polish_iters < 5000:
            improved = False
            polish_iters += 1
            pos = random.randint(0, n - 1)
            old_val = colors[pos]
            
            delta = 0
            for others in ap_neighbors[pos]:
                c0 = colors[others[0]]
                c1 = colors[others[1]]
                c2 = colors[others[2]]
                
                if old_val == 0:
                    if c0 == 0 and c1 == 0 and c2 == 0:
                        delta -= 1
                    if c0 == 1 and c1 == 1 and c2 == 1:
                        delta += 1
                else:
                    if c0 == 1 and c1 == 1 and c2 == 1:
                        delta -= 1
                    if c0 == 0 and c1 == 0 and c2 == 0:
                        delta += 1
                        
            if delta < 0:
                colors[pos] = 1 - old_val
                current_count += delta
                improved = True
                if current_count < global_best_count:
                    global_best_count = current_count
                    global_best_colors = colors.copy()
                    
    return global_best_colors.tolist()
