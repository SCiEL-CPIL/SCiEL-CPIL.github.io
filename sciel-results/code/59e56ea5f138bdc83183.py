"""
Search-based optimizer for two-colourings of [1,256] minimizing monochromatic 4-APs.
Uses multi-restart simulated annealing with efficient delta computation.
"""
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 115  # Safety margin before 120s limit
    
    # Precompute all 4-term APs as index tuples (0-indexed)
    aps = []
    for d in range(1, 86):
        max_start = 256 - 3 * d
        for a in range(1, max_start + 1):
            i0 = a - 1
            i1 = i0 + d
            i2 = i0 + 2 * d
            i3 = i0 + 3 * d
            aps.append((i0, i1, i2, i3))
    
    num_aps = len(aps)
    
    # For each position, store the other 3 positions in each AP it belongs to
    # Store as flat array for faster access
    pos_to_others = [[] for _ in range(256)]
    for (i0, i1, i2, i3) in aps:
        for pos in (i0, i1, i2, i3):
            others = [x for x in (i0, i1, i2, i3) if x != pos]
            pos_to_others[pos].append(others)
    
    # Precompute lengths for faster iteration
    pos_deg = [len(lst) for lst in pos_to_others]
    
    # Count monochromatic 4-APs using vectorized approach
    def count_mono(colors):
        count = 0
        for (i0, i1, i2, i3) in aps:
            if colors[i0] == colors[i1] == colors[i2] == colors[i3]:
                count += 1
        return count
    
    # Fast delta computation: check if flipping pos changes mono status
    # Key insight: for an AP to be mono, all 4 must be same color.
    # When we flip pos, the AP's mono status changes only if the other 3 are all same.
    # If other 3 are all color c, then:
    #   - If pos was c: AP was mono, after flip it's not -> delta -= 1
    #   - If pos was 1-c: AP was not mono, after flip it becomes mono -> delta += 1
    # If other 3 are not all same, flipping pos doesn't change mono status.
    def calc_delta(colors, pos):
        delta = 0
        for others in pos_to_others[pos]:
            o0, o1, o2 = others
            # Check if all 3 others have the same color
            if colors[o0] == colors[o1] == colors[o2]:
                # The 3 others are all the same color
                if colors[pos] == colors[o0]:
                    # Currently mono, will become non-mono
                    delta -= 1
                else:
                    # Currently non-mono, will become mono
                    delta += 1
        return delta
    
    # Greedy local search: keep flipping while it helps
    def greedy_local_search(colors, current_count, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for pos in range(256):
                delta = calc_delta(colors, pos)
                if delta < 0:
                    colors[pos] = 1 - colors[pos]
                    current_count += delta
                    improved = True
        return colors, current_count
    
    # Simulated annealing for one run
    def simulate_annealing(colors, current_count, T_start, T_end, cooling, time_limit, rng):
        T = T_start
        while time.time() < time_limit and T > T_end:
            pos = rng.randint(0, 255)
            delta = calc_delta(colors, pos)
            
            if delta <= 0:
                colors[pos] = 1 - colors[pos]
                current_count += delta
            else:
                # Accept with probability exp(-delta/T)
                if rng.random() < np.exp(-delta / T):
                    colors[pos] = 1 - colors[pos]
                    current_count += delta
            
            T *= cooling
        
        return colors, current_count
    
    best_count = float('inf')
    best_colors = None
    
    # Multiple restarts with different strategies
    num_restarts = 8
    time_per_restart = (deadline - time.time()) / num_restarts
    
    for restart in range(num_restarts):
        remaining_time = deadline - time.time()
        if remaining_time < 5:
            break
        
        # Allocate time: 70% SA, 30% greedy polish
        sa_time = remaining_time * 0.7
        greedy_time = remaining_time * 0.3
        
        # Different initialization strategies per restart
        if restart == 0:
            # Random
            colors = np.random.randint(0, 2, size=256)
        elif restart == 1:
            # Alternating
            colors = np.array([(i % 2) for i in range(256)], dtype=np.int8)
        elif restart == 2:
            # Blocks of 2
            colors = np.array([(i // 2) % 2 for i in range(256)], dtype=np.int8)
        elif restart == 3:
            # Blocks of 4
            colors = np.array([(i // 4) % 2 for i in range(256)], dtype=np.int8)
        else:
            # Random with bias toward previous best
            if best_colors is not None:
                colors = best_colors.copy()
                # Randomly flip some positions
                for _ in range(50):
                    colors[rng.randint(0, 255)] = 1 - colors[rng.randint(0, 255)]
            else:
                colors = np.random.randint(0, 2, size=256)
        
        current_count = count_mono(colors)
        
        if current_count < best_count:
            best_count = current_count
            best_colors = colors.copy()
        
        # Run simulated annealing
        rng = random.Random(42 + restart)
        T_start = 30.0
        T_end = 0.5
        cooling = 0.999
        
        sa_deadline = time.time() + sa_time
        colors, current_count = simulate_annealing(
            colors, current_count, T_start, T_end, cooling, sa_deadline, rng
        )
        
        if current_count < best_count:
            best_count = current_count
            best_colors = colors.copy()
        
        # Run greedy local search
        greedy_deadline = time.time() + greedy_time
        colors, current_count = greedy_local_search(colors, current_count, greedy_deadline)
        
        if current_count < best_count:
            best_count = current_count
            best_colors = colors.copy()
    
    # Final polish with all remaining time
    if best_colors is not None and time.time() < deadline - 2:
        colors = best_colors.copy()
        current_count = best_count
        final_deadline = deadline - 1
        colors, current_count = greedy_local_search(colors, current_count, final_deadline)
        if current_count < best_count:
            best_count = current_count
            best_colors = colors
    
    return [int(c) for c in best_colors]
