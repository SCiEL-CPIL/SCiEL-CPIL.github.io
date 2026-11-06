import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill
    
    N = 128
    
    # Pre-compute all 4-term APs
    aps = []
    ap_pos = []  # For each AP, store the 4 positions
    for d in range(1, N // 3 + 1):
        for i in range(N - 3 * d):
            ap = (i, i + d, i + 2 * d, i + 3 * d)
            aps.append(ap)
            ap_pos.append(ap)
    
    num_aps = len(aps)
    
    # Precompute for each position, which APs it's in
    pos_to_aps = [[] for _ in range(N)]
    for ap_idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(ap_idx)
    
    def count_violations(colors):
        count = 0
        for ap in aps:
            c1, c2, c3, c4 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if len({c1, c2, c3, c4}) < 3:
                count += 1
        return count
    
    def is_valid(colors):
        for ap in aps:
            c1, c2, c3, c4 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if len({c1, c2, c3, c4}) < 3:
                return False
        return True
    
    # Try to find a valid coloring with k colors using simulated annealing / local search
    def try_k_colors(k, time_budget):
        end_time = time.time() + time_budget
        
        for restart in range(500):
            if time.time() > end_time:
                break
            
            # Initialize with random colors from 0..k-1
            colors = [random.randint(0, k - 1) for _ in range(N)]
            violations = count_violations(colors)
            
            if violations == 0:
                return colors[:]
            
            # Local search with simulated annealing
            temperature = 10.0
            cooling = 0.995
            best_colors = colors[:]
            best_violations = violations
            
            for step in range(20000):
                if time.time() > end_time:
                    break
                
                # Pick a random position to change
                pos = random.randint(0, N - 1)
                old_c = colors[pos]
                new_c = random.randint(0, k - 1)
                
                if new_c == old_c:
                    continue
                
                colors[pos] = new_c
                new_violations = count_violations(colors)
                
                delta = new_violations - violations
                
                if delta <= 0 or random.random() < np.exp(-delta / max(temperature, 0.01)):
                    violations = new_violations
                    if new_violations < best_violations:
                        best_violations = new_violations
                        best_colors = colors[:]
                        if best_violations == 0:
                            return best_colors
                else:
                    colors[pos] = old_c
                
                temperature *= cooling
            
            if best_violations == 0:
                return best_colors
    
    # Start with a greedy construction to get a reasonable starting point
    def greedy_construct(k):
        colors = [-1] * N
        for x in range(N):
            # Try each color from 0..k-1
            best_color = None
            best_violations = k * 100  # Large number
            
            for c in range(k):
                colors[x] = c
                # Count violations involving position x
                v = 0
                for ap_idx in pos_to_aps[x]:
                    ap = aps[ap_idx]
                    c1, c2, c3, c4 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                    if any(c is -1 for c in (c1, c2, c3, c4)):
                        # Can't evaluate yet, skip
                        continue
                    if len({c1, c2, c3, c4}) < 3:
                        v += 1
                if v < best_violations:
                    best_violations = v
                    best_color = c
            
            colors[x] = best_color
        return colors
    
    # First, try to find a valid coloring with 3 colors
    # Then try 4, 5, etc.
    
    best_colors = None
    best_k = 128
    
    # Try k=3 first with a generous time budget
    time_left = DEADLINE - time.time()
    if time_left > 10:
        budget = min(20, time_left - 5)
        result = try_k_colors(3, budget)
        if result is not None:
            best_colors = result
            best_k = 3
    
    # If 3 colors didn't work, try 4
    if best_k > 4:
        time_left = DEADLINE - time.time()
        if time_left > 10:
            budget = min(20, time_left - 5)
            result = try_k_colors(4, budget)
            if result is not None:
                best_colors = result
                best_k = 4
    
    # If 4 colors didn't work, try 5
    if best_k > 5:
        time_left = DEADLINE - time.time()
        if time_left > 10:
            budget = min(15, time_left - 5)
            result = try_k_colors(5, budget)
            if result is not None:
                best_colors = result
                best_k = 5
    
    # If still no valid coloring, use greedy with more colors
    if best_colors is None:
        # Use greedy with 6 colors
        for k in [6, 7, 8, 10, 12]:
            colors = greedy_construct(k)
            if is_valid(colors):
                best_colors = colors
                best_k = k
                break
        
        # If still no valid coloring, use a larger k
        if best_colors is None:
            for k in [15, 20, 30, 50, 100]:
                colors = greedy_construct(k)
                if is_valid(colors):
                    best_colors = colors
                    best_k = k
                    break
    
    # Final fallback: use the seed-like approach
    if best_colors is None:
        colors = []
        for x in range(N):
            found = False
            for color in range(128):
                valid = True
                for d in range(1, (x) // 3 + 1):
                    if x - 3 * d < 0:
                        continue
                    prev_colors = {colors[x - j * d] for j in (1, 2, 3)}
                    if len({color} | prev_colors) < 3:
                        valid = False
                        break
                if valid:
                    colors.append(color)
                    found = True
                    break
            if not found:
                colors.append(random.randint(0, 127))
        best_colors = colors
    
    # Final validation
    if not is_valid(best_colors):
        # Try one more time with a simple approach
        for k in range(6, 129):
            colors = [random.randint(0, k - 1) for _ in range(N)]
            if is_valid(colors):
                best_colors = colors
                break
    
    return best_colors
