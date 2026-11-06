import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s limit
    
    N = 128
    
    # Precompute all 4-term progressions (a, a+d, a+2d, a+3d) with 1-based indexing for logic
    # We will use 0-based indexing for the array `colors`
    aps = []
    for a in range(1, N + 1):
        for d in range(1, N // 3 + 1):
            end = a + 3 * d
            if end <= N:
                aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Map position (1-based) to list of AP indices containing it
    # We will use 0-based indices in the `colors` array, so pos-1
    ap_indices = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            ap_indices[pos - 1].append(idx)
            
    # Helper to count distinct colors in an AP given a color assignment
    def count_distinct(ap_idx, colors):
        a, b, c, d = aps[ap_idx]
        return len({colors[a-1], colors[b-1], colors[c-1], colors[d-1]})

    def total_violations(colors):
        viol = 0
        for idx in range(len(aps)):
            if count_distinct(idx, colors) < 3:
                viol += 1
        return viol

    # Try to find a valid coloring with K colors using Simulated Annealing
    for K in range(3, 15):
        # Initialize
        colors = [random.randint(0, K-1) for _ in range(N)]
        
        # Calculate initial violations
        current_viol = total_violations(colors)
        
        if current_viol == 0:
            # Remap and return
            unique_colors = sorted(set(colors))
            color_map = {c: i for i, c in enumerate(unique_colors)}
            return [color_map[c] for c in colors]
            
        best_colors = colors[:]
        best_viol = current_viol
        
        T = 10.0
        T_min = 0.01
        cooling = 0.99
        
        # Run SA
        while T > T_min:
            if time.time() > deadline:
                break
                
            # Pick random position
            i = random.randint(0, N-1)
            old_color = colors[i]
            
            # Pick new color
            new_color = random.randint(0, K-1)
            if new_color == old_color:
                continue
                
            # Calculate delta violations
            # Only APs containing i are affected
            delta_viol = 0
            for ap_idx in ap_indices[i]:
                # Get positions
                a, b, c, d = aps[ap_idx]
                # Get indices in colors array
                idx_a, idx_b, idx_c, idx_d = a-1, b-1, c-1, d-1
                
                # Old colors
                old_c_a, old_c_b, old_c_c, old_c_d = colors[idx_a], colors[idx_b], colors[idx_c], colors[idx_d]
                old_set = {old_c_a, old_c_b, old_c_c, old_c_d}
                old_valid = len(old_set) >= 3
                
                # New colors
                new_c_a = new_color if idx_a == i else old_c_a
                new_c_b = new_color if idx_b == i else old_c_b
                new_c_c = new_color if idx_c == i else old_c_c
                new_c_d = new_color if idx_d == i else old_c_d
                new_set = {new_c_a, new_c_b, new_c_c, new_c_d}
                new_valid = len(new_set) >= 3
                
                if old_valid and not new_valid:
                    delta_viol += 1
                elif not old_valid and new_valid:
                    delta_viol -= 1
                    
            # Acceptance criterion
            if delta_viol < 0:
                # Better
                colors[i] = new_color
                current_viol += delta_viol
                if current_viol < best_viol:
                    best_viol = current_viol
                    best_colors = colors[:]
            else:
                # Worse or same
                # Accept with probability exp(-delta / T)
                if delta_viol == 0:
                    accept = True
                else:
                    accept = random.random() < pow(2.718281828, -delta_viol / T)
                
                if accept:
                    colors[i] = new_color
                    current_viol += delta_viol
                    
            T *= cooling
            
        # Check if we found a solution
        if best_viol == 0:
            unique_colors = sorted(set(best_colors))
            color_map = {c: i for i, c in enumerate(unique_colors)}
            return [color_map[c] for c in best_colors]
            
    # Fallback: use the seed's greedy approach if we haven't found a solution
    colors=[]
    for x in range(1,129):
        for color in range(128):
            if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors.append(color);break
    return colors
# EVOLVE-BLOCK-END
