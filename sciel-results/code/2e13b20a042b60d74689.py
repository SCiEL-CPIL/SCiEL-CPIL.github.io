import random
import time
import sys

def construct():
    N = 128
    # Precompute all 4-APs
    aps = []
    ap_indices = [[] for _ in range(N)]
    for i in range(N):
        # i is 0-indexed, position is i+1
        # max d such that i+3d < N
        max_d = (N - 1 - i) // 3
        for d in range(1, max_d + 1):
            ap = (i, i+d, i+2*d, i+3*d)
            aps.append(ap)
            for idx in ap:
                ap_indices[idx].append(len(aps) - 1)
    
    def get_cost(colors):
        cost = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if len({c0, c1, c2, c3}) < 3:
                cost += 1
        return cost

    def get_delta_cost(colors, idx, new_color):
        # Calculate change in cost if colors[idx] becomes new_color
        old_color = colors[idx]
        if old_color == new_color:
            return 0
        
        delta = 0
        for ap_idx in ap_indices[idx]:
            ap = aps[ap_idx]
            # Get current colors in this AP
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            
            # Determine if this AP is currently a violation
            current_violation = 1 if len({c0, c1, c2, c3}) < 3 else 0
            
            # Determine new colors
            new_c0, new_c1, new_c2, new_c3 = c0, c1, c2, c3
            if ap[0] == idx: new_c0 = new_color
            if ap[1] == idx: new_c1 = new_color
            if ap[2] == idx: new_c2 = new_color
            if ap[3] == idx: new_c3 = new_color
            
            new_violation = 1 if len({new_c0, new_c1, new_c2, new_c3}) < 3 else 0
            
            delta += new_violation - current_violation
            
        return delta

    start_time = time.time()
    deadline = 85.0 # Safety margin
    
    # Try increasing number of colors
    for k in range(3, 15):
        # Multiple restarts for each k
        for restart in range(50):
            if time.time() > deadline:
                break
                
            colors = [random.randint(0, k-1) for _ in range(N)]
            current_cost = get_cost(colors)
            
            if current_cost == 0:
                return colors
            
            T = 1.0
            best_cost = current_cost
            best_colors = colors[:]
            
            while current_cost > 0:
                if time.time() > deadline:
                    break
                
                # Pick a random index to change
                # Better: pick an index involved in a violation
                # For simplicity, random index first
                idx = random.randint(0, N-1)
                new_color = random.randint(0, k-1)
                
                delta = get_delta_cost(colors, idx, new_color)
                
                # Simulated annealing acceptance
                if delta <= 0 or (current_cost > 0 and random.random() < pow(2.71828, -delta / T)):
                    colors[idx] = new_color
                    current_cost += delta
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_colors = colors[:]
                    
                    if current_cost == 0:
                        return colors
                
                # Cool down
                T *= 0.999
                if T < 0.01:
                    T = 0.01

    # If no solution found, return a random valid one? 
    # Or just return the best found?
    # The task requires a valid solution. 
    # If we fail to find one, we might return an invalid one, which scores 0.
    # But for small N and small k, it should be found quickly.
    
    # Fallback: return a coloring with many colors (e.g., 128) which is trivially valid
    # but score will be low.
    return list(range(N))
