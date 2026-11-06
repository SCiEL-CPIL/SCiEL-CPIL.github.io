import time
import random
import sys

def construct():
    n = 128
    deadline = time.time() + 85.0  # Safety margin
    
    # Precompute all 4-term APs (0-indexed)
    # AP is (i, i+d, i+2d, i+3d)
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        for i in range(0, n - 3 * d):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))
    
    # For each position, list indices of APs it belongs to
    pos_to_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(idx)
    
    def count_violations(colors):
        v = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if len({c0, c1, c2, c3}) < 3:
                v += 1
        return v
    
    def get_ap_colors(colors, ap):
        return (colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]])
    
    def is_ap_valid(colors, ap):
        c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
        return len({c0, c1, c2, c3}) >= 3
    
    def local_search(k, time_limit):
        """
        Randomized local search to find a valid coloring with k colors.
        Returns the coloring if found, else None.
        """
        # Initialize random
        colors = [random.randint(0, k - 1) for _ in range(n)]
        current_violations = count_violations(colors)
        
        if current_violations == 0:
            return colors
        
        best_colors = colors[:]
        best_violations = current_violations
        
        # Simulated Annealing parameters
        temp = 1.0
        cooling = 0.999
        min_temp = 0.01
        
        while time.time() < time_limit:
            # Pick a random position
            pos = random.randint(0, n - 1)
            old_color = colors[pos]
            
            # Try all other colors
            best_new_color = old_color
            best_delta = 0
            
            # Calculate current violations for APs containing pos
            # We need to check which of those APs are currently invalid
            # And which become valid/invalid if we change color
            
            # Optimization: Only check APs containing pos
            affected_aps = pos_to_aps[pos]
            
            # Count how many affected APs are currently invalid
            currently_invalid_count = 0
            for ap_idx in affected_aps:
                ap = aps[ap_idx]
                if not is_ap_valid(colors, ap):
                    currently_invalid_count += 1
            
            # Try each new color
            for new_color in range(k):
                if new_color == old_color:
                    continue
                
                colors[pos] = new_color
                
                # Recount violations for affected APs
                new_invalid_count = 0
                for ap_idx in affected_aps:
                    ap = aps[ap_idx]
                    if not is_ap_valid(colors, ap):
                        new_invalid_count += 1
                
                delta = new_invalid_count - currently_invalid_count
                
                if delta < best_delta:
                    best_delta = delta
                    best_new_color = new_color
                elif delta == best_delta and random.random() < 0.5:
                    best_new_color = new_color
            
            # Revert to old color for now, then apply best
            colors[pos] = old_color
            
            # Decide whether to accept
            if best_delta < 0:
                accept = True
            elif best_delta == 0:
                accept = True
            else:
                # Accept worse with probability exp(-delta / temp)
                if temp > 0:
                    prob = pow(sys.float_info.max, -best_delta / temp) # Approximation, better use math.exp
                    try:
                        import math
                        prob = math.exp(-best_delta / temp)
                    except:
                        prob = 0.0
                    accept = random.random() < prob
                else:
                    accept = False
            
            if accept:
                colors[pos] = best_new_color
                current_violations += best_delta
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = colors[:]
                
                if current_violations == 0:
                    return colors
            
            temp *= cooling
            if temp < min_temp:
                break
        
        return best_colors if best_violations == 0 else None
    
    # Try increasing k
    k_values = [3, 4, 5, 6, 8, 10, 16, 32, 64, 128]
    
    for k in k_values:
        if time.time() > deadline - 5.0:
            break
        
        # Give each k a time slice, but prioritize smaller k
        # Small k is harder, so give more time? Or just fixed time?
        # Let's give 10s to k=3, 5s to k=4, etc.
        time_slice = 10.0 if k <= 4 else 5.0
        if time.time() + time_slice > deadline - 2.0:
            time_slice = max(1.0, deadline - 2.0 - time.time())
        
        result = local_search(k, time.time() + time_slice)
        if result is not None:
            return result
    
    # Fallback: Identity mapping (always valid, uses 128 colors)
    # This ensures we return a valid solution, though not optimal in color count
    return list(range(n))
