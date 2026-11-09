import numpy as np
import random
import time

def construct():
    N = 128
    # Pre-compute all 4-APs
    aps = []
    ap_map = [[] for _ in range(N)] # ap_map[i] contains indices of APs containing i
    
    ap_id = 0
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            terms = [a - 1, a - 1 + d, a - 1 + 2 * d, a - 1 + 3 * d]
            aps.append(terms)
            for t in terms:
                ap_map[t].append(ap_id)
            ap_id += 1
            
    num_aps = len(aps)
    
    def count_bad(colors):
        bad = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if len({c0, c1, c2, c3}) < 3:
                bad += 1
        return bad

    def count_bad_local(colors, i, old_c, new_c):
        # Only check APs containing i
        delta = 0
        for ap_idx in ap_map[i]:
            ap = aps[ap_idx]
            # Get colors of other 3 terms
            other_indices = [t for t in ap if t != i]
            c_others = [colors[t] for t in other_indices]
            
            # Old state
            old_set = {old_c} | set(c_others)
            old_bad = 1 if len(old_set) < 3 else 0
            
            # New state
            new_set = {new_c} | set(c_others)
            new_bad = 1 if len(new_set) < 3 else 0
            
            delta += new_bad - old_bad
        return delta

    # Try with increasing number of colors
    for K in range(3, 10):
        # Simulated Annealing
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        max_iters = 50000
        
        # Initialize
        colors = np.random.randint(0, K, N)
        current_bad = count_bad(colors)
        
        if current_bad == 0:
            # Check distinct colors
            used = set(colors)
            return list(colors)
            
        for it in range(max_iters):
            if T < min_T:
                break
            i = random.randint(0, N - 1)
            old_c = colors[i]
            new_c = random.randint(0, K - 1)
            if new_c == old_c:
                continue
                
            delta = count_bad_local(colors, i, old_c, new_c)
            
            # Acceptance
            if delta < 0:
                colors[i] = new_c
                current_bad += delta
            elif delta == 0:
                # Check if this reduces color count
                # Count occurrences of old_c
                # This is O(N) which is slow. 
                # Optimization: maintain counts array
                # For now, accept randomly if delta 0
                if random.random() < 0.5:
                    colors[i] = new_c
            else:
                # Boltzmann factor
                prob = np.exp(-delta / T)
                if random.random() < prob:
                    colors[i] = new_c
                    current_bad += delta
            
            T *= cooling
            
        # After SA, if bad is 0, try to minimize colors
        if current_bad == 0:
            # Greedy reduction of colors
            changed = True
            while changed:
                changed = False
                used_colors = set(colors)
                for c in list(used_colors):
                    # Try to recolor all c's to another color
                    # This is complex. Instead, just return if K is small.
                    pass
            
            # If we found a valid coloring with K colors, return it
            # We can try to map colors to 0..K-1 to ensure minimal labels?
            # The task says "labels need not be consecutive".
            # But "Minimize the number of labels actually used".
            # So the count of distinct values is what matters.
            
            # Try to reduce K by re-running with K-1?
            # For now, return the valid one found.
            return list(colors)
            
    # Fallback: Greedy
    colors = []
    for x in range(1, N + 1):
        for c in range(128):
            valid = True
            # Check all 4-APs ending at x? No, all 4-APs involving x.
            # The seed code did this.
            # We just use a simple greedy to ensure validity, even if colors are many.
            for ap in aps:
                if x - 1 in ap:
                    other = [t for t in ap if t != x - 1]
                    if len(other) == 3:
                        c_others = [colors[t] for t in other]
                        if len({c} | set(c_others)) < 3:
                            valid = False
                            break
            if valid:
                colors.append(c)
                break
        else:
            colors.append(127) # Should not happen
            
    return colors
