"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    n = 128
    
    # Precompute all 4-term APs
    aps = []
    for d in range(1, (n - 1) // 3 + 1):
        for i in range(n - 3 * d):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))
    
    M = len(aps)
    
    # Precompute which APs involve each index
    ap_indices_for_pos = [[] for _ in range(n)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            ap_indices_for_pos[pos].append(idx)
    
    def count_violations(colors):
        count = 0
        for ap in aps:
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if len({c0, c1, c2, c3}) < 3:
                count += 1
        return count
    
    def get_violations_for_pos(colors, pos, new_val):
        v_count = 0
        for ap_idx in ap_indices_for_pos[pos]:
            ap = aps[ap_idx]
            c0 = colors[ap[0]] if ap[0] != pos else new_val
            c1 = colors[ap[1]] if ap[1] != pos else new_val
            c2 = colors[ap[2]] if ap[2] != pos else new_val
            c3 = colors[ap[3]] if ap[3] != pos else new_val
            if len({c0, c1, c2, c3}) < 3:
                v_count += 1
        return v_count
    
    def greedy_construct(k):
        """Greedy construction with k colors, trying to minimize distinct colors used."""
        colors = [-1] * n
        
        # Try to use as few distinct colors as possible
        for pos in range(n):
            # Try colors 0..k-1, preferring already-used colors
            used = set(colors[:pos])
            color_order = list(used) + [c for c in range(k) if c not in used]
            
            found = False
            for color in color_order:
                valid = True
                for ap_idx in ap_indices_for_pos[pos]:
                    ap = aps[ap_idx]
                    # Check if assigning color to pos creates violation
                    c0 = colors[ap[0]] if ap[0] != pos else color
                    c1 = colors[ap[1]] if ap[1] != pos else color
                    c2 = colors[ap[2]] if ap[2] != pos else color
                    c3 = colors[ap[3]] if ap[3] != pos else color
                    
                    # Only check if all other positions in AP are assigned
                    if -1 in (c0, c1, c2, c3):
                        continue
                    
                    if len({c0, c1, c2, c3}) < 3:
                        valid = False
                        break
                
                if valid:
                    colors[pos] = color
                    found = True
                    break
            
            if not found:
                # Fallback: pick any color that doesn't create immediate violation
                for color in range(k):
                    valid = True
                    for ap_idx in ap_indices_for_pos[pos]:
                        ap = aps[ap_idx]
                        c0 = colors[ap[0]] if ap[0] != pos else color
                        c1 = colors[ap[1]] if ap[1] != pos else color
                        c2 = colors[ap[2]] if ap[2] != pos else color
                        c3 = colors[ap[3]] if ap[3] != pos else color
                        if -1 in (c0, c1, c2, c3):
                            continue
                        if len({c0, c1, c2, c3}) < 3:
                            valid = False
                            break
                    if valid:
                        colors[pos] = color
                        break
                else:
                    # Last resort
                    colors[pos] = 0
        
        return colors
    
    def local_search(colors, k, max_steps=50000):
        """Local search to reduce number of distinct colors."""
        colors = list(colors)
        current_violations = count_violations(colors)
        
        if current_violations == 0:
            return colors
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Pick a random position involved in a violation
            violated_aps = [idx for idx, ap in enumerate(aps) 
                          if len({colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}) < 3]
            
            if not violated_aps:
                break
            
            # Pick a random violated AP
            ap_idx = random.choice(violated_aps)
            ap = aps[ap_idx]
            
            # Pick a random position in this AP to change
            pos = random.choice(ap)
            current_val = colors[pos]
            
            # Try all k values
            best_new_val = current_val
            best_violations = current_violations
            
            for new_val in range(k):
                if new_val == current_val:
                    continue
                v = get_violations_for_pos(colors, pos, new_val)
                if v < best_violations:
                    best_violations = v
                    best_new_val = new_val
            
            # If no improvement, pick random
            if best_new_val == current_val:
                best_new_val = random.randint(0, k - 1)
            
            colors[pos] = best_new_val
            current_violations = count_violations(colors)
            
            if current_violations == 0:
                break
        
        return colors
    
    # Strategy: Try increasing k, use greedy + local search
    best_solution = None
    
    for k in range(3, 15):
        if time.time() > deadline:
            break
        
        # Time budget for this k
        k_deadline = min(deadline, start_time + 5.0 + k * 3.0)
        
        found = False
        
        # Multiple restarts with different random seeds
        for restart in range(10):
            if time.time() > k_deadline:
                break
            
            # Greedy construction
            colors = greedy_construct(k)
            
            # Check if valid
            if count_violations(colors) == 0:
                best_solution = colors
                found = True
                break
            
            # Local search
            colors = local_search(colors, k, max_steps=20000)
            
            if count_violations(colors) == 0:
                best_solution = colors
                found = True
                break
        
        if found:
            break
    
    # If no solution found, use fallback: all distinct colors
    if best_solution is None:
        best_solution = list(range(n))
    
    return best_solution
# EVOLVE-BLOCK-END
