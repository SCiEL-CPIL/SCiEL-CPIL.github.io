import time
import random
import numpy as np

def construct():
    N = 128
    deadline = time.time() + 55.0
    
    # Precompute all 4-term APs
    aps = []
    for d in range(1, (N - 1) // 3 + 1):
        for i in range(N - 3 * d):
            aps.append((i, i + d, i + 2 * d, i + 3 * d))
    
    M = len(aps)
    
    # Map position to list of (ap_index, role)
    pos_to_aps = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for j, pos in enumerate(ap):
            pos_to_aps[pos].append((idx, j))

    def is_violation(c0, c1, c2, c3):
        # Violation if distinct colors < 3
        # This is equivalent to:
        # 1. All 4 same
        # 2. 3 same, 1 different
        # 3. 2 same, 2 same (and the pairs are different)
        # Basically, if the set size is 1 or 2.
        if c0 == c1:
            if c2 == c3:
                return True # 2 pairs or 4 same
            if c2 == c0 or c3 == c0:
                return True # 3 same
            if c2 == c3:
                return True # 2 pairs (c0=c1, c2=c3, c0!=c2)
            return False
        if c0 == c2:
            if c1 == c3:
                return True
            if c1 == c0 or c3 == c0:
                return True
            if c1 == c3:
                return True
            return False
        if c0 == c3:
            if c1 == c2:
                return True
            if c1 == c0 or c2 == c0:
                return True
            if c1 == c2:
                return True
            return False
        if c1 == c2:
            if c0 == c3:
                return True
            if c0 == c1 or c3 == c1:
                return True
            if c0 == c3:
                return True
            return False
        if c1 == c3:
            if c0 == c2:
                return True
            if c0 == c1 or c2 == c1:
                return True
            if c0 == c2:
                return True
            return False
        if c2 == c3:
            if c0 == c1:
                return True
            if c0 == c2 or c1 == c2:
                return True
            if c0 == c1:
                return True
            return False
        return False

    def count_distinct(colors):
        return len(set(colors))

    def simulated_annealing(K, seed, max_steps=50000):
        random.seed(seed)
        # Initialize with K colors
        colors = [random.randint(0, K - 1) for _ in range(N)]
        
        # Initialize violation state
        ap_violation = [False] * M
        total_violations = 0
        for idx, ap in enumerate(aps):
            c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
            if is_violation(c0, c1, c2, c3):
                ap_violation[idx] = True
                total_violations += 1
        
        color_counts = [0] * K
        for c in colors:
            color_counts[c] += 1
        distinct_colors = sum(1 for c in color_counts if c > 0)
        
        # Energy: V^2 + W * D
        # We want to minimize V first, then D.
        # If V > 0, we want to reduce V.
        # If V == 0, we want to reduce D.
        # Let's use E = total_violations * 1000 + distinct_colors
        # This ensures V is prioritized.
        
        T = 10.0
        cooling = 0.999
        best_E = total_violations * 1000 + distinct_colors
        best_colors = colors[:]
        
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Pick random position
            pos = random.randint(0, N - 1)
            old_color = colors[pos]
            new_color = random.randint(0, K - 1)
            
            if old_color == new_color:
                continue
            
            # Calculate Delta E
            delta_v = 0
            # Check affected APs
            for ap_idx, role in pos_to_aps[pos]:
                ap = aps[ap_idx]
                # Get current colors
                c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                old_v = ap_violation[ap_idx]
                
                # Update color in temp tuple
                if role == 0:
                    nc = (new_color, c1, c2, c3)
                elif role == 1:
                    nc = (c0, new_color, c2, c3)
                elif role == 2:
                    nc = (c0, c1, new_color, c3)
                else:
                    nc = (c0, c1, c2, new_color)
                
                new_v = is_violation(*nc)
                
                if new_v and not old_v:
                    delta_v += 1
                elif not new_v and old_v:
                    delta_v -= 1
            
            # Update distinct colors
            delta_d = 0
            if color_counts[old_color] == 1:
                # Removing last of old_color
                delta_d -= 1
            if color_counts[new_color] == 0:
                # Adding first of new_color
                delta_d += 1
            
            delta_E = delta_v * 1000 + delta_d
            
            # Accept?
            if delta_E < 0:
                accept = True
            else:
                if T > 1e-6:
                    accept = random.random() < np.exp(-delta_E / T)
                else:
                    accept = False
            
            if accept:
                # Apply move
                colors[pos] = new_color
                total_violations += delta_v
                color_counts[old_color] -= 1
                color_counts[new_color] += 1
                distinct_colors += delta_d
                
                # Update ap_violation
                for ap_idx, role in pos_to_aps[pos]:
                    ap = aps[ap_idx]
                    c0, c1, c2, c3 = colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]
                    ap_violation[ap_idx] = is_violation(c0, c1, c2, c3)
                
                current_E = total_violations * 1000 + distinct_colors
                if current_E < best_E:
                    best_E = current_E
                    best_colors = colors[:]
                    if total_violations == 0:
                        # Found a valid coloring. 
                        # We can try to reduce colors further by restarting with K-1?
                        # Or just return this and let the outer loop handle K reduction.
                        pass
            
            T *= cooling
        
        # Check if best is valid
        if best_E < 1000: # i.e. total_violations == 0
            return best_colors, best_E
        return None, best_E

    # Try K from 3 to 128
    best_solution = None
    
    for K in range(3, 128):
        if time.time() > deadline:
            break
        
        found = False
        # Try multiple seeds for this K
        for seed in range(10):
            if time.time() > deadline:
                break
            sol, E = simulated_annealing(K, seed, max_steps=20000)
            if sol is not None:
                best_solution = sol
                found = True
                break
        
        if found:
            break
    
    if best_solution is None:
        # Fallback: All distinct
        best_solution = list(range(N))
        
    return best_solution
