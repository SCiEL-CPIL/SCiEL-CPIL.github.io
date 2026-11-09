import time
import random
import itertools

def construct():
    q = 29
    squares = {x * x % q for x in range(1, q)}
    
    n_base = q
    colors_base = []
    for i in range(n_base):
        for j in range(i + 1, n_base):
            diff = (j - i) % q
            if diff in squares:
                colors_base.append(0)
            else:
                colors_base.append(1)
                
    # Build adjacency bitmasks for base graph
    # adj_base[c][i] is a bitmask of vertices j < 29 such that (i,j) is color c
    adj_base = [[0] * n_base for _ in range(2)]
    idx = 0
    for i in range(n_base):
        for j in range(i + 1, n_base):
            c = colors_base[idx]
            idx += 1
            adj_base[c][i] |= (1 << j)
            adj_base[c][j] |= (1 << i)
            
    # Precompute all 4-subsets of 0..28? No, S depends on v and c.
    # Precompute list of vertices for each mask?
    
    n_new = n_base + 1
    # new_colors[i] is color of edge (i, 29)
    
    def count_violations(new_colors):
        # new_colors is list of 29 ints (0 or 1)
        # Build mask_29_c
        mask_29 = [0, 0]
        for i in range(n_base):
            mask_29[new_colors[i]] |= (1 << i)
            
        total_violations = 0
        for c in range(2):
            m29 = mask_29[c]
            for v in range(n_base):
                # S is set of u such that (v,u) is c and (29,u) is c
                S = adj_base[c][v] & m29
                # Get list of vertices in S
                verts = [i for i in range(n_base) if (S >> i) & 1]
                if len(verts) < 4:
                    continue
                    
                # Iterate 4-subsets of verts
                for combo in itertools.combinations(verts, 4):
                    a, b, c_v, d = combo
                    # Check 3 partitions
                    # 1. {a,b} vs {c_v,d}
                    if (adj_base[c][a] & (1 << c_v)) and (adj_base[c][a] & (1 << d)) and \
                       (adj_base[c][b] & (1 << c_v)) and (adj_base[c][b] & (1 << d)):
                        total_violations += 1
                    # 2. {a,c_v} vs {b,d}
                    if (adj_base[c][a] & (1 << b)) and (adj_base[c][a] & (1 << d)) and \
                       (adj_base[c][c_v] & (1 << b)) and (adj_base[c][c_v] & (1 << d)):
                        total_violations += 1
                    # 3. {a,d} vs {b,c_v}
                    if (adj_base[c][a] & (1 << b)) and (adj_base[c][a] & (1 << c_v)) and \
                       (adj_base[c][d] & (1 << b)) and (adj_base[c][d] & (1 << c_v)):
                        total_violations += 1
        return total_violations

    start_time = time.time()
    deadline = 110.0
    
    # Try multiple random starts
    best_violations = float('inf')
    best_colors = None
    
    for _ in range(1000): # Try 1000 random initializations
        if time.time() > deadline:
            break
            
        new_colors = [random.randint(0, 1) for _ in range(n_base)]
        v = count_violations(new_colors)
        
        if v < best_violations:
            best_violations = v
            best_colors = new_colors[:]
            if v == 0:
                final_colors = colors_base + new_colors
                return {'n': n_new, 'colors': final_colors}
                
    # Local search from best found
    if best_colors is not None:
        current_colors = best_colors[:]
        current_violations = best_violations
        
        # Simple hill climbing with restarts
        while time.time() < deadline:
            improved = False
            # Try flipping each bit
            for i in range(n_base):
                if time.time() > deadline:
                    break
                
                old_color = current_colors[i]
                new_color = 1 - old_color
                current_colors[i] = new_color
                
                new_violations = count_violations(current_colors)
                
                if new_violations < current_violations:
                    current_violations = new_violations
                    improved = True
                    if current_violations == 0:
                        final_colors = colors_base + current_colors
                        return {'n': n_new, 'colors': final_colors}
                else:
                    # Revert
                    current_colors[i] = old_color
                    
            if not improved:
                # Restart with new random
                current_colors = [random.randint(0, 1) for _ in range(n_base)]
                current_violations = count_violations(current_colors)
                if current_violations < best_violations:
                    best_violations = current_violations
                    best_colors = current_colors[:]
                    if best_violations == 0:
                        final_colors = colors_base + best_colors
                        return {'n': n_new, 'colors': final_colors}

    # Return n=29 if no solution found
    return {'n': n_base, 'colors': colors_base}
