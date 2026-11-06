import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 55.0
    
    N = 128
    
    aps4 = []
    for d in range(1, (N - 3) // 3 + 1):
        for i in range(0, N - 3 * d):
            aps4.append((i, i + d, i + 2 * d, i + 3 * d))
    
    pos_aps4 = [[] for _ in range(N)]
    for (a, b, c, d_pos) in aps4:
        pos_aps4[a].append((b, c, d_pos))
        pos_aps4[b].append((a, c, d_pos))
        pos_aps4[c].append((a, b, d_pos))
        pos_aps4[d_pos].append((a, b, c))
    
    def count_violations(colors):
        violations = 0
        for (a, b, c, d_pos) in aps4:
            if len({colors[a], colors[b], colors[c], colors[d_pos]}) < 3:
                violations += 1
        return violations

    def try_simulated_annealing(num_colors, max_iter=20000):
        best_colors = None
        best_violations = float('inf')
        
        for restart in range(5):
            if time.time() - start_time > DEADLINE:
                break
            
            colors = [random.randint(0, num_colors - 1) for _ in range(N)]
            current_violations = count_violations(colors)
            
            if current_violations == 0:
                return colors
            
            if current_violations < best_violations:
                best_violations = current_violations
                best_colors = colors[:]
            
            temperature = 5.0
            cooling_rate = 0.999
            
            for iteration in range(max_iter):
                if time.time() - start_time > DEADLINE:
                    return best_colors if best_violations == 0 else None
                
                pos = random.randint(0, N - 1)
                old_color = colors[pos]
                new_color = random.randint(0, num_colors - 1)
                
                if new_color == old_color:
                    continue
                
                delta_violations = 0
                for (p1, p2, p3) in pos_aps4[pos]:
                    c1, c2, c3 = colors[p1], colors[p2], colors[p3]
                    
                    # Inline distinct count check for speed
                    # Old
                    if old_color == c1:
                        if old_color == c2:
                            old_dist = 1 if old_color == c3 else 2
                        else:
                            old_dist = 2 if c1 == c3 else 3
                    else:
                        if c1 == c2:
                            old_dist = 2 if c1 == c3 else 3
                        else:
                            old_dist = 3 if c2 == c3 else 4
                    old_bad = 1 if old_dist < 3 else 0

                    # New
                    if new_color == c1:
                        if new_color == c2:
                            new_dist = 1 if new_color == c3 else 2
                        else:
                            new_dist = 2 if c1 == c3 else 3
                    else:
                        if c1 == c2:
                            new_dist = 2 if c1 == c3 else 3
                        else:
                            new_dist = 3 if c2 == c3 else 4
                    new_bad = 1 if new_dist < 3 else 0

                    delta_violations += new_bad - old_bad
                
                colors[pos] = new_color
                
                if delta_violations <= 0 or random.random() < np.exp(-delta_violations / temperature):
                    current_violations += delta_violations
                    
                    if current_violations == 0:
                        return colors
                    
                    if current_violations < best_violations:
                        best_violations = current_violations
                        best_colors = colors[:]
                else:
                    colors[pos] = old_color
                
                temperature *= cooling_rate
            
            if best_violations == 0:
                return best_colors
        
        return best_colors if best_violations == 0 else None

    for num_colors in [3, 4, 5]:
        if time.time() - start_time > DEADLINE:
            break
        
        result = try_simulated_annealing(num_colors)
        if result is not None:
            return result
    
    # Fallback: greedy with 8 colors
    colors = []
    for x in range(1, 129):
        for color in range(8):
            valid = True
            for d in range(1, (x-1)//3 + 1):
                if x-1-3*d >= 0:
                    if len({color, colors[x-1-d], colors[x-1-2*d], colors[x-1-3*d]}) < 3:
                        valid = False
                        break
            if valid:
                colors.append(color)
                break
        else:
            colors.append(0)
            
    return colors
