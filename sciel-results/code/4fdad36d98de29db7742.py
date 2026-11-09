import time
import random
import math
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    def time_left():
        return deadline - time.time()

    def solve_n(n, seed=42):
        if n < 4:
            return None
            
        random.seed(seed)
        
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_idx = {t: i for i, t in enumerate(triples)}
        
        # Generate K4s and K5s
        k4s = list(combinations(range(n), 4))
        k5s = list(combinations(range(n), 5))
        
        # Map each K4 to its 4 triple indices
        k4_triples = []
        for k4 in k4s:
            ts = []
            for tri in combinations(k4, 3):
                ts.append(triple_idx[tri])
            k4_triples.append(ts)
            
        # Map each K5 to its 10 triple indices
        k5_triples = []
        for k5 in k5s:
            ts = []
            for tri in combinations(k5, 3):
                ts.append(triple_idx[tri])
            k5_triples.append(ts)
            
        # Map triple index to list of K4 indices and K5 indices
        triple_to_k4 = [[] for _ in range(num_triples)]
        for i, ts in enumerate(k4_triples):
            for t_idx in ts:
                triple_to_k4[t_idx].append(i)
                
        triple_to_k5 = [[] for _ in range(num_triples)]
        for i, ts in enumerate(k5_triples):
            for t_idx in ts:
                triple_to_k5[t_idx].append(i)
                
        # Initialize colors: 0=Red, 1=Blue
        # Heuristic: random or structured
        # Let's try random
        colors = np.array([random.randint(0, 1) for _ in range(num_triples)], dtype=np.int8)
        
        # Initialize counts
        red_count_k4 = np.zeros(len(k4s), dtype=np.int8)
        blue_count_k5 = np.zeros(len(k5s), dtype=np.int8)
        
        for i, ts in enumerate(k4_triples):
            # Sum of (1 - colors[ts]) i.e., red is 0, so 1-color is red? 
            # Wait, colors[ts] is 0 for red. 
            # So red count is sum(1 - colors[ts])? No, if color is 0, it's red.
            # So red count is sum(1 - colors[ts]) if 0=red? 
            # Let's stick to: color 0 is Red.
            # red_count = sum(1 for c in ts if colors[c]==0)
            # Using numpy: np.sum(colors[ts] == 0)
            red_count_k4[i] = np.sum(colors[ts] == 0)
            
        for i, ts in enumerate(k5_triples):
            # blue count is sum(1 for c in ts if colors[c]==1)
            blue_count_k5[i] = np.sum(colors[ts] == 1)
            
        red_violations = np.sum(red_count_k4 == 4)
        blue_violations = np.sum(blue_count_k5 == 10)
        
        if red_violations == 0 and blue_violations == 0:
            return colors.tolist()
            
        T = 1.0
        cooling = 0.999
        min_T = 0.01
        max_steps = 500000
        
        for step in range(max_steps):
            if time_left() < 5.0:
                break
                
            # Pick a random triple
            t_idx = random.randint(0, num_triples - 1)
            
            old_color = colors[t_idx]
            new_color = 1 - old_color
            
            delta_red = 0
            delta_blue = 0
            
            # Update K4 counts
            k4_indices = triple_to_k4[t_idx]
            if k4_indices:
                old_counts = red_count_k4[k4_indices]
                if old_color == 0: # Was Red, becoming Blue
                    # Red count decreases by 1
                    new_counts = old_counts - 1
                    # Violation if count == 4
                    # Old violations: old_counts == 4
                    # New violations: new_counts == 4
                    # delta = new_violations - old_violations
                    # If old was 4, new is 3 -> -1
                    # If old was 5, new is 4 -> +1 (impossible, max 4)
                    # If old was 3, new is 2 -> 0
                    # So delta_red = sum(new_counts == 4) - sum(old_counts == 4)
                    delta_red += np.sum(new_counts == 4) - np.sum(old_counts == 4)
                    red_count_k4[k4_indices] = new_counts
                else: # Was Blue, becoming Red
                    # Red count increases by 1
                    new_counts = old_counts + 1
                    delta_red += np.sum(new_counts == 4) - np.sum(old_counts == 4)
                    red_count_k4[k4_indices] = new_counts

            # Update K5 counts
            k5_indices = triple_to_k5[t_idx]
            if k5_indices:
                old_counts = blue_count_k5[k5_indices]
                if old_color == 1: # Was Blue, becoming Red
                    # Blue count decreases by 1
                    new_counts = old_counts - 1
                    delta_blue += np.sum(new_counts == 10) - np.sum(old_counts == 10)
                    blue_count_k5[k5_indices] = new_counts
                else: # Was Red, becoming Blue
                    # Blue count increases by 1
                    new_counts = old_counts + 1
                    delta_blue += np.sum(new_counts == 10) - np.sum(old_counts == 10)
                    blue_count_k5[k5_indices] = new_counts
                    
            delta_E = delta_red + delta_blue
            
            accepted = False
            if delta_E < 0:
                accepted = True
            else:
                if T > min_T:
                    prob = math.exp(-delta_E / T)
                    if random.random() < prob:
                        accepted = True
                
            if accepted:
                colors[t_idx] = new_color
                red_violations += delta_red
                blue_violations += delta_blue
                
                if red_violations == 0 and blue_violations == 0:
                    return colors.tolist()
            else:
                # Rollback counts
                if k4_indices:
                    if old_color == 0:
                        red_count_k4[k4_indices] += 1
                    else:
                        red_count_k4[k4_indices] -= 1
                if k5_indices:
                    if old_color == 1:
                        blue_count_k5[k5_indices] += 1
                    else:
                        blue_count_k5[k5_indices] -= 1
            
            T *= cooling
            
        return None

    best_n = 9
    # Fallback coloring
    best_colors = [0 if (a+b+c) % 3 == 0 else 1 for a,b,c in combinations(range(9), 3)]
    
    # Try increasing n
    # We know n=34 is possible. Let's aim for 34.
    # If we have time, try 35.
    for n in [34, 35]:
        if time_left() < 20.0:
            break
            
        result = solve_n(n)
        if result is not None:
            best_n = n
            best_colors = result
            # If we found 34, we can stop or try 35.
            # If we found 35, that's even better.
        else:
            # If 34 fails, try 33? 
            # But 34 is the target. 
            # Let's try a few seeds for 34 if it fails.
            pass

    # If 34 failed with seed 42, try a few more seeds
    if best_n < 34:
        for seed in [1, 2, 3, 4, 5]:
            if time_left() < 10.0:
                break
            result = solve_n(34, seed=seed)
            if result is not None:
                best_n = 34
                best_colors = result
                break
                
    if best_n < 34:
        # Fallback to 33
        if time_left() > 5.0:
            result = solve_n(33, seed=42)
            if result is not None:
                best_n = 33
                best_colors = result

    return {'n': best_n, 'colors': best_colors}
