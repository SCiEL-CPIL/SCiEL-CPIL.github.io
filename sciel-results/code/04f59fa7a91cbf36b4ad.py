import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    best_n = 9
    best_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]
    
    # Try to find valid coloring for n=34
    # If successful, try 35, 36... but 34 is the known bound, so 34 is a great target.
    # We will focus on 34.
    
    n_target = 34
    
    # Precomputation for n_target
    vertices = range(n_target)
    triples = list(combinations(vertices, 3))
    num_triples = len(triples)
    idx_map = {t: i for i, t in enumerate(triples)}
    
    four_sets = list(combinations(vertices, 4))
    num_4sets = len(four_sets)
    
    five_sets = list(combinations(vertices, 5))
    num_5sets = len(five_sets)
    
    # Map triple index to list of 4-set indices it belongs to
    triple_to_4sets = [[] for _ in range(num_triples)]
    for i, quad in enumerate(four_sets):
        a, b, c, d = quad
        t1 = idx_map[(a, b, c)]
        t2 = idx_map[(a, b, d)]
        t3 = idx_map[(a, c, d)]
        t4 = idx_map[(b, c, d)]
        triple_to_4sets[t1].append(i)
        triple_to_4sets[t2].append(i)
        triple_to_4sets[t3].append(i)
        triple_to_4sets[t4].append(i)
        
    # Map triple index to list of 5-set indices it belongs to
    triple_to_5sets = [[] for _ in range(num_triples)]
    for i, quint in enumerate(five_sets):
        for t in combinations(quint, 3):
            t_idx = idx_map[t]
            triple_to_5sets[t_idx].append(i)
            
    # Local search
    # We will run multiple restarts
    max_restarts = 10
    restarts_done = 0
    
    while time.time() < deadline - 10 and restarts_done < max_restarts:
        restarts_done += 1
        
        # Initialize random coloring
        colors = np.random.randint(0, 2, num_triples)
        
        # Initialize counts
        red_counts_4 = np.zeros(num_4sets, dtype=np.int32)
        blue_counts_5 = np.zeros(num_5sets, dtype=np.int32)
        
        # Compute initial counts
        # For each 4-set, count reds
        for i, quad in enumerate(four_sets):
            a, b, c, d = quad
            t1 = idx_map[(a, b, c)]
            t2 = idx_map[(a, b, d)]
            t3 = idx_map[(a, c, d)]
            t4 = idx_map[(b, c, d)]
            red_counts_4[i] = (colors[t1] == 0) + (colors[t2] == 0) + (colors[t3] == 0) + (colors[t4] == 0)
            
        # For each 5-set, count blues
        for i, quint in enumerate(five_sets):
            count = 0
            for t in combinations(quint, 3):
                if colors[idx_map[t]] == 1:
                    count += 1
            blue_counts_5[i] = count
            
        # Count violations
        violations = np.sum(red_counts_4 == 4) + np.sum(blue_counts_5 == 10)
        
        if violations == 0:
            best_n = n_target
            best_colors = colors.tolist()
            break # Found solution, no need to try larger n or more restarts for this n
            
        # Simulated Annealing / Hill Climbing
        # Temperature starts high, cools down
        temp = 5.0
        min_temp = 0.1
        cooling = 0.99
        
        # We want to minimize violations
        # Step: pick random triple, calculate delta violations
        
        while time.time() < deadline - 10:
            # Pick random triple
            t = np.random.randint(0, num_triples)
            current_color = colors[t]
            
            delta_viol = 0
            
            # Update 4-sets
            for s_idx in triple_to_4sets[t]:
                old_count = red_counts_4[s_idx]
                if current_color == 0: # Flipping red to blue
                    new_count = old_count - 1
                else: # Flipping blue to red
                    new_count = old_count + 1
                    
                old_viol = 1 if old_count == 4 else 0
                new_viol = 1 if new_count == 4 else 0
                delta_viol += new_viol - old_viol
                
            # Update 5-sets
            for s_idx in triple_to_5sets[t]:
                old_count = blue_counts_5[s_idx]
                if current_color == 1: # Flipping blue to red
                    new_count = old_count - 1
                else: # Flipping red to blue
                    new_count = old_count + 1
                    
                old_viol = 1 if old_count == 10 else 0
                new_viol = 1 if new_count == 10 else 0
                delta_viol += new_viol - old_viol
                
            # Acceptance criterion
            # If delta < 0, always accept.
            # If delta == 0, accept with prob 0.5.
            # If delta > 0, accept with prob exp(-delta/temp).
            if delta_viol < 0:
                accept = True
            elif delta_viol == 0:
                accept = random.random() < 0.5
            else:
                accept = random.random() < np.exp(-delta_viol / temp)
                
            if accept:
                # Apply flip
                colors[t] = 1 - current_color
                new_color = colors[t]
                
                # Update counts
                for s_idx in triple_to_4sets[t]:
                    if new_color == 0:
                        red_counts_4[s_idx] += 1
                    else:
                        red_counts_4[s_idx] -= 1
                        
                for s_idx in triple_to_5sets[t]:
                    if new_color == 1:
                        blue_counts_5[s_idx] += 1
                    else:
                        blue_counts_5[s_idx] -= 1
                        
                violations += delta_viol
                
                if violations == 0:
                    best_n = n_target
                    best_colors = colors.tolist()
                    break
                    
            temp *= cooling
            if temp < min_temp:
                break
                
        if best_n == n_target:
            break
            
    return {'n': best_n, 'colors': best_colors}
