import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 100.0
    
    best_n = 1
    best_colors = None
    
    # We prioritize finding a valid coloring for high n
    # Known bound is 34.
    
    for n in range(34, 19, -1):
        if time.time() > deadline - 10.0:
            break
            
        triples = list(combinations(range(n), 3))
        M = len(triples)
        
        # Precompute reverse indices
        # This is the most expensive part, but done once per n
        # For n=34, M=5984.
        
        k4s = list(combinations(range(n), 4))
        k5s = list(combinations(range(n), 5))
        
        k4_to_idx = {k4: i for i, k4 in enumerate(k4s)}
        k5_to_idx = {k5: i for i, k5 in enumerate(k5s)}
        
        triple_k4s = [[] for _ in range(M)]
        triple_k5s = [[] for _ in range(M)]
        
        # To speed up, we can use sets or lists
        for i, (a, b, c) in enumerate(triples):
            for d in range(n):
                if d == a or d == b or d == c:
                    continue
                k4 = tuple(sorted((a, b, c, d)))
                triple_k4s[i].append(k4_to_idx[k4])
            
            remaining = [v for v in range(n) if v not in (a, b, c)]
            for d, e in combinations(remaining, 2):
                k5 = tuple(sorted((a, b, c, d, e)))
                triple_k5s[i].append(k5_to_idx[k5])
                
        # Initializations to try
        inits = []
        
        # 1. Mod 3 seed
        init_mod3 = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]
        inits.append(init_mod3)
        
        # 2. Random seeds
        for _ in range(5):
            inits.append([random.randint(0, 1) for _ in range(M)])
            
        found = False
        for init_colors in inits:
            if time.time() > deadline - 2.0:
                break
                
            colors = init_colors[:]
            
            red_count_k4 = [0] * len(k4s)
            blue_count_k5 = [0] * len(k5s)
            
            for i, c in enumerate(colors):
                if c == 0:
                    for k4_idx in triple_k4s[i]:
                        red_count_k4[k4_idx] += 1
                else:
                    for k5_idx in triple_k5s[i]:
                        blue_count_k5[k5_idx] += 1
                        
            current_violations = sum(1 for x in red_count_k4 if x == 4) + sum(1 for x in blue_count_k5 if x == 10)
            
            if current_violations == 0:
                best_n = n
                best_colors = colors
                found = True
                break
                
            # SA
            T = 1.0
            cooling = 0.995
            min_T = 0.01
            steps = 0
            max_steps = 20000
            
            while T > min_T and steps < max_steps:
                if time.time() > deadline - 1.0:
                    break
                    
                i = random.randint(0, M - 1)
                old_color = colors[i]
                new_color = 1 - old_color
                
                delta_v = 0
                
                if old_color == 0:
                    for k4_idx in triple_k4s[i]:
                        if red_count_k4[k4_idx] == 4:
                            delta_v -= 1
                        red_count_k4[k4_idx] -= 1
                else:
                    for k4_idx in triple_k4s[i]:
                        if red_count_k4[k4_idx] == 3:
                            delta_v += 1
                        red_count_k4[k4_idx] += 1
                        
                if old_color == 1:
                    for k5_idx in triple_k5s[i]:
                        if blue_count_k5[k5_idx] == 10:
                            delta_v -= 1
                        blue_count_k5[k5_idx] -= 1
                else:
                    for k5_idx in triple_k5s[i]:
                        if blue_count_k5[k5_idx] == 9:
                            delta_v += 1
                        blue_count_k5[k5_idx] += 1
                        
                if delta_v < 0:
                    colors[i] = new_color
                    current_violations += delta_v
                    if current_violations == 0:
                        best_n = n
                        best_colors = colors
                        found = True
                        break
                else:
                    if random.random() < math.exp(-delta_v / T):
                        colors[i] = new_color
                        current_violations += delta_v
                        
                T *= cooling
                steps += 1
                
            if found:
                break
                
        if found:
            break
            
    if best_colors is None:
        n = 9
        return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
        
    return {'n': best_n, 'colors': best_colors}
