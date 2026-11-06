import time
import random
import math
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    def build_indices(n):
        triples = list(combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        sets_4 = list(combinations(range(n), 4))
        sets_5 = list(combinations(range(n), 5))
        
        edge_to_4sets = [[] for _ in range(num_triples)]
        edge_to_5sets = [[] for _ in range(num_triples)]
        
        for i, s4 in enumerate(sets_4):
            t1 = (s4[0], s4[1], s4[2])
            t2 = (s4[0], s4[1], s4[3])
            t3 = (s4[0], s4[2], s4[3])
            t4 = (s4[1], s4[2], s4[3])
            for t in (t1, t2, t3, t4):
                edge_to_4sets[triple_to_idx[t]].append(i)
                
        for i, s5 in enumerate(sets_5):
            for t in combinations(s5, 3):
                edge_to_5sets[triple_to_idx[t]].append(i)
                
        return triples, triple_to_idx, sets_4, sets_5, edge_to_4sets, edge_to_5sets

    def simulate_annealing(n, time_limit):
        triples, triple_to_idx, sets_4, sets_5, edge_to_4sets, edge_to_5sets = build_indices(n)
        num_triples = len(triples)
        num_4sets = len(sets_4)
        num_5sets = len(sets_5)
        
        best_energy = float('inf')
        best_colors = None
        
        # Initializations
        inits = []
        inits.append(np.random.randint(0, 2, num_triples))
        # Add a few modular inits
        for m in [3, 5, 7, 11, 13, 17]:
            if m <= n:
                inits.append(np.array([0 if (a+b+c) % m < m//2 else 1 for a,b,c in triples]))
        
        for init in inits:
            if time.time() > time_limit:
                break
                
            colors = init.copy()
            red_counts_4 = np.zeros(num_4sets, dtype=np.int32)
            blue_counts_5 = np.zeros(num_5sets, dtype=np.int32)
            
            for i, c in enumerate(colors):
                if c == 0:
                    for s4_idx in edge_to_4sets[i]:
                        red_counts_4[s4_idx] += 1
                else:
                    for s5_idx in edge_to_5sets[i]:
                        blue_counts_5[s5_idx] += 1
                        
            current_energy = int(np.sum(red_counts_4 == 4) + np.sum(blue_counts_5 == 10))
            
            if current_energy == 0:
                return colors.tolist()
                
            if current_energy < best_energy:
                best_energy = current_energy
                best_colors = colors.copy()
                
            T = 1.0
            T_min = 0.05
            cooling_rate = 0.9999
            steps_since_improve = 0
            max_steps = 50000 # Hard cap per restart
            
            while time.time() < time_limit - 5 and steps_since_improve < max_steps:
                edge_idx = np.random.randint(num_triples)
                old_color = colors[edge_idx]
                
                delta = 0
                if old_color == 0:
                    for s4_idx in edge_to_4sets[edge_idx]:
                        oc = red_counts_4[s4_idx]
                        red_counts_4[s4_idx] = oc - 1
                        if oc == 4: delta -= 1
                    for s5_idx in edge_to_5sets[edge_idx]:
                        oc = blue_counts_5[s5_idx]
                        blue_counts_5[s5_idx] = oc + 1
                        if oc == 9: delta += 1
                else:
                    for s4_idx in edge_to_4sets[edge_idx]:
                        oc = red_counts_4[s4_idx]
                        red_counts_4[s4_idx] = oc + 1
                        if oc == 3: delta += 1
                    for s5_idx in edge_to_5sets[edge_idx]:
                        oc = blue_counts_5[s5_idx]
                        blue_counts_5[s5_idx] = oc - 1
                        if oc == 10: delta -= 1
                        
                if delta < 0 or (delta > 0 and math.exp(-delta / T) > random.random()):
                    colors[edge_idx] = 1 - old_color
                    current_energy += delta
                    if current_energy == 0:
                        return colors.tolist()
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_colors = colors.copy()
                        steps_since_improve = 0
                    else:
                        steps_since_improve += 1
                else:
                    steps_since_improve += 1
                    
                T *= cooling_rate
                if T < T_min: T = T_min
                
        if best_colors is not None and best_energy == 0:
            return best_colors.tolist()
        return None

    # Try n from 34 down
    # Allocate time based on n
    time_left = deadline - time.time()
    for n in range(34, 9, -1):
        if time_left < 5: break
        
        # Heuristic time allocation
        if n >= 32:
            t_alloc = min(30, time_left - 5)
        elif n >= 25:
            t_alloc = min(15, time_left - 5)
        else:
            t_alloc = min(5, time_left - 5)
            
        if t_alloc <= 0: break
        
        t_start = time.time()
        res = simulate_annealing(n, t_start + t_alloc)
        
        if res is not None:
            return {'n': n, 'colors': res}
            
        time_left = deadline - time.time()

    # Fallback
    n = 9
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]
    return {'n': n, 'colors': colors}
