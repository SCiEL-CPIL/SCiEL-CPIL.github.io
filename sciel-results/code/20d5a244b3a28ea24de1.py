import time
import random
import math
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s budget
    
    # Helper to run local search for a given n
    def solve_n(n, time_limit):
        t_start = time.time()
        
        # Precomputation
        triples = []
        triple_index = {}
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    idx = len(triples)
                    triples.append((a,b,c))
                    triple_index[(a,b,c)] = idx
        
        M = len(triples)
        
        four_sets = []
        four_set_indices = {}
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    for d in range(c+1, n):
                        idx = len(four_sets)
                        four_sets.append((a,b,c,d))
                        four_set_indices[(a,b,c,d)] = idx
        
        triple_to_4sets = [[] for _ in range(M)]
        for idx, (a,b,c,d) in enumerate(four_sets):
            t1 = triple_index[(a,b,c)]
            t2 = triple_index[(a,b,d)]
            t3 = triple_index[(a,c,d)]
            t4 = triple_index[(b,c,d)]
            triple_to_4sets[t1].append(idx)
            triple_to_4sets[t2].append(idx)
            triple_to_4sets[t3].append(idx)
            triple_to_4sets[t4].append(idx)
            
        five_sets = []
        five_set_indices = {}
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    for d in range(c+1, n):
                        for e in range(d+1, n):
                            idx = len(five_sets)
                            five_sets.append((a,b,c,d,e))
                            five_set_indices[(a,b,c,d,e)] = idx
                            
        triple_to_5sets = [[] for _ in range(M)]
        for idx, (a,b,c,d,e) in enumerate(five_sets):
            t1 = triple_index[(a,b,c)]
            t2 = triple_index[(a,b,d)]
            t3 = triple_index[(a,b,e)]
            t4 = triple_index[(a,c,d)]
            t5 = triple_index[(a,c,e)]
            t6 = triple_index[(a,d,e)]
            t7 = triple_index[(b,c,d)]
            t8 = triple_index[(b,c,e)]
            t9 = triple_index[(b,d,e)]
            t10 = triple_index[(c,d,e)]
            
            triple_to_5sets[t1].append(idx)
            triple_to_5sets[t2].append(idx)
            triple_to_5sets[t3].append(idx)
            triple_to_5sets[t4].append(idx)
            triple_to_5sets[t5].append(idx)
            triple_to_5sets[t6].append(idx)
            triple_to_5sets[t7].append(idx)
            triple_to_5sets[t8].append(idx)
            triple_to_5sets[t9].append(idx)
            triple_to_5sets[t10].append(idx)

        # Local Search
        # Multiple restarts if time permits
        while time.time() < t_start + time_limit:
            colors = np.random.randint(0, 2, M)
            
            red_count_4 = np.zeros(len(four_sets), dtype=np.int32)
            blue_count_5 = np.zeros(len(five_sets), dtype=np.int32)
            
            # Init counts
            for idx, (a,b,c,d) in enumerate(four_sets):
                t1 = triple_index[(a,b,c)]
                t2 = triple_index[(a,b,d)]
                t3 = triple_index[(a,c,d)]
                t4 = triple_index[(b,c,d)]
                red_count_4[idx] = (colors[t1]==0) + (colors[t2]==0) + (colors[t3]==0) + (colors[t4]==0)
                
            for idx, (a,b,c,d,e) in enumerate(five_sets):
                t1 = triple_index[(a,b,c)]
                t2 = triple_index[(a,b,d)]
                t3 = triple_index[(a,b,e)]
                t4 = triple_index[(a,c,d)]
                t5 = triple_index[(a,c,e)]
                t6 = triple_index[(a,d,e)]
                t7 = triple_index[(b,c,d)]
                t8 = triple_index[(b,c,e)]
                t9 = triple_index[(b,d,e)]
                t10 = triple_index[(c,d,e)]
                blue_count_5[idx] = (colors[t1]==1) + (colors[t2]==1) + (colors[t3]==1) + (colors[t4]==1) + (colors[t5]==1) + (colors[t6]==1) + (colors[t7]==1) + (colors[t8]==1) + (colors[t9]==1) + (colors[t10]==1)

            total_bad_red = np.sum(red_count_4 == 4)
            total_bad_blue = np.sum(blue_count_5 == 10)
            total_energy = total_bad_red + total_bad_blue
            
            if total_energy == 0:
                return {'n': n, 'colors': colors.tolist()}
                
            T = 1.0
            min_T = 0.01
            cooling = 0.999
            steps = 0
            max_steps = 50000 # Adjust based on speed
            
            while steps < max_steps and time.time() < t_start + time_limit:
                t_idx = np.random.randint(M)
                new_color = 1 - colors[t_idx]
                
                delta_red = 0
                delta_blue = 0
                
                # Calculate Delta
                # Red 4-sets
                for s_idx in triple_to_4sets[t_idx]:
                    old_count = red_count_4[s_idx]
                    if colors[t_idx] == 0:
                        new_count = old_count - 1
                    else:
                        new_count = old_count + 1
                        
                    if old_count == 4 and new_count != 4:
                        delta_red -= 1
                    elif old_count != 4 and new_count == 4:
                        delta_red += 1
                        
                # Blue 5-sets
                for s_idx in triple_to_5sets[t_idx]:
                    old_count = blue_count_5[s_idx]
                    if colors[t_idx] == 1:
                        new_count = old_count - 1
                    else:
                        new_count = old_count + 1
                        
                    if old_count == 10 and new_count != 10:
                        delta_blue -= 1
                    elif old_count != 10 and new_count == 10:
                        delta_blue += 1
                        
                delta_total = delta_red + delta_blue
                
                accept = False
                if delta_total < 0:
                    accept = True
                elif delta_total == 0:
                    accept = True
                else:
                    if T > min_T:
                        # Use math.exp for speed
                        prob = math.exp(-delta_total / T)
                        if random.random() < prob:
                            accept = True
                            
                if accept:
                    colors[t_idx] = new_color
                    
                    # Update counts
                    for s_idx in triple_to_4sets[t_idx]:
                        if new_color == 0:
                            red_count_4[s_idx] += 1
                        else:
                            red_count_4[s_idx] -= 1
                            
                    for s_idx in triple_to_5sets[t_idx]:
                        if new_color == 1:
                            blue_count_5[s_idx] += 1
                        else:
                            blue_count_5[s_idx] -= 1
                            
                    total_bad_red += delta_red
                    total_bad_blue += delta_blue
                    total_energy += delta_total
                    
                    if total_energy == 0:
                        return {'n': n, 'colors': colors.tolist()}
                        
                T *= cooling
                steps += 1

        return None

    # Try n from 34 down
    # Allocate time slices
    time_slices = {
        34: 40,
        33: 30,
        32: 20,
        31: 10,
        30: 5
    }
    
    for n in range(34, 29, -1):
        t_limit = time_slices.get(n, 5)
        if time.time() > deadline - 5:
            break
        res = solve_n(n, t_limit)
        if res:
            return res
            
    # Fallback
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
