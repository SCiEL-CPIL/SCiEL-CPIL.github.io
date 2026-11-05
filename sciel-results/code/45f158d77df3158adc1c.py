import time
import random
import numpy as np

def construct():
    start_time = time.time()
    time_limit = 145.0
    
    target_n = 34
    
    def try_solve(n, time_budget):
        triples = []
        triple_indices = {}
        idx = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    triples.append((a, b, c))
                    triple_indices[(a, b, c)] = idx
                    idx += 1
        m = len(triples)
        
        four_sets = []
        four_set_indices = {}
        idx4 = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    for d in range(c+1, n):
                        four_sets.append((a, b, c, d))
                        four_set_indices[(a, b, c, d)] = idx4
                        idx4 += 1
        num_4sets = len(four_sets)
        
        five_sets = []
        five_set_indices = {}
        idx5 = 0
        for a in range(n):
            for b in range(a+1, n):
                for c in range(b+1, n):
                    for d in range(c+1, n):
                        for e in range(d+1, n):
                            five_sets.append((a, b, c, d, e))
                            five_set_indices[(a, b, c, d, e)] = idx5
                            idx5 += 1
        num_5sets = len(five_sets)
        
        triple_to_4sets = [[] for _ in range(m)]
        triple_to_5sets = [[] for _ in range(m)]
        
        for s4 in four_sets:
            idx4 = four_set_indices[s4]
            a, b, c, d = s4
            triples_in_4set = [(a,b,c), (a,b,d), (a,c,d), (b,c,d)]
            for t in triples_in_4set:
                triple_to_4sets[triple_indices[t]].append(idx4)
                
        for s5 in five_sets:
            idx5 = five_set_indices[s5]
            a, b, c, d, e = s5
            triples_in_5set = [
                (a,b,c), (a,b,d), (a,b,e),
                (a,c,d), (a,c,e),
                (a,d,e),
                (b,c,d), (b,c,e),
                (b,d,e),
                (c,d,e)
            ]
            for t in triples_in_5set:
                triple_to_5sets[triple_indices[t]].append(idx5)

        best_energy = float('inf')
        best_colors = None
        
        # Multiple restarts
        while time.time() - start_time < time_limit:
            colors = np.random.randint(0, 2, size=m)
            
            red_count_in_4set = np.zeros(num_4sets, dtype=np.int32)
            blue_count_in_5set = np.zeros(num_5sets, dtype=np.int32)
            
            for t_idx, t in enumerate(triples):
                if colors[t_idx] == 0:
                    for s4_idx in triple_to_4sets[t_idx]:
                        red_count_in_4set[s4_idx] += 1
                else:
                    for s5_idx in triple_to_5sets[t_idx]:
                        blue_count_in_5set[s5_idx] += 1
            
            energy = np.sum(red_count_in_4set == 4) + np.sum(blue_count_in_5set == 10)
            
            if energy < best_energy:
                best_energy = energy
                best_colors = colors.copy()
                if energy == 0:
                    return n, best_colors.tolist()
            
            T = 1.0
            cooling_rate = 0.999
            min_T = 0.01
            steps_in_run = 10000
            no_improve_steps = 0
            run_best_energy = energy
            
            for step in range(steps_in_run):
                if time.time() - start_time > time_limit:
                    break
                
                t_idx = np.random.randint(0, m)
                old_color = colors[t_idx]
                new_color = 1 - old_color
                
                delta_E = 0
                affected_4sets = triple_to_4sets[t_idx]
                affected_5sets = triple_to_5sets[t_idx]
                
                for s4_idx in affected_4sets:
                    old_rc = red_count_in_4set[s4_idx]
                    if old_color == 0:
                        if old_rc == 4:
                            delta_E -= 1
                    else:
                        if old_rc == 3:
                            delta_E += 1
                            
                for s5_idx in affected_5sets:
                    old_bc = blue_count_in_5set[s5_idx]
                    if old_color == 1:
                        if old_bc == 10:
                            delta_E -= 1
                    else:
                        if old_bc == 9:
                            delta_E += 1
                
                if delta_E < 0:
                    accept = True
                else:
                    if T > 0:
                        # Avoid underflow
                        if delta_E > 100 * T:
                            accept = False
                        else:
                            accept = random.random() < np.exp(-delta_E / T)
                    else:
                        accept = False
                
                if accept:
                    colors[t_idx] = new_color
                    for s4_idx in affected_4sets:
                        if old_color == 0:
                            red_count_in_4set[s4_idx] -= 1
                        else:
                            red_count_in_4set[s4_idx] += 1
                    for s5_idx in affected_5sets:
                        if old_color == 1:
                            blue_count_in_5set[s5_idx] -= 1
                        else:
                            blue_count_in_5set[s5_idx] += 1
                    energy += delta_E
                    
                    if energy < run_best_energy:
                        run_best_energy = energy
                        no_improve_steps = 0
                        if energy < best_energy:
                            best_energy = energy
                            best_colors = colors.copy()
                            if energy == 0:
                                return n, best_colors.tolist()
                    else:
                        no_improve_steps += 1
                        
                    if no_improve_steps > 500:
                        break
                else:
                    no_improve_steps += 1
                    if no_improve_steps > 500:
                        break
                
                T *= cooling_rate
                if T < min_T:
                    T = min_T
        
        if best_energy == 0:
            return n, best_colors.tolist()
        else:
            return 0, []

    n_sol, colors_sol = try_solve(target_n, time_limit)
    
    if colors_sol:
        return {'n': n_sol, 'colors': colors_sol}
    else:
        from itertools import combinations
        n_seed = 9
        return {'n': n_seed, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n_seed), 3)]}
