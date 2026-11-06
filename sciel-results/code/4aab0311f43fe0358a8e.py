import itertools
import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    def solve_for_n(n, deadline):
        triples = list(itertools.combinations(range(n), 3))
        num_triples = len(triples)
        triple_index = {t: i for i, t in enumerate(triples)}
        
        # Precompute K4s and K5s
        k4s = []
        for quad in itertools.combinations(range(n), 4):
            idxs = [triple_index[t] for t in itertools.combinations(quad, 3)]
            k4s.append(idxs)
            
        k5s = []
        for quint in itertools.combinations(range(n), 5):
            idxs = [triple_index[t] for t in itertools.combinations(quint, 3)]
            k5s.append(idxs)
            
        num_k4s = len(k4s)
        num_k5s = len(k5s)
        
        # Precompute which K4s and K5s each triple belongs to
        triple_to_k4s = [[] for _ in range(num_triples)]
        for i, idxs in enumerate(k4s):
            for idx in idxs:
                triple_to_k4s[idx].append(i)
                
        triple_to_k5s = [[] for _ in range(num_triples)]
        for i, idxs in enumerate(k5s):
            for idx in idxs:
                triple_to_k5s[idx].append(i)
                
        def init_energy(colors):
            red_counts = [0] * num_k4s
            blue_counts = [0] * num_k5s
            
            for i, c in enumerate(colors):
                if c == 0: # Red
                    for k4_idx in triple_to_k4s[i]:
                        red_counts[k4_idx] += 1
                else: # Blue
                    for k5_idx in triple_to_k5s[i]:
                        blue_counts[k5_idx] += 1
            
            energy = sum(1 for c in red_counts if c == 4) + sum(1 for c in blue_counts if c == 10)
            return energy, red_counts, blue_counts
            
        best_energy = float('inf')
        best_colors = None
        
        max_restarts = 50
        rng = random.Random(42)
        
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            colors = [rng.randint(0, 1) for _ in range(num_triples)]
            energy, red_counts, blue_counts = init_energy(colors)
            
            if energy == 0:
                return True, colors
                
            if energy < best_energy:
                best_energy = energy
                best_colors = colors[:]
                
            T = 1.0
            cooling = 0.99
            min_T = 0.01
            max_steps = 2000
            no_improvement = 0
            max_no_improvement = 100
            
            for step in range(max_steps):
                if time.time() > deadline - 5:
                    break
                    
                t_idx = rng.randint(0, num_triples - 1)
                old_color = colors[t_idx]
                new_color = 1 - old_color
                
                delta_energy = 0
                
                for k4_idx in triple_to_k4s[t_idx]:
                    if old_color == 0:
                        red_counts[k4_idx] -= 1
                        if red_counts[k4_idx] == 3:
                            delta_energy -= 1
                    else:
                        red_counts[k4_idx] += 1
                        if red_counts[k4_idx] == 4:
                            delta_energy += 1

                for k5_idx in triple_to_k5s[t_idx]:
                    if old_color == 1:
                        blue_counts[k5_idx] -= 1
                        if blue_counts[k5_idx] == 9:
                            delta_energy -= 1
                    else:
                        blue_counts[k5_idx] += 1
                        if blue_counts[k5_idx] == 10:
                            delta_energy += 1

                if delta_energy < 0:
                    colors[t_idx] = new_color
                    energy += delta_energy
                    no_improvement = 0
                    if energy == 0:
                        return True, colors
                else:
                    if T > 0:
                        # Avoid overflow in exp
                        if delta_energy / T < 700: 
                            prob = math.exp(-delta_energy / T)
                        else:
                            prob = 0
                        if rng.random() < prob:
                            colors[t_idx] = new_color
                            energy += delta_energy
                    no_improvement += 1
                    
                T *= cooling
                if T < min_T:
                    T = min_T
                    
                if no_improvement > max_no_improvement:
                    break
                    
        return False, best_colors

    for n in [34, 33, 32, 31, 30, 25, 20, 15, 10]:
        if time.time() > deadline - 10:
            break
            
        success, colors = solve_for_n(n, deadline)
        if success:
            return {'n': n, 'colors': colors}
            
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in itertools.combinations(range(n), 3)]}
