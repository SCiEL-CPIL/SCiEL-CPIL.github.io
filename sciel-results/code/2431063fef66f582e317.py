import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    for n in range(34, 9, -1):
        if time.time() > deadline:
            break
            
        triples = list(combinations(range(n), 3))
        idx_map = {t: i for i, t in enumerate(triples)}
        num_triples = len(triples)
        
        quads = list(combinations(range(n), 4))
        quints = list(combinations(range(n), 5))
        
        # Map triple index to quad/quint indices
        triple_to_quads = [[] for _ in range(num_triples)]
        for q_idx, q in enumerate(quads):
            for t in combinations(q, 3):
                t_idx = idx_map[t]
                triple_to_quads[t_idx].append(q_idx)
                
        triple_to_quints = [[] for _ in range(num_triples)]
        for q_idx, q in enumerate(quints):
            for t in combinations(q, 3):
                t_idx = idx_map[t]
                triple_to_quints[t_idx].append(q_idx)

        for restart in range(20):
            if time.time() > deadline:
                break
                
            p_red = 0.05 + 0.02 * restart
            colors = np.random.binomial(1, p_red, num_triples)
            
            # Initialize counts
            quad_counts = np.zeros(len(quads), dtype=np.int8)
            quint_counts = np.zeros(len(quints), dtype=np.int8)
            
            # Calculate initial counts
            for i, t in enumerate(triples):
                if colors[i] == 0:
                    for q_idx in triple_to_quads[i]:
                        quad_counts[q_idx] += 1
                else:
                    for q_idx in triple_to_quints[i]:
                        quint_counts[q_idx] += 1
                        
            current_cost = np.sum(quad_counts == 4) + np.sum(quint_counts == 10)
            
            if current_cost == 0:
                return {'n': n, 'colors': colors.tolist()}
                
            best_cost = current_cost
            best_colors = colors.copy()
            
            T = 1.0
            T_min = 0.01
            cooling = 0.99995
            no_improve_steps = 0
            
            restart_deadline = time.time() + (deadline - time.time()) / (20 - restart)
            
            while time.time() < restart_deadline:
                idx = random.randint(0, num_triples - 1)
                old_val = colors[idx]
                
                # Apply flip
                if old_val == 0:
                    for q_idx in triple_to_quads[idx]:
                        quad_counts[q_idx] -= 1
                    for q_idx in triple_to_quints[idx]:
                        quint_counts[q_idx] += 1
                    colors[idx] = 1
                else:
                    for q_idx in triple_to_quads[idx]:
                        quad_counts[q_idx] += 1
                    for q_idx in triple_to_quints[idx]:
                        quint_counts[q_idx] -= 1
                    colors[idx] = 0
                    
                new_cost = np.sum(quad_counts == 4) + np.sum(quint_counts == 10)
                
                delta = new_cost - current_cost
                if delta < 0 or random.random() < np.exp(-delta / T):
                    current_cost = new_cost
                    no_improve_steps = 0
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_colors = colors.copy()
                        if best_cost == 0:
                            return {'n': n, 'colors': best_colors.tolist()}
                else:
                    # Revert
                    if old_val == 0:
                        for q_idx in triple_to_quads[idx]:
                            quad_counts[q_idx] += 1
                        for q_idx in triple_to_quints[idx]:
                            quint_counts[q_idx] -= 1
                        colors[idx] = 0
                    else:
                        for q_idx in triple_to_quads[idx]:
                            quad_counts[q_idx] -= 1
                        for q_idx in triple_to_quints[idx]:
                            quint_counts[q_idx] += 1
                        colors[idx] = 1
                    no_improve_steps += 1
                
                T *= cooling
                if T < T_min:
                    T = T_min
                    if no_improve_steps > 500:
                        T = 1.0
                        no_improve_steps = 0
                        if random.random() < 0.1:
                            # Perturb
                            for _ in range(5):
                                r_idx = random.randint(0, num_triples - 1)
                                # Flip r_idx
                                if colors[r_idx] == 0:
                                    for q_idx in triple_to_quads[r_idx]:
                                        quad_counts[q_idx] -= 1
                                    for q_idx in triple_to_quints[r_idx]:
                                        quint_counts[q_idx] += 1
                                    colors[r_idx] = 1
                                else:
                                    for q_idx in triple_to_quads[r_idx]:
                                        quad_counts[q_idx] += 1
                                    for q_idx in triple_to_quints[r_idx]:
                                        quint_counts[q_idx] -= 1
                                    colors[r_idx] = 0
                            current_cost = np.sum(quad_counts == 4) + np.sum(quint_counts == 10)
                            
            if best_cost == 0:
                return {'n': n, 'colors': best_colors.tolist()}
                
    # Fallback
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
