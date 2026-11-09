"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s kill
    
    # Fallback: a known valid coloring for n=9
    def _fallback():
        n = 9
        return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
    
    # Try n from 34 down to 10
    for n in range(34, 9, -1):
        if time.time() > deadline - 5:
            break
            
        num_triples = n * (n - 1) * (n - 2) // 6
        if num_triples == 0:
            continue
            
        triples = list(combinations(range(n), 3))
        idx_map = {t: i for i, t in enumerate(triples)}
        
        quads = list(combinations(range(n), 4))
        quints = list(combinations(range(n), 5))
        
        # Map triple index to lists of (quad_index, other_triple_indices_in_quad)
        # and (quint_index, other_triple_indices_in_quint)
        triple_to_quads = [[] for _ in range(num_triples)]
        for q_idx, q in enumerate(quads):
            q_triples = list(combinations(q, 3))
            q_indices = [idx_map[t] for t in q_triples]
            for i in range(4):
                others = [j for j in range(4) if j != i]
                triple_to_quads[q_indices[i]].append((q_idx, [q_indices[j] for j in others]))
                
        triple_to_quints = [[] for _ in range(num_triples)]
        for q_idx, q in enumerate(quints):
            q_triples = list(combinations(q, 3))
            q_indices = [idx_map[t] for t in q_triples]
            for i in range(10):
                others = [j for j in range(10) if j != i]
                triple_to_quints[q_indices[i]].append((q_idx, [q_indices[j] for j in others]))

        # We will use a simulated annealing approach with incremental updates
        # State: colors (np array of 0s and 1s)
        # Objective: minimize number of red K4s + blue K5s
        
        best_solution = None
        best_cost = float('inf')
        
        # Number of restarts depends on remaining time
        # For n=34, num_triples=5984. 
        # For n=30, num_triples=4060.
        
        max_restarts = 5
        # If we have plenty of time, do more restarts
        if time.time() < deadline - 60:
            max_restarts = 10
            
        for restart in range(max_restarts):
            if time.time() > deadline - 5:
                break
                
            # Random initialization
            # Bias towards blue (1) to avoid red K4s? 
            # R(4,5;3) means no red K4, no blue K5.
            # Red K4 is a "smaller" structure (4 vertices) than Blue K5 (5 vertices).
            # So it is "easier" to accidentally form a Red K4? 
            # Actually, probability of all 4 red is p^4. Probability of all 10 blue is (1-p)^10.
            # If p=0.5, p^4 = 1/16, (1-p)^10 = 1/1024.
            # So Red K4s are much more likely. We should bias towards Blue (1) to reduce Red K4s.
            # Let's try p_red = 0.2 or 0.3.
            
            p_red = 0.2 + 0.05 * restart
            if p_red > 0.6:
                p_red = 0.6
                
            colors = np.random.binomial(1, p_red, num_triples).astype(np.int8)
            
            # Initialize counts
            # quad_counts[q_idx] = number of red triples in quad q_idx
            # quint_counts[q_idx] = number of blue triples in quint q_idx
            quad_counts = np.zeros(len(quads), dtype=np.int8)
            quint_counts = np.zeros(len(quints), dtype=np.int8)
            
            # Calculate initial counts
            for i in range(num_triples):
                if colors[i] == 0:  # Red
                    for q_idx, others in triple_to_quads[i]:
                        quad_counts[q_idx] += 1
                else:  # Blue
                    for q_idx, others in triple_to_quints[i]:
                        quint_counts[q_idx] += 1
                        
            # Current cost
            current_cost = np.sum(quad_counts == 4) + np.sum(quint_counts == 10)
            
            if current_cost == 0:
                return {'n': n, 'colors': colors.tolist()}
                
            best_cost_in_restart = current_cost
            best_colors_in_restart = colors.copy()
            
            # Simulated Annealing parameters
            T = 1.0
            T_min = 0.01
            cooling = 0.9999
            restart_deadline = time.time() + (deadline - 5 - time.time()) / (max_restarts - restart)
            
            no_improve_steps = 0
            max_no_improve = 1000
            
            while time.time() < restart_deadline:
                idx = random.randint(0, num_triples - 1)
                old_val = colors[idx]
                
                # Apply flip
                if old_val == 0:  # Red -> Blue
                    # Update quad counts: decrement
                    for q_idx, others in triple_to_quads[idx]:
                        quad_counts[q_idx] -= 1
                    # Update quint counts: increment
                    for q_idx, others in triple_to_quints[idx]:
                        quint_counts[q_idx] += 1
                    colors[idx] = 1
                else:  # Blue -> Red
                    # Update quad counts: increment
                    for q_idx, others in triple_to_quads[idx]:
                        quad_counts[q_idx] += 1
                    # Update quint counts: decrement
                    for q_idx, others in triple_to_quints[idx]:
                        quint_counts[q_idx] -= 1
                    colors[idx] = 0
                    
                # Calculate new cost
                # We can do this incrementally too, but np.sum is fast enough for moderate sizes?
                # len(quads) for n=34 is C(34,4) = 46376.
                # len(quints) for n=34 is C(34,5) = 278256.
                # np.sum on array of size 324k is very fast in C.
                new_cost = np.sum(quad_counts == 4) + np.sum(quint_counts == 10)
                
                delta = new_cost - current_cost
                if delta < 0 or random.random() < np.exp(-delta / T):
                    current_cost = new_cost
                    no_improve_steps = 0
                    if current_cost < best_cost_in_restart:
                        best_cost_in_restart = current_cost
                        best_colors_in_restart = colors.copy()
                        if best_cost_in_restart == 0:
                            return {'n': n, 'colors': best_colors_in_restart.tolist()}
                else:
                    # Revert
                    if old_val == 0:
                        for q_idx, others in triple_to_quads[idx]:
                            quad_counts[q_idx] += 1
                        for q_idx, others in triple_to_quints[idx]:
                            quint_counts[q_idx] -= 1
                        colors[idx] = 0
                    else:
                        for q_idx, others in triple_to_quads[idx]:
                            quad_counts[q_idx] -= 1
                        for q_idx, others in triple_to_quints[idx]:
                            quint_counts[q_idx] += 1
                        colors[idx] = 1
                    no_improve_steps += 1
                
                T *= cooling
                if T < T_min:
                    T = T_min
                    if no_improve_steps > max_no_improve:
                        # Restart temperature or perturb
                        T = 1.0
                        no_improve_steps = 0
                        # Perturb
                        for _ in range(10):
                            r_idx = random.randint(0, num_triples - 1)
                            if colors[r_idx] == 0:
                                for q_idx, others in triple_to_quads[r_idx]:
                                    quad_counts[q_idx] -= 1
                                for q_idx, others in triple_to_quints[r_idx]:
                                    quint_counts[q_idx] += 1
                                colors[r_idx] = 1
                            else:
                                for q_idx, others in triple_to_quads[r_idx]:
                                    quad_counts[q_idx] += 1
                                for q_idx, others in triple_to_quints[r_idx]:
                                    quint_counts[q_idx] -= 1
                                colors[r_idx] = 0
                        current_cost = np.sum(quad_counts == 4) + np.sum(quint_counts == 10)
                        
            if best_cost_in_restart < best_cost:
                best_cost = best_cost_in_restart
                best_solution = best_colors_in_restart
                
                if best_cost == 0:
                    return {'n': n, 'colors': best_solution.tolist()}
                    
    # If we didn't find a valid coloring for n > 9, return fallback
    return _fallback()
# EVOLVE-BLOCK-END
