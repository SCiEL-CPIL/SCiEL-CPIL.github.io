import numpy as np
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # We try to find n=35 first, then fall back to n=34
    # The seed achieves n=34. We aim for n=35 to get score > 1.0
    
    for n in [35, 34, 33, 32, 31, 30]:
        if time.time() > deadline - 10:
            break
            
        triples = list(combinations(range(n), 3))
        idx_map = {t: i for i, t in enumerate(triples)}
        num_triples = len(triples)
        
        quads = list(combinations(range(n), 4))
        quints = list(combinations(range(n), 5))
        
        # Precompute mappings
        triple_to_quads = [[] for _ in range(num_triples)]
        triple_to_quints = [[] for _ in range(num_triples)]
        
        for q_idx, q in enumerate(quads):
            for t in combinations(q, 3):
                t_idx = idx_map[t]
                triple_to_quads[t_idx].append(q_idx)
                
        for f_idx, f in enumerate(quints):
            for t in combinations(f, 3):
                t_idx = idx_map[t]
                triple_to_quints[t_idx].append(f_idx)

        # WalkSAT-style Local Search
        # Cost: Number of violated 4-sets (all red) + Number of violated 5-sets (all blue)
        
        best_cost = float('inf')
        best_colors = None
        
        for restart in range(100):
            if time.time() > deadline - 5:
                break
                
            # Random initial assignment, biased towards Blue for 5-sets, Red for 4-sets?
            # Actually, just random is fine, WalkSAT will fix it.
            colors = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
            
            # Initialize counts
            quad_counts = np.zeros(len(quads), dtype=np.int8) # Count of RED triples in quad
            quint_counts = np.zeros(len(quints), dtype=np.int8) # Count of BLUE triples in quint
            
            for i in range(num_triples):
                if colors[i] == 0: # Red
                    for q_idx in triple_to_quads[i]:
                        quad_counts[q_idx] += 1
                else: # Blue
                    for f_idx in triple_to_quints[i]:
                        quint_counts[f_idx] += 1
                        
            # Current violations
            viol_quads = np.sum(quad_counts == 4)
            viol_quints = np.sum(quint_counts == 10)
            current_cost = viol_quads + viol_quints
            
            if current_cost == 0:
                return {'n': n, 'colors': colors.tolist()}
                
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors.copy()
                
            T = 1.0
            T_min = 0.01
            cooling = 0.9999
            steps = 0
            max_steps = 50000
            
            while steps < max_steps and time.time() < deadline - 5:
                # Pick a random violated clause
                # If no violations, we are done (handled above)
                
                # To make it faster, we can pick a random triple and check if it's in a violation
                # But WalkSAT picks a violated clause.
                
                # Let's pick a random violated 4-set or 5-set
                # We need to know which ones are violated.
                # Maintaining a list of violated sets is expensive.
                # Instead, we pick a random triple, and if it's in a violated set, we flip it.
                # This is less efficient but simpler.
                
                # Better: Pick a random triple.
                idx = random.randint(0, num_triples - 1)
                
                # Calculate delta cost
                old_val = colors[idx]
                
                # Simulate flip
                if old_val == 0: # Red -> Blue
                    # quad_counts decreases for q in triple_to_quads[idx]
                    # quint_counts increases for f in triple_to_quints[idx]
                    
                    # Check if any quad goes from 4 to 3 (cost -1) or 3 to 2 (cost 0)
                    # Check if any quint goes from 10 to 11 (cost 0) or 9 to 10 (cost +1)
                    
                    delta = 0
                    for q_idx in triple_to_quads[idx]:
                        c = quad_counts[q_idx]
                        if c == 4:
                            delta -= 1 # Fixed a violation
                        elif c == 3:
                            delta += 0 # 3->2, no change in violation count
                        # c > 4 is impossible, c < 3 no change
                            
                    for f_idx in triple_to_quints[idx]:
                        c = quint_counts[f_idx]
                        if c == 10:
                            delta += 0 # 10->11, no change
                        elif c == 9:
                            delta += 1 # Created a violation
                            
                else: # Blue -> Red
                    for q_idx in triple_to_quads[idx]:
                        c = quad_counts[q_idx]
                        if c == 4:
                            delta += 0 # 4->5, no change (but 5 is not a violation, 4 was)
                            # Wait, if c==4, it was a violation. After flip, c becomes 3? No.
                            # If old_val is Blue, it is NOT counted in quad_counts.
                            # So quad_counts[q] is the count of RED triples.
                            # If we flip Blue->Red, quad_counts[q] increases by 1.
                            # If c was 3, it becomes 4 (violation created). Delta +1.
                            # If c was 4, it becomes 5 (violation remains? No, 4 is violation, 5 is not).
                            # Wait, the violation is "ALL 4 are Red".
                            # So if count == 4, it is a violation.
                            # If count becomes 5, it's impossible (max 4).
                            # So if c==3, delta += 1.
                            # If c==4, it was a violation. After flip, c becomes... wait.
                            # If old_val is Blue, it is not in the red count.
                            # So red count is 3. After flip, red count is 4. Violation created.
                            # If red count was 4, it means all 4 were red. But old_val is Blue. Contradiction.
                            # So c cannot be 4 if old_val is Blue.
                            # So:
                            if c == 3:
                                delta += 1
                                
                    for f_idx in triple_to_quints[idx]:
                        # Blue -> Red
                        # quint_counts is count of BLUE triples.
                        # If c==10, it was a violation. After flip, c becomes 9. Violation fixed. Delta -1.
                        # If c==9, it becomes 10. Violation created. Delta +1.
                        if c == 10:
                            delta -= 1
                        elif c == 9:
                            delta += 1

                if delta < 0 or random.random() < np.exp(-delta / T):
                    # Apply flip
                    if old_val == 0:
                        for q_idx in triple_to_quads[idx]:
                            quad_counts[q_idx] -= 1
                        for f_idx in triple_to_quints[idx]:
                            quint_counts[f_idx] += 1
                        colors[idx] = 1
                    else:
                        for q_idx in triple_to_quads[idx]:
                            quad_counts[q_idx] += 1
                        for f_idx in triple_to_quints[idx]:
                            quint_counts[f_idx] -= 1
                        colors[idx] = 0
                        
                    current_cost += delta
                    
                    if current_cost == 0:
                        return {'n': n, 'colors': colors.tolist()}
                        
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_colors = colors.copy()
                        
                T *= cooling
                if T < T_min:
                    T = T_min
                steps += 1
                
            # End while
            
        # End restarts
        
        if best_colors is not None:
            return {'n': n, 'colors': best_colors.tolist()}
            
    # Fallback
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}
