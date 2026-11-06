import numpy as np
import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Target n values
    # R(4,5;3) is known to be 35, so 34 is the max valid n.
    # We try 34 down to 20.
    target_ns = [34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20]
    
    for n in target_ns:
        if time.time() > deadline - 5.0:
            break
            
        # Precompute structures
        triples = list(itertools.combinations(range(n), 3))
        M = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Map each triple to the list of 4-sets it belongs to
        # and the list of 5-sets it belongs to
        # We need indices for 4-sets and 5-sets to track counts
        quads = list(itertools.combinations(range(n), 4))
        N4 = len(quads)
        quad_to_idx = {q: i for i, q in enumerate(quads)}
        
        quints = list(itertools.combinations(range(n), 5))
        N5 = len(quints)
        quint_to_idx = {q: i for i, q in enumerate(quints)}
        
        triple_to_quads = [[] for _ in range(M)]
        triple_to_quints = [[] for _ in range(M)]
        
        for q in quads:
            q_idx = quad_to_idx[q]
            for i in range(4):
                # Get the triple formed by removing the i-th vertex
                # q is a tuple of 4 vertices
                # The triples in q are combinations of 3
                # It's easier to iterate combinations of q
                pass
                
        # Better way to build mapping
        # For each quad, get its 4 triples
        # For each quint, get its 10 triples
        
        # Re-iterate to build maps correctly
        for q in quads:
            q_idx = quad_to_idx[q]
            # Get 3-subsets of q
            for t in itertools.combinations(q, 3):
                t_idx = triple_to_idx[t]
                triple_to_quads[t_idx].append(q_idx)
                
        for q in quints:
            q_idx = quint_to_idx[q]
            for t in itertools.combinations(q, 3):
                t_idx = triple_to_idx[t]
                triple_to_quints[t_idx].append(q_idx)

        def solve_n(n, time_limit):
            M = len(triples)
            
            # Local Search / Greedy Repair
            # We maintain:
            # C: array of colors (0=red, 1=blue)
            # red_counts: array of size N4, count of red triples in each quad
            # blue_counts: array of size N5, count of blue triples in each quint
            # total_conflicts: number of quads with red_count==4 + number of quints with blue_count==10
            
            max_restarts = 200
            max_steps_per_restart = 5000
            
            for restart in range(max_restarts):
                if time.time() > time_limit:
                    return None
                
                # Random initialization
                C = np.random.randint(0, 2, M).astype(np.int8)
                
                # Initialize counts
                red_counts = np.zeros(N4, dtype=np.int32)
                blue_counts = np.zeros(N5, dtype=np.int32)
                
                # Calculate initial counts
                # For each triple, if red, increment red_counts for its quads
                # If blue, increment blue_counts for its quints
                for t_idx in range(M):
                    if C[t_idx] == 0: # Red
                        for q_idx in triple_to_quads[t_idx]:
                            red_counts[q_idx] += 1
                    else: # Blue
                        for q_idx in triple_to_quints[t_idx]:
                            blue_counts[q_idx] += 1
                
                # Calculate initial conflicts
                # Conflicts are quads where red_count == 4 and quints where blue_count == 10
                # We can track this with a running sum, but initially we just count.
                # To update efficiently, we need to know if a count transitioned to/from a conflict state.
                
                total_conflicts = np.sum(red_counts == 4) + np.sum(blue_counts == 10)
                
                if total_conflicts == 0:
                    return C.tolist()
                
                steps = 0
                stuck_counter = 0
                last_conflict = total_conflicts
                
                while steps < max_steps_per_restart:
                    if time.time() > time_limit:
                        return None
                    
                    # Pick a random triple to flip
                    # To be more effective, we could pick a triple involved in a conflict.
                    # Let's pick a random triple first. If that fails, we can refine.
                    t_idx = np.random.randint(M)
                    
                    old_val = C[t_idx]
                    new_val = 1 - old_val
                    
                    # Calculate delta in conflicts
                    delta_conflicts = 0
                    
                    # Update red_counts for quads containing t_idx
                    if old_val == 0: # Was Red, becoming Blue
                        # red_counts decreases by 1
                        for q_idx in triple_to_quads[t_idx]:
                            old_count = red_counts[q_idx]
                            new_count = old_count - 1
                            red_counts[q_idx] = new_count
                            if old_count == 4:
                                delta_conflicts -= 1
                            elif new_count == 4: # Impossible if decreasing from 3? No, 3->2.
                                delta_conflicts += 1
                    else: # Was Blue, becoming Red
                        # red_counts increases by 1
                        for q_idx in triple_to_quads[t_idx]:
                            old_count = red_counts[q_idx]
                            new_count = old_count + 1
                            red_counts[q_idx] = new_count
                            if new_count == 4:
                                delta_conflicts += 1
                            elif old_count == 4: # Impossible if increasing from 3? No, 3->4.
                                delta_conflicts -= 1
                            
                    # Update blue_counts for quints containing t_idx
                    if old_val == 1: # Was Blue, becoming Red
                        # blue_counts decreases by 1
                        for q_idx in triple_to_quints[t_idx]:
                            old_count = blue_counts[q_idx]
                            new_count = old_count - 1
                            blue_counts[q_idx] = new_count
                            if old_count == 10:
                                delta_conflicts -= 1
                            elif new_count == 10: # Impossible if decreasing from 9? No, 9->8.
                                delta_conflicts += 1
                    else: # Was Red, becoming Blue
                        # blue_counts increases by 1
                        for q_idx in triple_to_quints[t_idx]:
                            old_count = blue_counts[q_idx]
                            new_count = old_count + 1
                            blue_counts[q_idx] = new_count
                            if new_count == 10:
                                delta_conflicts += 1
                            elif old_count == 10: # Impossible if increasing from 9? No, 9->10.
                                delta_conflicts -= 1
                                
                    total_conflicts += delta_conflicts
                    
                    # Acceptance: 
                    # If delta < 0, always accept.
                    # If delta == 0, accept with prob 0.5 (to explore)
                    # If delta > 0, accept with small prob (annealing)
                    
                    # Simple Hill Climbing with Random Restarts on Stagnation
                    accept = False
                    if delta_conflicts < 0:
                        accept = True
                    elif delta_conflicts == 0:
                        accept = np.random.rand() < 0.5
                    else:
                        # Only accept worsening moves if we are stuck?
                        # Or use a simple annealing schedule
                        # For now, reject worsening moves to keep it simple and fast.
                        # This is pure hill climbing.
                        accept = False
                        
                    if accept:
                        C[t_idx] = new_val
                        stuck_counter = 0
                    else:
                        # Roll back
                        C[t_idx] = old_val
                        # Roll back counts
                        if old_val == 0:
                            for q_idx in triple_to_quads[t_idx]:
                                red_counts[q_idx] += 1
                            for q_idx in triple_to_quints[t_idx]:
                                blue_counts[q_idx] -= 1
                        else:
                            for q_idx in triple_to_quads[t_idx]:
                                red_counts[q_idx] -= 1
                            for q_idx in triple_to_quints[t_idx]:
                                blue_counts[q_idx] += 1
                        total_conflicts -= delta_conflicts
                        stuck_counter += 1
                        
                    if total_conflicts == 0:
                        return C.tolist()
                        
                    if stuck_counter > 100:
                        break # Restart
                    
                    steps += 1
                
            return None

        # Allocate time for this n
        # If n=34, give it most of the time.
        # If n=33, give it less.
        # Let's give 100s for 34, 10s for 33, etc?
        # Or just loop until deadline.
        
        res = solve_n(n, deadline - 2.0)
        if res is not None:
            return {'n': n, 'colors': res}

    # Fallback: return a small valid coloring
    # n=9 is easy.
    n = 9
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in itertools.combinations(range(n), 3)]
    return {'n': n, 'colors': colors}
