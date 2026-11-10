import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Target n = 34
    n = 34
    
    # Precompute triples
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    # Precompute K4s and K5s
    k4_list = list(combinations(range(n), 4))
    num_k4 = len(k4_list)
    k5_list = list(combinations(range(n), 5))
    num_k5 = len(k5_list)
    
    # Map each triple to the K4s and K5s it belongs to
    triple_to_k4s = [[] for _ in range(num_triples)]
    k4_triples = []  # k4_triples[i] = list of triple indices in K4 i
    for i, k4 in enumerate(k4_list):
        ts = []
        for j in range(4):
            t = tuple(sorted(k4[:j] + k4[j+1:]))
            idx = triple_to_idx[t]
            ts.append(idx)
            triple_to_k4s[idx].append(i)
        k4_triples.append(ts)
        
    triple_to_k5s = [[] for _ in range(num_triples)]
    k5_triples = []  # k5_triples[i] = list of triple indices in K5 i
    for i, k5 in enumerate(k5_list):
        ts = []
        for combo in combinations(range(5), 3):
            t = tuple(sorted([k5[j] for j in combo]))
            idx = triple_to_idx[t]
            ts.append(idx)
            triple_to_k5s[idx].append(i)
        k5_triples.append(ts)

    # We'll use a simulated annealing / walk-sat hybrid approach
    # Key insight: we need to find a coloring with 0 violations
    
    # For efficiency, let's use numpy arrays for counts
    # colors: 0 for Red, 1 for Blue
    # k4_counts[i] = number of red triples in K4 i
    # k5_counts[i] = number of blue triples in K5 i
    
    def init_random():
        colors = np.random.randint(0, 2, size=num_triples).astype(np.int8)
        k4_counts = np.zeros(num_k4, dtype=np.int32)
        k5_counts = np.zeros(num_k5, dtype=np.int32)
        
        for t_idx in range(num_triples):
            val = colors[t_idx]
            if val == 0:  # Red
                for k4_idx in triple_to_k4s[t_idx]:
                    k4_counts[k4_idx] += 1
            else:  # Blue
                for k5_idx in triple_to_k5s[t_idx]:
                    k5_counts[k5_idx] += 1
        
        violated_k4s = set(i for i in range(num_k4) if k4_counts[i] == 4)
        violated_k5s = set(i for i in range(num_k5) if k5_counts[i] == 10)
        current_violations = len(violated_k4s) + len(violated_k5s)
        
        return colors, k4_counts, k5_counts, violated_k4s, violated_k5s, current_violations
    
    # Try a few random initializations first
    best_violations = float('inf')
    best_colors = None
    
    for attempt in range(5):
        if time.time() > deadline - 5.0:
            break
            
        colors, k4_counts, k5_counts, violated_k4s, violated_k5s, current_violations = init_random()
        
        if current_violations < best_violations:
            best_violations = current_violations
            best_colors = colors.copy()
            
        if current_violations == 0:
            return {'n': 34, 'colors': colors.tolist()}
    
    # Now do a more systematic search from the best starting point
    if best_colors is not None:
        colors, k4_counts, k5_counts, violated_k4s, violated_k5s, current_violations = init_random()
        # Actually, let's restart from the best we found
        # Re-initialize from best_colors
        k4_counts = np.zeros(num_k4, dtype=np.int32)
        k5_counts = np.zeros(num_k5, dtype=np.int32)
        for t_idx in range(num_triples):
            val = best_colors[t_idx]
            if val == 0:
                for k4_idx in triple_to_k4s[t_idx]:
                    k4_counts[k4_idx] += 1
            else:
                for k5_idx in triple_to_k5s[t_idx]:
                    k5_counts[k5_idx] += 1
        violated_k4s = set(i for i in range(num_k4) if k4_counts[i] == 4)
        violated_k5s = set(i for i in range(num_k5) if k5_counts[i] == 10)
        current_violations = len(violated_k4s) + len(violated_k5s)
        colors = best_colors.copy()
    
    # WalkSAT with restarts
    max_restarts = 200
    max_steps_per_restart = 50000
    
    for restart in range(max_restarts):
        if time.time() > deadline - 2.0:
            break
            
        if current_violations == 0:
            return {'n': 34, 'colors': colors.tolist()}
        
        if current_violations < best_violations:
            best_violations = current_violations
            best_colors = colors.copy()
        
        # WalkSAT inner loop
        run_start = time.time()
        steps = 0
        max_steps = max_steps_per_restart
        
        while steps < max_steps and time.time() < run_start + 1.0:
            if current_violations == 0:
                return {'n': 34, 'colors': colors.tolist()}
                
            if current_violations < best_violations:
                best_violations = current_violations
                best_colors = colors.copy()
                
            # Pick a random violated clause
            # We need to pick from violated_k4s or violated_k5s
            # Convert to list for random choice (expensive if large, but violations should be manageable)
            all_violated = []
            for k4_idx in violated_k4s:
                all_violated.append(('k4', k4_idx))
            for k5_idx in violated_k5s:
                all_violated.append(('k5', k5_idx))
                
            if not all_violated:
                break
                
            # Pick a random violated clause
            clause_type, clause_idx = random.choice(all_violated)
            
            # Pick a random triple in that clause to flip
            if clause_type == 'k4':
                triple_indices = k4_triples[clause_idx]
            else:
                triple_indices = k5_triples[clause_idx]
                
            idx = random.choice(triple_indices)
            
            # Flip the triple
            old_val = colors[idx]
            new_val = 1 - old_val
            colors[idx] = new_val
            
            delta_red = 1 if new_val == 0 else -1
            delta_blue = 1 if new_val == 1 else -1
            
            # Update K4 counts and violations
            for k4_idx in triple_to_k4s[idx]:
                old_count = k4_counts[k4_idx]
                new_count = old_count + delta_red
                k4_counts[k4_idx] = new_count
                
                was_viol = (old_count == 4)
                is_viol = (new_count == 4)
                
                if was_viol and not is_viol:
                    violated_k4s.discard(k4_idx)
                    current_violations -= 1
                elif not was_viol and is_viol:
                    violated_k4s.add(k4_idx)
                    current_violations += 1
                    
            # Update K5 counts and violations
            for k5_idx in triple_to_k5s[idx]:
                old_count = k5_counts[k5_idx]
                new_count = old_count + delta_blue
                k5_counts[k5_idx] = new_count
                
                was_viol = (old_count == 10)
                is_viol = (new_count == 10)
                
                if was_viol and not is_viol:
                    violated_k5s.discard(k5_idx)
                    current_violations -= 1
                elif not was_viol and is_viol:
                    violated_k5s.add(k5_idx)
                    current_violations += 1
                    
            steps += 1
            
        # End of restart inner loop
        
        # If we haven't found a solution, do a random restart or perturb
        if time.time() > deadline - 5.0:
            break
            
        if best_violations == 0:
            break
            
        # Random restart
        if random.random() < 0.3 or restart % 10 == 0:
            colors, k4_counts, k5_counts, violated_k4s, violated_k5s, current_violations = init_random()
        else:
            # Perturb current solution
            for _ in range(5):
                idx = random.randint(0, num_triples - 1)
                old_val = colors[idx]
                new_val = 1 - old_val
                colors[idx] = new_val
                
                delta_red = 1 if new_val == 0 else -1
                delta_blue = 1 if new_val == 1 else -1
                
                for k4_idx in triple_to_k4s[idx]:
                    old_count = k4_counts[k4_idx]
                    new_count = old_count + delta_red
                    k4_counts[k4_idx] = new_count
                    
                    was_viol = (old_count == 4)
                    is_viol = (new_count == 4)
                    
                    if was_viol and not is_viol:
                        violated_k4s.discard(k4_idx)
                        current_violations -= 1
                    elif not was_viol and is_viol:
                        violated_k4s.add(k4_idx)
                        current_violations += 1
                        
                for k5_idx in triple_to_k5s[idx]:
                    old_count = k5_counts[k5_idx]
                    new_count = old_count + delta_blue
                    k5_counts[k5_idx] = new_count
                    
                    was_viol = (old_count == 10)
                    is_viol = (new_count == 10)
                    
                    if was_viol and not is_viol:
                        violated_k5s.discard(k5_idx)
                        current_violations -= 1
                    elif not was_viol and is_viol:
                        violated_k5s.add(k5_idx)
                        current_violations += 1

    if best_violations == 0:
        return {'n': 34, 'colors': best_colors.tolist()}
    else:
        # Fallback to smaller n if we failed
        # Try n=9 which is easier
        n_fallback = 9
        colors_fallback = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n_fallback), 3)]
        return {'n': n_fallback, 'colors': colors_fallback}
