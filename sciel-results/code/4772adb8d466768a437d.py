import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # We aim for n=34.
    # If we fail, we might try n=33, 32... but n=34 is the target.
    # Let's focus on n=34.
    
    n = 34
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    # Precompute K4s and K5s
    k4_list = list(combinations(range(n), 4))
    k5_list = list(combinations(range(n), 5))
    
    # Map triple index to list of K4 indices and K5 indices it belongs to
    triple_to_k4s = [[] for _ in range(num_triples)]
    triple_to_k5s = [[] for _ in range(num_triples)]
    
    # k4_data[i] = list of 4 triple indices
    # k5_data[i] = list of 10 triple indices
    k4_data = []
    k5_data = []
    
    for i, k4 in enumerate(k4_list):
        ts = []
        for j in range(4):
            t = tuple(sorted(k4[:j] + k4[j+1:]))
            idx = triple_to_idx[t]
            ts.append(idx)
            triple_to_k4s[idx].append(i)
        k4_data.append(ts)
        
    for i, k5 in enumerate(k5_list):
        ts = []
        for combo in combinations(range(5), 3):
            t = tuple(sorted([k5[j] for j in combo]))
            idx = triple_to_idx[t]
            ts.append(idx)
            triple_to_k5s[idx].append(i)
        k5_data.append(ts)

    # Convert to numpy arrays for potential vectorization, though we will use incremental updates
    # We need fast access to the list of triples in a clause.
    # k4_data and k5_data are lists of lists of ints.
    
    def get_initial_state():
        colors = np.random.randint(0, 2, size=num_triples).astype(np.int8)
        # Compute counts
        k4_counts = np.zeros(len(k4_list), dtype=np.int32)
        k5_counts = np.zeros(len(k5_list), dtype=np.int32)
        
        for t_idx, c in enumerate(colors):
            if c == 0: # Red
                for k4_idx in triple_to_k4s[t_idx]:
                    k4_counts[k4_idx] += 1
            else: # Blue
                for k5_idx in triple_to_k5s[t_idx]:
                    k5_counts[k5_idx] += 1
        return colors, k4_counts, k5_counts

    def count_violations(k4_counts, k5_counts):
        return np.sum(k4_counts == 4) + np.sum(k5_counts == 10)

    best_n = 0
    best_colors = None
    
    # Try to find solution for n=34
    # We will run multiple restarts of Min-Conflicts
    
    max_restarts = 5000
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        colors, k4_counts, k5_counts = get_initial_state()
        current_violations = count_violations(k4_counts, k5_counts)
        
        if current_violations == 0:
            return {'n': n, 'colors': colors.tolist()}
            
        # Min-Conflicts Loop
        # We maintain a set of violated K4s and K5s for fast random selection
        violated_k4s = set(np.where(k4_counts == 4)[0])
        violated_k5s = set(np.where(k5_counts == 10)[0])
        
        steps = 0
        max_steps = 50000
        stuck_counter = 0
        best_local_violations = current_violations
        best_local_colors = colors.copy()
        
        while steps < max_steps:
            if time.time() > deadline:
                break
                
            if current_violations == 0:
                return {'n': n, 'colors': colors.tolist()}
                
            if current_violations < best_local_violations:
                best_local_violations = current_violations
                best_local_colors = colors.copy()
                stuck_counter = 0
            else:
                stuck_counter += 1
                if stuck_counter > 20000:
                    break # Restart if stuck too long
            
            # Pick a violated clause
            # We need to pick a random violated K4 or K5
            # To avoid converting sets to lists repeatedly (expensive), 
            # we can maintain a list of violated indices and update it? 
            # Or just use random.choice on the set if it's small? 
            # Set iteration is fast, but random.choice requires a sequence.
            # For small sets, list(set) is fast. For large sets, it's slower.
            # In the beginning, violations are many. In the end, few.
            
            if len(violated_k4s) == 0 and len(violated_k5s) == 0:
                break
                
            # Heuristic: Pick from the smaller set to reduce variance? Or random?
            # Let's just pick randomly from the union.
            # Since we don't have a unified list, we pick a type first.
            # Probability proportional to size?
            
            total_viol = len(violated_k4s) + len(violated_k5s)
            if total_viol == 0:
                break
                
            # Pick a random violated clause
            # To make this fast, we can just pick a random triple and check if it's in a violation?
            # No, that's slow if violations are rare.
            
            # Let's convert to list if the set is small (< 1000)
            # If large, we are in the early phase, so random flips are okay, but Min-Conflicts is better.
            
            if len(violated_k4s) < 1000 and len(violated_k5s) < 1000:
                if violated_k4s and (not violated_k5s or random.random() < len(violated_k4s)/total_viol):
                    k4_idx = random.choice(list(violated_k4s))
                    t_idx = random.choice(k4_data[k4_idx])
                else:
                    k5_idx = random.choice(list(violated_k5s))
                    t_idx = random.choice(k5_data[k5_idx])
            else:
                # Early phase: Pick random triple, but bias towards violations if possible?
                # Actually, in early phase, almost all are violations.
                # Just pick random triple.
                t_idx = random.randint(0, num_triples - 1)
                
            # Flip t_idx
            old_val = colors[t_idx]
            new_val = 1 - old_val
            colors[t_idx] = new_val
            
            delta_red = 1 if new_val == 0 else -1
            delta_blue = 1 if new_val == 1 else -1
            
            # Update K4 counts
            for k4_idx in triple_to_k4s[t_idx]:
                old_count = k4_counts[k4_idx]
                new_count = old_count + delta_red
                k4_counts[k4_idx] = new_count
                
                was_viol = (old_count == 4)
                is_viol = (new_count == 4)
                
                if was_viol and not is_viol:
                    violated_k4s.discard(k4_idx)
                elif not was_viol and is_viol:
                    violated_k4s.add(k4_idx)
                    
                if was_viol:
                    current_violations -= 1
                if is_viol:
                    current_violations += 1
                    
            # Update K5 counts
            for k5_idx in triple_to_k5s[t_idx]:
                old_count = k5_counts[k5_idx]
                new_count = old_count + delta_blue
                k5_counts[k5_idx] = new_count
                
                was_viol = (old_count == 10)
                is_viol = (new_count == 10)
                
                if was_viol and not is_viol:
                    violated_k5s.discard(k5_idx)
                elif not was_viol and is_viol:
                    violated_k5s.add(k5_idx)
                    
                if was_viol:
                    current_violations -= 1
                if is_viol:
                    current_violations += 1
                    
            steps += 1
            
        # End of Min-Conflicts run
        if best_local_violations < (0 if best_colors is None else count_violations(*[None]*2) if False else 0): # This logic is messy
            pass
        
        # Simpler: Track global best
        if best_local_violations == 0:
            return {'n': n, 'colors': best_local_colors.tolist()}
            
        if best_colors is None or best_local_violations < best_violations_global:
             # Need to store global best violations
             pass

    # If we didn't find a solution for n=34, we should return the best valid n we can find quickly?
    # Or just return n=34 with random colors? No, it must be valid.
    # The task says "A valid colouring on n points certifies...".
    # If I return invalid colors, score is 0.
    # I should return a valid coloring for a smaller n if I can't find 34.
    # But finding a valid coloring for n=33 is also hard.
    # However, n=9 is trivial.
    
    # Let's try to find a valid coloring for n=34 more aggressively.
    # If time is up, return a known small valid coloring.
    
    # Fallback: n=9
    n_fallback = 9
    colors_fallback = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n_fallback), 3)]
    return {'n': n_fallback, 'colors': colors_fallback}
