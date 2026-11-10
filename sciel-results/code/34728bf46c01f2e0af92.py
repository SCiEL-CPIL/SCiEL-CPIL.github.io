import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit (or 150s as per instructions)
    
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
    triple_to_k5s = [[] for _ in range(num_triples)]
    
    # k4_data[i] = list of 4 triple indices in K4 i
    # k5_data[i] = list of 10 triple indices in K5 i
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

    # Convert to numpy arrays for fast vectorized operations for initial count
    k4_arr = np.array(k4_data, dtype=np.int32)
    k5_arr = np.array(k5_data, dtype=np.int32)
    
    def get_initial_counts_and_violations(colors):
        # colors: 0=Red, 1=Blue
        red_mask = (colors == 0)
        k4_counts = red_mask[k4_arr].sum(axis=1)
        
        blue_mask = (colors == 1)
        k5_counts = blue_mask[k5_arr].sum(axis=1)
        
        violated_k4s = set(np.where(k4_counts == 4)[0])
        violated_k5s = set(np.where(k5_counts == 10)[0])
        
        violations = len(violated_k4s) + len(violated_k5s)
        return k4_counts, k5_counts, violated_k4s, violated_k5s, violations

    def flip_with_sets(idx, colors, k4_counts, k5_counts, violated_k4s, violated_k5s):
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
            elif not was_viol and is_viol:
                violated_k4s.add(k4_idx)
                
        for k5_idx in triple_to_k5s[idx]:
            old_count = k5_counts[k5_idx]
            new_count = old_count + delta_blue
            k5_counts[k5_idx] = new_count
            
            was_viol = (old_count == 10)
            is_viol = (new_count == 10)
            
            if was_viol and not is_viol:
                violated_k5s.discard(k5_idx)
            elif not was_viol and is_viol:
                violated_k5s.add(k5_idx)
                
        return len(violated_k4s) + len(violated_k5s)

    # Main search loop
    max_restarts = 1000
    best_violations = float('inf')
    best_colors = None
    
    # Optimal probability for Red based on analysis: ~0.43
    p_red = 0.43
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Initialize state for this restart
        if restart > 50 and best_violations < 20 and random.random() < 0.5:
            # Perturb best
            colors = best_colors.copy()
            k4_counts, k5_counts, violated_k4s, violated_k5s, current_violations = get_initial_counts_and_violations(colors)
            
            # Do a few random flips to escape local min
            for _ in range(20):
                idx = random.randint(0, num_triples - 1)
                current_violations = flip_with_sets(idx, colors, k4_counts, k5_counts, violated_k4s, violated_k5s)
        else:
            # Biased Random Restart
            colors = (np.random.rand(num_triples) < p_red).astype(np.int8)
            k4_counts, k5_counts, violated_k4s, violated_k5s, current_violations = get_initial_counts_and_violations(colors)
            
        if current_violations == 0:
            return {'n': 34, 'colors': colors.tolist()}
            
        if current_violations < best_violations:
            best_violations = current_violations
            best_colors = colors.copy()
            
        # WalkSAT Loop with Max-Pressure Heuristic
        run_start = time.time()
        max_steps = 50000 # Cap steps per restart
        steps = 0
        
        while steps < max_steps and time.time() < run_start + 2.0:
            if current_violations == 0:
                return {'n': 34, 'colors': colors.tolist()}
                
            if current_violations < best_violations:
                best_violations = current_violations
                best_colors = colors.copy()
                
            # Pick a violated clause and a variable
            # Heuristic: Pick the variable involved in the most violated clauses
            # To keep it fast, we sample if the number of violations is large
            if len(violated_k4s) + len(violated_k5s) > 100:
                # Sample a few violated clauses to estimate pressure
                pressure = {}
                sample_size = min(50, len(violated_k4s) + len(violated_k5s))
                
                # Sample from K4s
                if violated_k4s:
                    k4_sample = random.sample(list(violated_k4s), min(25, len(violated_k4s)))
                    for k4_idx in k4_sample:
                        for t_idx in k4_data[k4_idx]:
                            pressure[t_idx] = pressure.get(t_idx, 0) + 1
                            
                # Sample from K5s
                if violated_k5s:
                    k5_sample = random.sample(list(violated_k5s), min(25, len(violated_k5s)))
                    for k5_idx in k5_sample:
                        for t_idx in k5_data[k5_idx]:
                            pressure[t_idx] = pressure.get(t_idx, 0) + 1
                
                if pressure:
                    # Pick max pressure, break ties randomly
                    max_p = max(pressure.values())
                    candidates = [t for t, p in pressure.items() if p == max_p]
                    t_idx = random.choice(candidates)
                else:
                    t_idx = random.randint(0, num_triples - 1)
            else:
                # Few violations, pick exact max pressure
                pressure = {}
                for k4_idx in violated_k4s:
                    for t_idx in k4_data[k4_idx]:
                        pressure[t_idx] = pressure.get(t_idx, 0) + 1
                for k5_idx in violated_k5s:
                    for t_idx in k5_data[k5_idx]:
                        pressure[t_idx] = pressure.get(t_idx, 0) + 1
                
                if pressure:
                    max_p = max(pressure.values())
                    candidates = [t for t, p in pressure.items() if p == max_p]
                    t_idx = random.choice(candidates)
                else:
                    t_idx = random.randint(0, num_triples - 1)
                    
            current_violations = flip_with_sets(t_idx, colors, k4_counts, k5_counts, violated_k4s, violated_k5s)
            steps += 1
            
        if best_violations == 0:
            break

    if best_violations == 0:
        return {'n': 34, 'colors': best_colors.tolist()}
    else:
        # Fallback to smaller n if we failed
        # n=9 is a safe bet
        n_fallback = 9
        colors_fallback = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n_fallback), 3)]
        return {'n': n_fallback, 'colors': colors_fallback}
