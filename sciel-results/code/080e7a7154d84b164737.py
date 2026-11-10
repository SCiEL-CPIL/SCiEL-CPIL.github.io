import time
import random
import math
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    # We aim for n=34
    n = 34
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    # Precompute K4s and K5s
    k4_list = list(combinations(range(n), 4))
    k5_list = list(combinations(range(n), 5))
    
    # Map triples to the K4s and K5s they belong to
    # triple_to_k4s[i] = list of indices of K4s containing triple i
    # triple_to_k5s[i] = list of indices of K5s containing triple i
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

    def get_initial_violations(colors):
        k4_counts = [0] * len(k4_data)
        k5_counts = [0] * len(k5_data)
        
        for i, c in enumerate(colors):
            if c == 0: # Red
                for k4_idx in triple_to_k4s[i]:
                    k4_counts[k4_idx] += 1
            else: # Blue
                for k5_idx in triple_to_k5s[i]:
                    k5_counts[k5_idx] += 1
                    
        violations = sum(1 for c in k4_counts if c == 4) + sum(1 for c in k5_counts if c == 10)
        return k4_counts, k5_counts, violations

    def flip_and_update(idx, colors, k4_counts, k5_counts, violations):
        """
        Flip color of triple idx. Returns new violations count.
        colors, k4_counts, k5_counts are modified in place.
        """
        old_val = colors[idx]
        new_val = 1 - old_val
        colors[idx] = new_val
        
        # Delta for Red count:
        # If 0->1 (R->B), Red count decreases by 1.
        # If 1->0 (B->R), Red count increases by 1.
        delta_red = -1 if old_val == 0 else 1
        
        # Delta for Blue count:
        # If 0->1 (R->B), Blue count increases by 1.
        # If 1->0 (B->R), Blue count decreases by 1.
        delta_blue = 1 if old_val == 0 else -1
        
        # Update K4s (Red constraint)
        for k4_idx in triple_to_k4s[idx]:
            old_count = k4_counts[k4_idx]
            new_count = old_count + delta_red
            k4_counts[k4_idx] = new_count
            
            if old_count == 4:
                violations -= 1
            if new_count == 4:
                violations += 1
                
        # Update K5s (Blue constraint)
        for k5_idx in triple_to_k5s[idx]:
            old_count = k5_counts[k5_idx]
            new_count = old_count + delta_blue
            k5_counts[k5_idx] = new_count
            
            if old_count == 10:
                violations -= 1
            if new_count == 10:
                violations += 1
                
        return violations

    best_violations = float('inf')
    best_colors = None
    
    # Simulated Annealing Parameters
    # We need to run multiple restarts to cover the space
    max_restarts = 50
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Initialize random coloring
        colors = [random.randint(0, 1) for _ in range(num_triples)]
        k4_counts, k5_counts, current_violations = get_initial_violations(colors)
        
        if current_violations == 0:
            return {'n': n, 'colors': colors}
            
        if current_violations < best_violations:
            best_violations = current_violations
            best_colors = colors[:]
            
        # SA Loop
        # Start temperature high enough to explore, cool down
        T = 10.0
        cooling = 0.995
        min_T = 0.1
        
        # Run SA for a fixed number of steps or time slice
        run_start = time.time()
        time_slice = 1.5 # seconds per restart
        
        while time.time() < run_start + time_slice:
            if current_violations == 0:
                return {'n': n, 'colors': colors}
                
            if current_violations < best_violations:
                best_violations = current_violations
                best_colors = colors[:]
                
            if T < min_T:
                break
                
            # Pick random triple
            idx = random.randint(0, num_triples - 1)
            
            # Calculate delta
            # We need to simulate the flip to get delta, or calculate it directly.
            # Direct calculation is faster than flipping and unflipping.
            old_val = colors[idx]
            
            # Delta in violations
            delta_v = 0
            
            # Check K4s
            for k4_idx in triple_to_k4s[idx]:
                old_count = k4_counts[k4_idx]
                # If we flip, red count changes by delta_red
                # delta_red = -1 if old_val==0 else 1
                # new_count = old_count + delta_red
                # Violation if count == 4
                
                # Shortcut:
                # If old_val == 0 (Red), flipping to Blue reduces red count.
                #   If old_count == 4, it was violated, now it's 3 (not violated). delta_v -= 1
                #   If old_count == 3, it was not violated, now it's 2 (not violated). delta_v += 0
                # If old_val == 1 (Blue), flipping to Red increases red count.
                #   If old_count == 4, it was violated, now it's 5 (not violated? No, 5!=4). 
                #   Wait, violation is ONLY if count == 4.
                #   If old_count == 3, flipping to Red makes it 4 (violated). delta_v += 1
                #   If old_count == 4, flipping to Red makes it 5 (not violated). delta_v -= 1
                
                if old_val == 0:
                    if old_count == 4:
                        delta_v -= 1
                else:
                    if old_count == 3:
                        delta_v += 1
                    elif old_count == 4:
                        delta_v -= 1
                        
            # Check K5s
            for k5_idx in triple_to_k5s[idx]:
                old_count = k5_counts[k5_idx]
                # Blue count changes by delta_blue
                # delta_blue = 1 if old_val==0 else -1
                
                # Violation if count == 10
                
                if old_val == 0:
                    # Flipping R->B increases blue count
                    if old_count == 9:
                        # Becomes 10 (violated)
                        delta_v += 1
                    elif old_count == 10:
                        # Becomes 11 (not violated)
                        delta_v -= 1
                else:
                    # Flipping B->R decreases blue count
                    if old_count == 10:
                        # Becomes 9 (not violated)
                        delta_v -= 1
                    elif old_count == 11:
                        # Becomes 10 (violated)
                        delta_v += 1
                        
            # Acceptance probability
            if delta_v < 0:
                accept = True
            else:
                # P = exp(-delta_v / T)
                # If delta_v is 0, accept always? Or with prob 1?
                # Usually accept if random < exp(-delta/T). If delta=0, exp(0)=1.
                if delta_v == 0:
                    accept = True
                else:
                    # Optimization: if T is small and delta is large, reject
                    if T < 0.01 and delta_v > 10:
                        accept = False
                    else:
                        # Avoid overflow in exp
                        if -delta_v / T < -50:
                            accept = False
                        else:
                            accept = random.random() < math.exp(-delta_v / T)
                            
            if accept:
                current_violations = flip_and_update(idx, colors, k4_counts, k5_counts, current_violations)
            
            T *= cooling
            
        if best_violations == 0:
            break

    if best_violations == 0:
        return {'n': n, 'colors': best_colors}
    else:
        # Fallback
        n_fallback = 9
        colors_fallback = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n_fallback), 3)]
        return {'n': n_fallback, 'colors': colors_fallback}
