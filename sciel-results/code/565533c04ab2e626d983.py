import time
import random
import math
from itertools import combinations

def construct():
    deadline = time.time() + 115.0
    
    # Try n from 34 down to 10
    for n in range(34, 9, -1):
        if time.time() > deadline - 10.0:
            break
        result = solve_n(n, time.time(), deadline)
        if result is not None:
            return result
            
    # Fallback: return a small valid coloring
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n), 3)]}

def solve_n(n, start_time, deadline):
    # Precompute structures
    triples = list(combinations(range(n), 3))
    triple_index = {t: i for i, t in enumerate(triples)}
    num_triples = len(triples)
    
    quads = list(combinations(range(n), 4))
    quints = list(combinations(range(n), 5))
    num_quads = len(quads)
    num_quints = len(quints)
    
    # Map triple index to list of quad indices containing it
    triple_to_quads = [[] for _ in range(num_triples)]
    # Map triple index to list of quint indices containing it
    triple_to_quints = [[] for _ in range(num_triples)]
    
    # Map quad index to its 4 triple indices
    quad_to_triples = [None] * num_quads
    for q_idx, q in enumerate(quads):
        a, b, c, d = q
        t1 = triple_index[(a, b, c)]
        t2 = triple_index[(a, b, d)]
        t3 = triple_index[(a, c, d)]
        t4 = triple_index[(b, c, d)]
        quad_to_triples[q_idx] = [t1, t2, t3, t4]
        for t in [t1, t2, t3, t4]:
            triple_to_quads[t].append(q_idx)
            
    # Map quint index to its 10 triple indices
    quint_to_triples = [None] * num_quints
    for p_idx, p in enumerate(quints):
        a, b, c, d, e = p
        ts = []
        for comb in combinations([a,b,c,d,e], 3):
            ts.append(triple_index[comb])
        quint_to_triples[p_idx] = ts
        for t in ts:
            triple_to_quints[t].append(p_idx)
            
    # Initialize state
    colors = [random.randint(0, 1) for _ in range(num_triples)]
    red_quad_counts = [0] * num_quads
    blue_quint_counts = [0] * num_quints
    
    # Calculate initial counts
    for q_idx, ts in enumerate(quad_to_triples):
        count = 0
        for t in ts:
            if colors[t] == 0:
                count += 1
        red_quad_counts[q_idx] = count
        
    for p_idx, ts in enumerate(quint_to_triples):
        count = 0
        for t in ts:
            if colors[t] == 1:
                count += 1
        blue_quint_counts[p_idx] = count
        
    # Calculate initial violations
    total_violations = 0
    for c in red_quad_counts:
        if c == 4:
            total_violations += 1
    for c in blue_quint_counts:
        if c == 10:
            total_violations += 1
            
    best_violations = total_violations
    best_colors = colors[:]
    
    if best_violations == 0:
        return {'n': n, 'colors': best_colors}
        
    # Simulated Annealing
    T = 1.0
    cooling_rate = 0.999
    min_T = 0.01
    
    steps = 0
    max_steps = 50000 # Adjust based on time
    
    while time.time() < deadline - 5.0:
        # Pick random triple
        i = random.randint(0, num_triples - 1)
        old_color = colors[i]
        new_color = 1 - old_color
        
        delta_v = 0
        
        # Update quads
        for q_idx in triple_to_quads[i]:
            old_count = red_quad_counts[q_idx]
            if old_color == 0: # Red -> Blue
                new_count = old_count - 1
                if old_count == 4:
                    delta_v -= 1
                # new_count cannot be 4 if old_count was 3
            else: # Blue -> Red
                new_count = old_count + 1
                if new_count == 4:
                    delta_v += 1
                # old_count cannot be 4 if new_count is 5? No, old_count could be 3.
            red_quad_counts[q_idx] = new_count
            
        # Update quints
        for p_idx in triple_to_quints[i]:
            old_count = blue_quint_counts[p_idx]
            if old_color == 0: # Red -> Blue
                new_count = old_count + 1
                if new_count == 10:
                    delta_v += 1
            else: # Blue -> Red
                new_count = old_count - 1
                if old_count == 10:
                    delta_v -= 1
            blue_quint_counts[p_idx] = new_count
            
        new_violations = total_violations + delta_v
        
        if new_violations < total_violations or (T > min_T and random.random() < math.exp(-delta_v / T)):
            # Accept
            colors[i] = new_color
            total_violations = new_violations
            
            if total_violations < best_violations:
                best_violations = total_violations
                best_colors = colors[:]
                if best_violations == 0:
                    return {'n': n, 'colors': best_colors}
                    
        # Cool down
        T *= cooling_rate
        steps += 1
        
        if steps > max_steps:
            # Restart or break
            # If we haven't found a solution, break to try smaller n or restart
            # Let's just break and rely on the outer loop or a restart mechanism
            # For simplicity, break and let the caller decide
            break
            
    if best_violations == 0:
        return {'n': n, 'colors': best_colors}
    return None

