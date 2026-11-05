import numpy as np
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def get_triples(n):
        return list(combinations(range(n), 3))

    def precompute(n):
        triples = get_triples(n)
        num_triples = len(triples)
        triple_index = {t: i for i, t in enumerate(triples)}
        
        four_sets = list(combinations(range(n), 4))
        five_sets = list(combinations(range(n), 5))
        
        triple_to_4sets = [[] for _ in range(num_triples)]
        triple_to_5sets = [[] for _ in range(num_triples)]
        
        # For each 4-set, add its 4 triples to the mapping
        for i, s4 in enumerate(four_sets):
            a, b, c, d = s4
            for t in combinations(s4, 3):
                triple_to_4sets[triple_index[t]].append(i)
                
        # For each 5-set, add its 10 triples to the mapping
        for i, s5 in enumerate(five_sets):
            for t in combinations(s5, 3):
                triple_to_5sets[triple_index[t]].append(i)
                
        return triples, triple_index, four_sets, five_sets, triple_to_4sets, triple_to_5sets

    def initial_coloring(n, mode, seed=42):
        """
        Generate structured initial colorings.
        Mode 0: Random
        Mode 1: Modular (x+y+z mod 2)
        Mode 2: Quadratic form (x*y + y*z + z*x mod 2)
        Mode 3: Threshold
        Mode 4: Cyclic quadratic residue based
        """
        rng = np.random.RandomState(seed)
        colors = np.zeros(len(get_triples(n)), dtype=np.int8)
        for idx, (a, b, c) in enumerate(get_triples(n)):
            if mode == 0:
                colors[idx] = rng.randint(0, 2)
            elif mode == 1:
                colors[idx] = (a + b + c) % 2
            elif mode == 2:
                colors[idx] = (a * b + b * c + c * a) % 2
            elif mode == 3:
                colors[idx] = 0 if (a + b + c) < n else 1
            elif mode == 4:
                # Cyclic construction: color based on differences
                colors[idx] = ((b - a) * (c - a)) % 2
            else:
                colors[idx] = rng.randint(0, 2)
        return colors

    def local_search(n, time_limit, init_mode, seed=42):
        triples, triple_index, four_sets, five_sets, triple_to_4sets, triple_to_5sets = precompute(n)
        num_triples = len(triples)
        num_4sets = len(four_sets)
        num_5sets = len(five_sets)
        
        colors = initial_coloring(n, init_mode, seed)
        
        # Maintain counts of red triples in each K4 and blue triples in each K5
        red_count_4 = np.zeros(num_4sets, dtype=np.int32)
        blue_count_5 = np.zeros(num_5sets, dtype=np.int32)
        
        # Initialize counts
        for t_idx in range(num_triples):
            is_red = (colors[t_idx] == 0)
            for s4_idx in triple_to_4sets[t_idx]:
                if is_red:
                    red_count_4[s4_idx] += 1
            for s5_idx in triple_to_5sets[t_idx]:
                if not is_red:
                    blue_count_5[s5_idx] += 1
                    
        # Initial cost
        current_cost = np.sum(red_count_4 == 4) + np.sum(blue_count_5 == 10)
        
        if current_cost == 0:
            return colors, True
            
        best_cost = current_cost
        best_colors = colors.copy()
        
        # Local Search: Greedy flip with random restarts if stuck
        # We'll do a simple hill climbing step: find a flip that reduces cost.
        # To be fast, we randomly sample flips and check delta.
        
        steps = 0
        max_steps = 20000
        rng = np.random.RandomState(seed + 1)
        
        while time.time() < time_limit and steps < max_steps:
            # Pick a random triple to flip
            t_idx = rng.randint(num_triples)
            old_color = colors[t_idx]
            new_color = 1 - old_color
            
            # Calculate delta cost
            delta = 0
            
            # Update K4s
            for s4_idx in triple_to_4sets[t_idx]:
                rc = red_count_4[s4_idx]
                if old_color == 0: # Was Red
                    if rc == 4:
                        delta -= 1
                    elif rc == 3:
                        delta += 1
                else: # Was Blue
                    if rc == 3:
                        delta += 1
                    
            for s5_idx in triple_to_5sets[t_idx]:
                bc = blue_count_5[s5_idx]
                if old_color == 1: # Was Blue
                    if bc == 10:
                        delta -= 1
                    elif bc == 9:
                        delta += 1
                else: # Was Red
                    if bc == 9:
                        delta += 1
                        
            new_cost = current_cost + delta
            
            if new_cost < current_cost:
                # Apply flip
                colors[t_idx] = new_color
                for s4_idx in triple_to_4sets[t_idx]:
                    if old_color == 0:
                        red_count_4[s4_idx] -= 1
                    else:
                        red_count_4[s4_idx] += 1
                for s5_idx in triple_to_5sets[t_idx]:
                    if old_color == 1:
                        blue_count_5[s5_idx] -= 1
                    else:
                        blue_count_5[s5_idx] += 1
                        
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors.copy()
                    if best_cost == 0:
                        return best_colors, True
                steps += 1
            else:
                # No improvement, just count step to avoid infinite loop
                steps += 1
                
        return best_colors, (best_cost == 0)

    # Strategy: Try different n values, prioritizing higher n if time permits.
    # Start with smaller n to find valid colorings quickly, then try larger.
    # But we want to maximize n.
    # Let's try n=34 first? If it fails, try 33, etc.
    # Or start with a known good n?
    # The seed was n=9.
    # Let's try a sequence: 34, 33, 32, ... down to 10.
    # But 34 is hard. Maybe start with 20?
    # Let's try a few modes for n=34.
    
    final_n = 9
    final_colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]
    
    # Try n=34 with different modes
    # If that fails, try n=33, etc.
    
    target_ns = [34, 33, 32, 30, 25, 20, 15, 12, 10]
    
    for n in target_ns:
        if time.time() > deadline - 5:
            break
            
        # Allocate time based on n?
        # Let's give 5 seconds for n=34, 3 seconds for others?
        # Or just 5 seconds each.
        time_limit = time.time() + 5.0
        
        # Try a few modes
        modes = [4, 2, 1, 0, 3]
        
        for mode in modes:
            if time.time() > time_limit:
                break
            try:
                colors, is_valid = local_search(n, time_limit, mode)
                if is_valid:
                    final_n = n
                    final_colors = colors.tolist()
                    # print(f"Found valid coloring for n={n} with mode={mode}")
                    break # Move to next n
            except Exception:
                continue
                
    return {'n': final_n, 'colors': final_colors}
