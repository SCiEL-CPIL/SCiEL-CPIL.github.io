import numpy as np
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Precompute triples for various n
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
                # For Z_n, difference d = b-a mod n.
                # Let's try a simple pattern: (b-a)*(c-a) mod 2? Or just random with structure.
                # A better cyclic construction for hypergraph Ramsey often uses specific polynomials.
                # Let's try: color = ((b-a) % n * (c-a) % n) % 2
                # Or simpler: (a*b + b*c + c*a + a + b + c) % 2
                colors[idx] = ((b - a) * (c - a)) % 2
            else:
                colors[idx] = rng.randint(0, 2)
        return colors

    def count_violations(n, colors, four_sets, five_sets, triple_index, triples):
        """
        Count red K4s and blue K5s.
        """
        red_k4 = 0
        for s4 in four_sets:
            # Check if all 4 triples are red (0)
            a, b, c, d = s4
            if (colors[triple_index[(a,b,c)]] == 0 and
                colors[triple_index[(a,b,d)]] == 0 and
                colors[triple_index[(a,c,d)]] == 0 and
                colors[triple_index[(b,c,d)]] == 0):
                red_k4 += 1
                
        blue_k5 = 0
        for s5 in five_sets:
            # Check if all 10 triples are blue (1)
            is_blue = True
            for t in combinations(s5, 3):
                if colors[triple_index[t]] != 1:
                    is_blue = False
                    break
            if is_blue:
                blue_k5 += 1
                
        return red_k4, blue_k5

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
                # Before:
                if old_color == 0:
                    # Was red
                    if red_count_4[s4_idx] == 4:
                        delta -= 1 # Removing a red from a red K4 breaks it
                    elif red_count_4[s4_idx] == 3:
                        delta += 1 # Adding a red to a K4 with 3 reds creates a violation
                else:
                    # Was blue
                    if red_count_4[s4_idx] == 4:
                        delta += 1 # Wait, if it was blue, red_count didn't include it.
                        # If we change blue->red, and red_count was 4? No, if it was blue, it wasn't counted.
                        # If red_count_4[s4_idx] == 3, and we change blue->red, it becomes 4.
                        # If red_count_4[s4_idx] == 4, that's impossible if old_color was blue?
                        # Wait, red_count_4 counts reds. If old_color is blue, it is not in the count.
                        # So if red_count_4 == 4, and we change blue->red, it becomes 5? No, max is 4.
                        # Ah, if red_count_4 == 4, all 4 are red. So old_color must be red.
                        # So this case (old_color blue, red_count_4 == 4) is impossible.
                        # Correct logic:
                        # If old_color == 0 (Red):
                        #   If red_count == 4: Delta -1 (breaks violation)
                        #   If red_count == 3: Delta +1 (creates violation)
                        #   Else: Delta 0
                        # If old_color == 1 (Blue):
                        #   If red_count == 3: Delta +1 (creates violation, 3->4)
                        #   If red_count == 4: Impossible
                        #   Else: Delta 0
                        
                        pass # Handled below for clarity
                    # Let's rewrite cleanly
                    pass

            # Let's do a cleaner delta calculation
            delta = 0
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
                    # rc==4 impossible if old was blue? 
                    # Wait. If old was blue, it wasn't counted. So rc is count of OTHER reds.
                    # If rc==3, other 3 are red. Changing this to red makes 4. So +1.
                    # If rc==4, other 4 are red? Impossible, there are only 3 others.
                    # Max reds in K4 is 4. If this one is blue, max others is 3. So rc <= 3.
                    # So if rc==3, delta +1.
                    
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
                    # bc==10 impossible if old was red.
                    
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
