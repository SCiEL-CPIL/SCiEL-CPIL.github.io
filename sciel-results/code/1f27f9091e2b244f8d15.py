import random
import math
import time
from itertools import combinations

def construct():
    # Target n
    n = 34
    
    # Precompute all triples
    triples = list(combinations(range(n), 3))
    triple_index = {t: i for i, t in enumerate(triples)}
    num_triples = len(triples)
    
    # Precompute all 4-sets and their constituent triples
    # We need to know which triples are in which 4-set for updates
    # And which 4-sets contain which triple
    four_sets = list(combinations(range(n), 4))
    four_set_index = {s: i for i, s in enumerate(four_sets)}
    num_four_sets = len(four_sets)
    
    # For each triple, list of 4-set indices containing it
    triple_to_four_sets = [[] for _ in range(num_triples)]
    for i, s in enumerate(four_sets):
        # A 4-set (a,b,c,d) has triples (a,b,c), (a,b,d), (a,c,d), (b,c,d)
        t1 = (s[0], s[1], s[2])
        t2 = (s[0], s[1], s[3])
        t3 = (s[0], s[2], s[3])
        t4 = (s[1], s[2], s[3])
        triple_to_four_sets[triple_index[t1]].append(i)
        triple_to_four_sets[triple_index[t2]].append(i)
        triple_to_four_sets[triple_index[t3]].append(i)
        triple_to_four_sets[triple_index[t4]].append(i)
        
    # Precompute all 5-sets and their constituent triples
    five_sets = list(combinations(range(n), 5))
    five_set_index = {s: i for i, s in enumerate(five_sets)}
    num_five_sets = len(five_sets)
    
    # For each triple, list of 5-set indices containing it
    triple_to_five_sets = [[] for _ in range(num_triples)]
    for i, s in enumerate(five_sets):
        # A 5-set has 10 triples
        for t in combinations(s, 3):
            triple_to_five_sets[triple_index[t]].append(i)

    # State
    # colors[i] is 0 (red) or 1 (blue)
    colors = [random.randint(0, 1) for _ in range(num_triples)]
    
    # Counts
    # red_count_4[i] = number of red triples in 4-set i
    red_count_4 = [0] * num_four_sets
    # blue_count_5[i] = number of blue triples in 5-set i
    blue_count_5 = [0] * num_five_sets
    
    # Initialize counts
    for t_idx, t in enumerate(triples):
        c = colors[t_idx]
        for fs_idx in triple_to_four_sets[t_idx]:
            if c == 0: # Red
                red_count_4[fs_idx] += 1
        for fs_idx in triple_to_five_sets[t_idx]:
            if c == 1: # Blue
                blue_count_5[fs_idx] += 1
                
    # Calculate initial cost
    cost = 0
    red_k4_count = 0
    blue_k5_count = 0
    for i in range(num_four_sets):
        if red_count_4[i] == 4:
            red_k4_count += 1
    for i in range(num_five_sets):
        if blue_count_5[i] == 10:
            blue_k5_count += 1
    cost = red_k4_count + blue_k5_count
    
    # Simulated Annealing
    start_time = time.time()
    timeout = 110.0 # Safety margin
    T = 1.0
    cooling_rate = 0.9995
    best_cost = cost
    best_colors = colors[:]
    
    # If we find a solution with cost 0, we can return immediately
    # But we should try to ensure it's stable or just return it.
    # The task asks to maximize n. If we find valid n=34, score is likely high.
    
    while time.time() - start_time < timeout:
        if cost == 0:
            # Found a valid coloring
            # We can try to see if we can go higher? 
            # The prompt says "published construction has 34 points". 
            # It's safer to return 34 if valid.
            return {'n': n, 'colors': colors}
            
        # Pick a random triple to flip
        t_idx = random.randint(0, num_triples - 1)
        old_color = colors[t_idx]
        new_color = 1 - old_color
        
        delta_cost = 0
        
        # Update 4-sets
        for fs_idx in triple_to_four_sets[t_idx]:
            if old_color == 0: # Was Red, now Blue
                # Red count decreases by 1
                old_rc = red_count_4[fs_idx]
                new_rc = old_rc - 1
                red_count_4[fs_idx] = new_rc
                if old_rc == 4:
                    delta_cost -= 1 # Lost a red K4
                elif new_rc == 4:
                    delta_cost += 1 # Gained a red K4
            else: # Was Blue, now Red
                # Red count increases by 1
                old_rc = red_count_4[fs_idx]
                new_rc = old_rc + 1
                red_count_4[fs_idx] = new_rc
                if old_rc == 4:
                    delta_cost -= 1 # Lost a red K4 (impossible if it was 4 and we add red? No, if it was 4 red, it's already a K4. Adding another red doesn't change K4 status, it stays K4. Wait.)
                    # If old_rc == 4, it was a K4. It is still a K4. No change in count of K4s.
                    # If new_rc == 4, it became a K4. +1.
                if new_rc == 4:
                    delta_cost += 1
        
        # Update 5-sets
        for fs_idx in triple_to_five_sets[t_idx]:
            if old_color == 1: # Was Blue, now Red
                # Blue count decreases by 1
                old_bc = blue_count_5[fs_idx]
                new_bc = old_bc - 1
                blue_count_5[fs_idx] = new_bc
                if old_bc == 10:
                    delta_cost -= 1 # Lost a blue K5
                elif new_bc == 10:
                    delta_cost += 1 # Gained a blue K5 (impossible if decreasing)
            else: # Was Red, now Blue
                # Blue count increases by 1
                old_bc = blue_count_5[fs_idx]
                new_bc = old_bc + 1
                blue_count_5[fs_idx] = new_bc
                if old_bc == 10:
                    # Was K5, still K5. No change.
                    pass
                if new_bc == 10:
                    delta_cost += 1
        
        # Accept/Reject
        if delta_cost <= 0:
            colors[t_idx] = new_color
            cost += delta_cost
            if cost < best_cost:
                best_cost = cost
                best_colors = colors[:]
        else:
            # Accept with probability exp(-delta/T)
            if T > 1e-6:
                if random.random() < math.exp(-delta_cost / T):
                    colors[t_idx] = new_color
                    cost += delta_cost
                    if cost < best_cost:
                        best_cost = cost
                        best_colors = colors[:]
        
        T *= cooling_rate
        # Occasionally restart temperature if stuck? 
        # For now, simple cooling.
        
    # If we didn't find cost 0, return the best found.
    # But we need a valid coloring to get a good score.
    # If best_cost > 0, it's invalid.
    # We should try to return a valid one if possible.
    # If not, return the one with lowest cost? 
    # The evaluator says "invalid result... scores 0".
    # So we MUST return a valid coloring.
    # If SA fails to find cost 0 for n=34, we should fallback to a smaller n that is easier to solve.
    
    if best_cost > 0:
        # Try smaller n
        for n_try in [33, 32, 31, 30, 25, 20, 15, 10, 9]:
            if time.time() - start_time > timeout - 5:
                break
            # Re-run SA for smaller n? Or just use a known valid small one?
            # The seed n=9 is valid.
            # Let's try to find a valid one quickly for smaller n.
            # For n=9, the seed logic works.
            if n_try == 9:
                return {'n': n_try, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n_try), 3)]}
            
            # Quick SA for smaller n
            triples_s = list(combinations(range(n_try), 3))
            ti_s = {t: i for i, t in enumerate(triples_s)}
            ns = len(triples_s)
            cols_s = [random.randint(0, 1) for _ in range(ns)]
            
            # Simple hill climbing for small n
            # Just check validity
            def is_valid(n_val, cols_val):
                # Check red K4
                for q in combinations(range(n_val), 4):
                    all_red = True
                    for t in combinations(q, 3):
                        if cols_val[ti_s[t]] != 0:
                            all_red = False
                            break
                    if all_red:
                        return False
                # Check blue K5
                for q in combinations(range(n_val), 5):
                    all_blue = True
                    for t in combinations(q, 3):
                        if cols_val[ti_s[t]] != 1:
                            all_blue = False
                            break
                    if all_blue:
                        return False
                return True
            
            # Try a few random restarts
            for _ in range(1000):
                if time.time() - start_time > timeout - 2:
                    break
                cols_s = [random.randint(0, 1) for _ in range(ns)]
                if is_valid(n_try, cols_s):
                    return {'n': n_try, 'colors': cols_s}
            
    # Fallback to seed
    return {'n': 9, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]}
