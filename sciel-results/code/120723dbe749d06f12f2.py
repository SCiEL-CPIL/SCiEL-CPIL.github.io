import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    best_n = 9
    best_colors = None # from seed
    
    # Try to find valid coloring for n up to 40
    # Start with n=34 as it's the known bound
    for n in range(34, 41):
        if time.time() > deadline - 20: break
        
        # Initialize
        num_triples = len(list(combinations(range(n), 3)))
        # Use a list of 0/1
        colors = [random.randint(0, 1) for _ in range(num_triples)]
        
        # Build index map
        idx_map = {}
        triples_list = []
        for i, (a,b,c) in enumerate(combinations(range(n), 3)):
            idx_map[(a,b,c)] = i
            triples_list.append((a,b,c))
            
        # Precompute 4-sets and 5-sets structure
        # For each 4-set, store list of 3 triple indices? No, 4 triples.
        # For each 5-set, store list of 10 triple indices.
        
        # This precomputation might take some time.
        # Let's check if it's worth it.
        # For n=34, binom(34,4)=46376, binom(34,5)=278256.
        # Storing these lists is fine.
        
        four_sets = []
        for quad in combinations(range(n), 4):
            t_idxs = []
            # 4 triples in a quad
            # (0,1,2), (0,1,3), (0,2,3), (1,2,3)
            # quad is (a,b,c,d)
            a,b,c,d = quad
            t_idxs.append(idx_map[(a,b,c)])
            t_idxs.append(idx_map[(a,b,d)])
            t_idxs.append(idx_map[(a,c,d)])
            t_idxs.append(idx_map[(b,c,d)])
            four_sets.append(t_idxs)
            
        five_sets = []
        for quint in combinations(range(n), 5):
            t_idxs = []
            # 10 triples
            for triple in combinations(quint, 3):
                t_idxs.append(idx_map[triple])
            five_sets.append(t_idxs)
            
        # Initial counts
        red_counts_4 = [0] * len(four_sets)
        blue_counts_5 = [0] * len(five_sets)
        
        # Helper to update counts
        # This is tricky to do incrementally without storing the current state of each set's counts.
        # We store red_counts_4[i] = number of red triples in 4-set i.
        # We store blue_counts_5[i] = number of blue triples in 5-set i.
        
        # Initialize counts
        for i, t_idxs in enumerate(four_sets):
            count = 0
            for t in t_idxs:
                if colors[t] == 0: count += 1
            red_counts_4[i] = count
            
        for i, t_idxs in enumerate(five_sets):
            count = 0
            for t in t_idxs:
                if colors[t] == 1: count += 1
            blue_counts_5[i] = count
            
        # Count violations
        violations = sum(1 for c in red_counts_4 if c == 4) + sum(1 for c in blue_counts_5 if c == 10)
        
        # Local Search
        while time.time() < deadline - 5 and violations > 0:
            # Pick a random violation
            # If red violation exists, pick one
            # Else pick blue violation
            
            # To be efficient, we could maintain lists of violating set indices.
            # But updating them is overhead.
            # For n=34, scanning is fast enough?
            # 46k + 278k = 324k checks.
            # 10k iterations * 324k = 3.2 billion ops. Too slow in Python.
            
            # We need a faster way to pick a violation.
            # Maintain lists: red_violations, blue_violations.
            # When a count changes, update the list?
            # Or just pick a random set and check if it's a violation?
            # Probability of picking a violation if there are few is low.
            
            # Better: Maintain a list of currently violating 4-sets and 5-sets.
            # Update it when we flip a triple.
            
            # Let's restart with a simpler approach:
            # Random flip, check delta.
            # But calculating delta requires knowing the current counts.
            
            # Let's stick to:
            # 1. Maintain red_counts_4 and blue_counts_5.
            # 2. Maintain list of indices of violating 4-sets and 5-sets.
            
            red_viol_idx = [i for i, c in enumerate(red_counts_4) if c == 4]
            blue_viol_idx = [i for i, c in enumerate(blue_counts_5) if c == 10]
            
            # Re-initialize these lists inside the loop? No, update them.
            # This is getting complex.
            
            # Alternative:
            # Since n is small, maybe just use a SAT solver? No, std lib only.
            
            # Let's try a simpler local search:
            # Flip random triple.
            # Check if it reduces violations.
            # To check, we need to know which 4-sets and 5-sets are affected.
            # For a triple t, affected 4-sets are those containing t.
            # Affected 5-sets are those containing t.
            
            # Precompute for each triple t:
            # affected_4_sets[t] = list of 4-set indices containing t
            # affected_5_sets[t] = list of 5-set indices containing t
            
            # This is better.
            
            # Re-do precomputation
            affected_4 = [[] for _ in range(num_triples)]
            affected_5 = [[] for _ in range(num_triples)]
            
            for i, t_idxs in enumerate(four_sets):
                for t in t_idxs:
                    affected_4[t].append(i)
                    
            for i, t_idxs in enumerate(five_sets):
                for t in t_idxs:
                    affected_5[t].append(i)
            
            # Now, local search loop
            # We need to track red_counts_4 and blue_counts_5.
            
            # Initialize counts again (done above)
            
            # Loop
            while time.time() < deadline - 5:
                # Pick random triple
                t = random.randint(0, num_triples - 1)
                
                # Calculate delta
                # Current color c = colors[t]
                # New color 1-c
                
                # For each 4-set in affected_4[t]:
                # If c == 0 (red), flipping to blue decreases red_count by 1.
                # If c == 1 (blue), flipping to red increases red_count by 1.
                
                delta_viol = 0
                
                for s_idx in affected_4[t]:
                    old_count = red_counts_4[s_idx]
                    if colors[t] == 0:
                        new_count = old_count - 1
                    else:
                        new_count = old_count + 1
                    
                    old_viol = 1 if old_count == 4 else 0
                    new_viol = 1 if new_count == 4 else 0
                    delta_viol += new_viol - old_viol
                    
                for s_idx in affected_5[t]:
                    old_count = blue_counts_5[s_idx]
                    if colors[t] == 1:
                        new_count = old_count - 1
                    else:
                        new_count = old_count + 1
                    
                    old_viol = 1 if old_count == 10 else 0
                    new_viol = 1 if new_count == 10 else 0
                    delta_viol += new_viol - old_viol
                    
                # Accept if delta < 0 or (delta == 0 and random < 0.5)
                if delta_viol < 0 or (delta_viol == 0 and random.random() < 0.1):
                    # Apply flip
                    colors[t] = 1 - colors[t]
                    
                    # Update counts
                    for s_idx in affected_4[t]:
                        if colors[t] == 0: # Now red
                            red_counts_4[s_idx] += 1
                        else:
                            red_counts_4[s_idx] -= 1
                            
                    for s_idx in affected_5[t]:
                        if colors[t] == 1: # Now blue
                            blue_counts_5[s_idx] += 1
                        else:
                            blue_counts_5[s_idx] -= 1
                            
                    violations += delta_viol
                    
                    if violations == 0:
                        break
                        
        if violations == 0:
            best_n = n
            best_colors = colors[:]
            # Try next n
        else:
            # Failed for this n, try next n? Or break?
            # If we failed 34, 35 might be harder.
            # But maybe random init was bad.
            # Let's try a few restarts for the same n if time permits.
            pass

    # If we didn't find anything better than 9, return 9?
    # Or return the best found.
    
    if best_n == 9:
        # Fallback to seed
        return {'n': 9, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]}
    else:
        return {'n': best_n, 'colors': best_colors}
