import numpy as np
import time
import itertools
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 110s to be safe
    
    # Target n
    n = 34
    
    # Generate all triples
    triples = list(itertools.combinations(range(n), 3))
    num_triples = len(triples)
    
    # Map triple to index
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    # Generate all 4-sets and 5-sets
    sets_4 = list(itertools.combinations(range(n), 4))
    sets_5 = list(itertools.combinations(range(n), 5))
    
    num_4sets = len(sets_4)
    num_5sets = len(sets_5)
    
    # Precompute which triples are in each 4-set and 5-set
    # For a 4-set {a,b,c,d}, the triples are (a,b,c), (a,b,d), (a,c,d), (b,c,d)
    # For a 5-set {a,b,c,d,e}, the triples are all 10 combinations
    
    # We need for each triple index, the list of 4-set indices and 5-set indices it belongs to.
    # This is an inverted index.
    
    edge_to_4sets = [[] for _ in range(num_triples)]
    edge_to_5sets = [[] for _ in range(num_triples)]
    
    for i, s4 in enumerate(sets_4):
        # Generate 4 triples from s4
        t1 = (s4[0], s4[1], s4[2])
        t2 = (s4[0], s4[1], s4[3])
        t3 = (s4[0], s4[2], s4[3])
        t4 = (s4[1], s4[2], s4[3])
        for t in (t1, t2, t3, t4):
            edge_to_4sets[triple_to_idx[t]].append(i)
            
    for i, s5 in enumerate(sets_5):
        # Generate 10 triples from s5
        for t in itertools.combinations(s5, 3):
            edge_to_5sets[triple_to_idx[t]].append(i)
            
    # State: colors array of 0s and 1s
    # 0 = Red, 1 = Blue
    
    best_energy = float('inf')
    best_colors = None
    
    # Try a few random restarts
    for restart in range(5):
        if time.time() > deadline - 10: break
        
        colors = np.random.randint(0, 2, num_triples)
        
        # Calculate initial energy
        # Energy = (# of all-red 4-sets) + (# of all-blue 5-sets)
        
        # We can maintain counts:
        # red_count_4[i] = number of red triples in 4-set i. If 4, it's a violation.
        # blue_count_5[i] = number of blue triples in 5-set i. If 10, it's a violation.
        
        red_counts_4 = np.zeros(num_4sets, dtype=np.int32)
        blue_counts_5 = np.zeros(num_5sets, dtype=np.int32)
        
        # Initialize counts
        for i, c in enumerate(colors):
            if c == 0: # Red
                for s4_idx in edge_to_4sets[i]:
                    red_counts_4[s4_idx] += 1
            else: # Blue
                for s5_idx in edge_to_5sets[i]:
                    blue_counts_5[s5_idx] += 1
                    
        current_energy = np.sum(red_counts_4 == 4) + np.sum(blue_counts_5 == 10)
        
        if current_energy == 0:
            return {'n': n, 'colors': colors.tolist()}
            
        if current_energy < best_energy:
            best_energy = current_energy
            best_colors = colors.copy()
            
        # Simulated Annealing
        T = 1.0
        T_min = 0.1
        cooling_rate = 0.999
        
        while time.time() < deadline - 5:
            # Pick random edge
            edge_idx = np.random.randint(num_triples)
            old_color = colors[edge_idx]
            new_color = 1 - old_color
            
            delta_energy = 0
            
            # Update red_counts_4
            # If old was Red, it contributed to red_counts. New is Blue, so it contributes to blue_counts (but we track blue separately for 5-sets).
            # Wait, the energy is based on "all red" in 4-sets and "all blue" in 5-sets.
            # A 4-set violation depends ONLY on red edges.
            # A 5-set violation depends ONLY on blue edges.
            
            # So flipping an edge from Red to Blue:
            # 1. Decreases red_counts_4 for all 4-sets containing this edge.
            # 2. Increases blue_counts_5 for all 5-sets containing this edge.
            
            # Flipping Blue to Red:
            # 1. Increases red_counts_4 for all 4-sets containing this edge.
            # 2. Decreases blue_counts_5 for all 5-sets containing this edge.
            
            if old_color == 0: # Red -> Blue
                # Red counts decrease
                for s4_idx in edge_to_4sets[edge_idx]:
                    old_c = red_counts_4[s4_idx]
                    red_counts_4[s4_idx] = old_c - 1
                    # Energy change for this 4-set:
                    if old_c == 4:
                        delta_energy -= 1 # Was violation, now not
                    elif old_c - 1 == 4:
                        delta_energy += 1 # Was not, now is? No, 3->4 is not possible if decreasing.
                    
                # Blue counts increase
                for s5_idx in edge_to_5sets[edge_idx]:
                    old_c = blue_counts_5[s5_idx]
                    blue_counts_5[s5_idx] = old_c + 1
                    if old_c == 9:
                        delta_energy += 1 # Was not violation (9 blue), now is (10 blue)
                    elif old_c + 1 == 10:
                         # This is the same condition
                         pass
            
            else: # Blue -> Red
                # Red counts increase
                for s4_idx in edge_to_4sets[edge_idx]:
                    old_c = red_counts_4[s4_idx]
                    red_counts_4[s4_idx] = old_c + 1
                    if old_c == 3:
                        delta_energy += 1 # Was not (3 red), now is (4 red)
                    
                # Blue counts decrease
                for s5_idx in edge_to_5sets[edge_idx]:
                    old_c = blue_counts_5[s5_idx]
                    blue_counts_5[s5_idx] = old_c - 1
                    if old_c == 10:
                        delta_energy -= 1 # Was violation, now not
            
            if delta_energy < 0:
                colors[edge_idx] = new_color
                current_energy += delta_energy
                if current_energy == 0:
                    return {'n': n, 'colors': colors.tolist()}
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_colors = colors.copy()
            else:
                # Metropolis criterion
                if np.random.rand() < np.exp(-delta_energy / T):
                    colors[edge_idx] = new_color
                    current_energy += delta_energy
                    if current_energy < best_energy:
                        best_energy = current_energy
                        best_colors = colors.copy()
            
            T *= cooling_rate
            if T < T_min:
                T = T_min
        
    # If we didn't find energy 0, we return the best found?
    # But best found might be invalid.
    # We must return a VALID coloring.
    # If best_energy > 0, we should try smaller n?
    # Or just return the seed solution?
    
    # Let's verify the best_colors if it's not None
    if best_colors is not None and best_energy == 0:
        return {'n': n, 'colors': best_colors.tolist()}
        
    # Fallback: Return a valid small coloring
    # The seed coloring for n=9 is valid.
    # Let's just return the seed solution to ensure non-zero score.
    n_fallback = 9
    return {'n': n_fallback, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in itertools.combinations(range(n_fallback), 3)]}
