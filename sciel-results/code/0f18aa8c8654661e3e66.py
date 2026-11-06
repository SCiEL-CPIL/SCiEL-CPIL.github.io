import numpy as np
import time
import itertools
import random

def construct():
    # Target n values to try, in descending order
    target_ns = list(range(34, 8, -1))
    
    # Deadline
    start_time = time.time()
    total_time_limit = 140.0  # Safety margin for 150s limit
    
    best_result = None
    best_n = 0
    
    for n in target_ns:
        # Check time
        if time.time() - start_time > total_time_limit - 10:
            break
            
        # Try to find a valid coloring for this n
        result = solve_for_n(n, time_limit=10.0)
        
        if result is not None:
            return result
            
        # If not found, continue to next n
        # Note: If we fail for 34, we try 33. 
        # But if we find a valid one for 33, we return it.
        # The loop returns immediately on success.
        
    # If no valid coloring found for n > 8, return the seed program's result
    # to ensure we don't score 0
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in itertools.combinations(range(n), 3)]}

def solve_for_n(n, time_limit):
    # Precomputation
    triples = list(itertools.combinations(range(n), 3))
    M = len(triples)
    triple_index = {t: i for i, t in enumerate(triples)}
    
    four_sets = list(itertools.combinations(range(n), 4))
    K4 = len(four_sets)
    four_set_index = {s: i for i, s in enumerate(four_sets)}
    
    five_sets = list(itertools.combinations(range(n), 5))
    K5 = len(five_sets)
    five_set_index = {s: i for i, s in enumerate(five_sets)}
    
    # Map triples to the 4-sets and 5-sets they belong to
    # t_4_sets[i] = list of indices of 4-sets containing triple i
    # t_5_sets[i] = list of indices of 5-sets containing triple i
    t_4_sets = [[] for _ in range(M)]
    t_5_sets = [[] for _ in range(M)]
    
    for s in four_sets:
        idx_s = four_set_index[s]
        a, b, c, d = s
        # Triples in s: (a,b,c), (a,b,d), (a,c,d), (b,c,d)
        t1 = (a, b, c)
        t2 = (a, b, d)
        t3 = (a, c, d)
        t4 = (b, c, d)
        t_4_sets[triple_index[t1]].append(idx_s)
        t_4_sets[triple_index[t2]].append(idx_s)
        t_4_sets[triple_index[t3]].append(idx_s)
        t_4_sets[triple_index[t4]].append(idx_s)
        
    for s in five_sets:
        idx_s = five_set_index[s]
        a, b, c, d, e = s
        # Triples in s: 10 combinations
        # (a,b,c), (a,b,d), (a,b,e), (a,c,d), (a,c,e), (a,d,e), (b,c,d), (b,c,e), (b,d,e), (c,d,e)
        triples_in_s = [
            (a, b, c), (a, b, d), (a, b, e),
            (a, c, d), (a, c, e), (a, d, e),
            (b, c, d), (b, c, e), (b, d, e),
            (c, d, e)
        ]
        for t in triples_in_s:
            t_5_sets[triple_index[t]].append(idx_s)
            
    # Initialize colors randomly
    colors = np.random.randint(0, 2, M)
    
    # Initialize counts
    red_counts_4 = np.zeros(K4, dtype=np.int32)
    blue_counts_5 = np.zeros(K5, dtype=np.int32)
    
    # Calculate initial counts
    for i, t in enumerate(triples):
        c = colors[i]
        if c == 0: # Red
            for j in t_4_sets[i]:
                red_counts_4[j] += 1
        else: # Blue
            for k in t_5_sets[i]:
                blue_counts_5[k] += 1
                
    N_red_4 = np.sum(red_counts_4 == 4)
    N_blue_5 = np.sum(blue_counts_5 == 10)
    E = N_red_4 + N_blue_5
    
    if E == 0:
        return {'n': n, 'colors': colors.tolist()}
        
    # Simulated Annealing
    T = 1.0
    cooling = 0.99
    min_T = 0.01
    
    deadline = time.time() + time_limit
    steps = 0
    
    while time.time() < deadline:
        # Pick random triple
        i = np.random.randint(0, M)
        c = colors[i]
        
        # Calculate delta E
        dE = 0
        
        if c == 0: # Red -> Blue
            # Red counts decrease
            for j in t_4_sets[i]:
                if red_counts_4[j] == 4:
                    dE -= 1 # Removes a red clique
                # else if 3, becomes 2, no change in count of cliques
            # Blue counts increase
            for k in t_5_sets[i]:
                if blue_counts_5[k] == 9:
                    dE += 1 # Creates a blue clique
                # else if 10, becomes 11? No, max 10. If it was 10, it was a clique.
                # Wait, if it was 10, it was a blue clique. Flipping blue to red?
                # No, we are flipping Red (0) to Blue (1).
                # So the edge was NOT blue. So blue count was < 10.
                # If it was 9, it becomes 10. New blue clique. dE += 1.
                # If it was < 9, no change.
                
        else: # Blue -> Red
            # Red counts increase
            for j in t_4_sets[i]:
                if red_counts_4[j] == 3:
                    dE += 1 # Creates a red clique
            # Blue counts decrease
            for k in t_5_sets[i]:
                if blue_counts_5[k] == 10:
                    dE -= 1 # Removes a blue clique
                # If it was 9, becomes 8. No change in count of cliques.
                
        # Acceptance
        if dE < 0 or (T > min_T and np.random.rand() < np.exp(-dE / T)):
            # Apply flip
            colors[i] = 1 - c
            if c == 0:
                for j in t_4_sets[i]:
                    red_counts_4[j] -= 1
                for k in t_5_sets[i]:
                    blue_counts_5[k] += 1
            else:
                for j in t_4_sets[i]:
                    red_counts_4[j] += 1
                for k in t_5_sets[i]:
                    blue_counts_5[k] -= 1
                    
            E += dE
            if E == 0:
                return {'n': n, 'colors': colors.tolist()}
                
        T *= cooling
        
    return None

# EVOLVE-BLOCK-START
def construct():
    # Wrapper to match the expected interface if needed, but the above is the logic.
    # The prompt asks to rewrite the program. The function name is construct.
    # I will place the logic inside construct or call a helper.
    # To keep it clean, I'll put the logic in construct directly or call solve_for_n.
    
    target_ns = list(range(34, 8, -1))
    start_time = time.time()
    total_time_limit = 140.0
    
    for n in target_ns:
        if time.time() - start_time > total_time_limit - 10:
            break
        result = solve_for_n(n, time_limit=10.0)
        if result is not None:
            return result
            
    # Fallback
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in itertools.combinations(range(n), 3)]}
# EVOLVE-BLOCK-END
