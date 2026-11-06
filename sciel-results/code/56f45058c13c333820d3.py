import random
import math
import time
from itertools import combinations

def construct():
    n = 34
    # Total triples
    triples = list(combinations(range(n), 3))
    T = len(triples)
    
    # Map triple to index
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    # Precompute which K4s and K5s contain which triples
    # For each triple, store the list of K4 indices and K5 indices it belongs to.
    # K4 is defined by 4 vertices. Its 4 triples are the edges.
    # K5 is defined by 5 vertices. Its 10 triples are the edges.
    
    # List of all K4s (as tuple of 4 vertices)
    k4s = list(combinations(range(n), 4))
    # List of all K5s (as tuple of 5 vertices)
    k5s = list(combinations(range(n), 5))
    
    # For each K4, get its 4 triple indices
    k4_triples = []
    for k4 in k4s:
        t_indices = [triple_to_idx[t] for t in combinations(k4, 3)]
        k4_triples.append(t_indices)
        
    # For each K5, get its 10 triple indices
    k5_triples = []
    for k5 in k5s:
        t_indices = [triple_to_idx[t] for t in combinations(k5, 3)]
        k5_triples.append(t_indices)
        
    # For each triple, which K4s and K5s does it affect?
    # triple_idx -> list of k4_indices
    # triple_idx -> list of k5_indices
    triple_to_k4s = [[] for _ in range(T)]
    triple_to_k5s = [[] for _ in range(T)]
    
    for i, t_inds in enumerate(k4_triples):
        for t in t_inds:
            triple_to_k4s[t].append(i)
            
    for i, t_inds in enumerate(k5_triples):
        for t in t_inds:
            triple_to_k5s[t].append(i)

    # Initialize colors randomly
    # Bias towards blue? R(4,5) means red K4 is forbidden, blue K5 is forbidden.
    # Usually, to avoid small red cliques, we make red rare? Or blue rare?
    # Red K4 needs 4 edges. Blue K5 needs 10 edges.
    # It is easier to avoid 4 edges than 10 edges?
    # If p_red is small, red K4 is very unlikely.
    # If p_blue is small, blue K5 is very unlikely.
    # Let's try p_red = 0.2, p_blue = 0.8.
    p_red = 0.2
    colors = [0 if random.random() < p_red else 1 for _ in range(T)]
    
    # Helper to check if a K4 is red (all 4 triples red)
    def is_red_k4(k4_idx):
        t_inds = k4_triples[k4_idx]
        return all(colors[t] == 0 for t in t_inds)
        
    # Helper to check if a K5 is blue (all 10 triples blue)
    def is_blue_k5(k5_idx):
        t_inds = k5_triples[k5_idx]
        return all(colors[t] == 1 for t in t_inds)

    # Calculate initial violations
    violations = 0
    red_violations = 0
    blue_violations = 0
    for i in range(len(k4s)):
        if is_red_k4(i):
            violations += 1
            red_violations += 1
    for i in range(len(k5s)):
        if is_blue_k5(i):
            violations += 1
            blue_violations += 1
            
    start_time = time.time()
    timeout = 115 # Leave 5s for overhead
    
    # Simulated Annealing / WalkSAT
    T_curr = 1.0
    cooling = 0.999
    
    while violations > 0 and time.time() - start_time < timeout:
        # Pick a random triple to flip
        # Better: Pick a triple involved in a violation
        # We can pick a random violated K4 or K5, then a random triple in it
        if random.random() < 0.5 and red_violations > 0:
            # Pick random red K4
            # We don't have a list of current violations, so we pick a random K4 and check?
            # That's inefficient.
            # Let's just pick a random triple from the whole set for now, 
            # or maintain a list of violated indices?
            # Maintaining a list is better.
            pass
            
        # Simple approach: Pick random triple
        t_idx = random.randint(0, T - 1)
        
        # Calculate change in violations if we flip t_idx
        delta_violations = 0
        
        # Check affected K4s
        for k4_idx in triple_to_k4s[t_idx]:
            t_inds = k4_triples[k4_idx]
            # Check if currently red
            is_red_now = all(colors[x] == 0 for x in t_inds)
            # Check if will be red (t_idx flips 0->1 or 1->0)
            # If colors[t_idx] is 0, it becomes 1.
            # If it was red, and we change one edge to blue, it's no longer red.
            # If it wasn't red, and we change one edge to red, it might become red.
            
            if is_red_now:
                # It was a violation.
                # If we flip t_idx to 1, it's no longer red.
                if colors[t_idx] == 0:
                    # Becomes blue, so K4 is no longer all red.
                    delta_violations -= 1
                else:
                    # This case shouldn't happen if is_red_now is true and t_idx is in it?
                    # If is_red_now is true, all are 0. So colors[t_idx] must be 0.
                    pass
            else:
                # It was not a violation.
                # If we flip t_idx to 0, it might become red.
                if colors[t_idx] == 1:
                    # Becomes red.
                    # Check if all OTHER triples are red.
                    # If yes, it becomes a violation.
                    all_others_red = True
                    for x in t_inds:
                        if x != t_idx and colors[x] != 0:
                            all_others_red = False
                            break
                    if all_others_red:
                        delta_violations += 1
                else:
                    # Becomes blue. Cannot create red K4.
                    pass
        
        # Check affected K5s
        for k5_idx in triple_to_k5s[t_idx]:
            t_inds = k5_triples[k5_idx]
            is_blue_now = all(colors[x] == 1 for x in t_inds)
            
            if is_blue_now:
                # It was a violation.
                if colors[t_idx] == 1:
                    # Becomes red, so K5 is no longer all blue.
                    delta_violations -= 1
                else:
                    pass
            else:
                if colors[t_idx] == 0:
                    # Becomes blue.
                    all_others_blue = True
                    for x in t_inds:
                        if x != t_idx and colors[x] != 1:
                            all_others_blue = False
                            break
                    if all_others_blue:
                        delta_violations += 1
                else:
                    pass
                    
        # Acceptance probability
        if delta_violations < 0:
            accept = True
        elif delta_violations == 0:
            accept = random.random() < 0.5
        else:
            # Boltzmann factor
            prob = math.exp(-delta_violations / T_curr)
            accept = random.random() < prob
            
        if accept:
            colors[t_idx] = 1 - colors[t_idx]
            violations += delta_violations
            # Update red/blue violation counts roughly? 
            # We just need total violations for the loop condition.
            # For picking specific violations later, we might need breakdown.
            # For now, total is enough for the while loop.
            
        T_curr *= cooling
        
    # Final check
    if violations == 0:
        return {'n': n, 'colors': colors}
    else:
        # Fallback: return a smaller n if we can find a valid one quickly?
        # Or just return the best effort? The task says "Maximize n".
        # If we fail on 34, we should try 33, 32...
        # But the time is likely exhausted.
        # Let's try a quick fallback for n=33?
        # No, the seed is n=9. If we return n=9, score is 1.0.
        # If we return invalid n=34, score is 0.
        # We MUST return a valid coloring.
        
        # Let's try to construct a valid n=33 quickly if 34 failed?
        # Or just return the seed n=9 to be safe?
        # The seed is n=9.
        # If we can't find 34, we should try to find a valid 33, 32, etc.
        # But the search for 34 consumed most of the time.
        
        # Safe fallback:
        return {'n': 9, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(9), 3)]}

# EVOLVE-BLOCK-START
def construct():
    n = 34
    triples = list(combinations(range(n), 3))
    T = len(triples)
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    k4s = list(combinations(range(n), 4))
    k5s = list(combinations(range(n), 5))
    
    k4_triples = []
    for k4 in k4s:
        t_indices = [triple_to_idx[t] for t in combinations(k4, 3)]
        k4_triples.append(t_indices)
        
    k5_triples = []
    for k5 in k5s:
        t_indices = [triple_to_idx[t] for t in combinations(k5, 3)]
        k5_triples.append(t_indices)
        
    triple_to_k4s = [[] for _ in range(T)]
    triple_to_k5s = [[] for _ in range(T)]
    
    for i, t_inds in enumerate(k4_triples):
        for t in t_inds:
            triple_to_k4s[t].append(i)
            
    for i, t_inds in enumerate(k5_triples):
        for t in t_inds:
            triple_to_k5s[t].append(i)

    p_red = 0.2
    colors = [0 if random.random() < p_red else 1 for _ in range(T)]
    
    def is_red_k4(k4_idx):
        t_inds = k4_triples[k4_idx]
        return all(colors[t] == 0 for t in t_inds)
        
    def is_blue_k5(k5_idx):
        t_inds = k5_triples[k5_idx]
        return all(colors[t] == 1 for t in t_inds)

    violations = 0
    for i in range(len(k4s)):
        if is_red_k4(i):
            violations += 1
    for i in range(len(k5s)):
        if is_blue_k5(i):
            violations += 1
            
    start_time = time.time()
    timeout = 115
    
    T_curr = 1.0
    cooling = 0.9995
    
    while violations > 0 and time.time() - start_time < timeout:
        t_idx = random.randint(0, T - 1)
        
        delta_violations = 0
        
        for k4_idx in triple_to_k4s[t_idx]:
            t_inds = k4_triples[k4_idx]
            is_red_now = all(colors[x] == 0 for x in t_inds)
            
            if is_red_now:
                if colors[t_idx] == 0:
                    delta_violations -= 1
            else:
                if colors[t_idx] == 1:
                    all_others_red = True
                    for x in t_inds:
                        if x != t_idx and colors[x] != 0:
                            all_others_red = False
                            break
                    if all_others_red:
                        delta_violations += 1
        
        for k5_idx in triple_to_k5s[t_idx]:
            t_inds = k5_triples[k5_idx]
            is_blue_now = all(colors[x] == 1 for x in t_inds)
            
            if is_blue_now:
                if colors[t_idx] == 1:
                    delta_violations -= 1
            else:
                if colors[t_idx] == 0:
                    all_others_blue = True
                    for x in t_inds:
                        if x != t_idx and colors[x] != 1:
                            all_others_blue = False
                            break
                    if all_others_blue:
                        delta_violations += 1
                        
        if delta_violations < 0:
            accept = True
        elif delta_violations == 0:
            accept = random.random() < 0.5
        else:
            prob = math.exp(-delta_violations / T_curr)
            accept = random.random() < prob
            
        if accept:
            colors[t_idx] = 1 - colors[t_idx]
            violations += delta_violations
            
        T_curr *= cooling
        
    if violations == 0:
        return {'n': n, 'colors': colors}
    else:
        # Fallback to seed
        n_seed = 9
        return {'n': n_seed, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in combinations(range(n_seed), 3)]}
# EVOLVE-BLOCK-END
