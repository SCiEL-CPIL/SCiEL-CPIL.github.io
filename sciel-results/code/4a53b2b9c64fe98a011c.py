import time
from itertools import combinations
import sys
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # We target n=34, then fall back to smaller n
    target_ns = [34, 33, 32, 31, 30]
    
    for n in target_ns:
        if time.time() > deadline - 10:
            break
            
        try:
            result = solve_sa(n, time_limit=deadline - 5.0)
            if result is not None:
                return result
        except Exception as e:
            # If error, try next n
            pass
            
    # Fallback: return a small valid coloring if nothing else works
    # For n=9, a simple coloring works?
    # Let's just return a trivial valid one for small n if all else fails
    n_fallback = 9
    triples = list(combinations(range(n_fallback), 3))
    colors = [0 if (a+b+c)%3==0 else 1 for a,b,c in triples]
    return {'n': n_fallback, 'colors': colors}

def solve_sa(n, time_limit):
    triples = list(combinations(range(n), 3))
    num_triples = len(triples)
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    # Build K4s and K5s
    k4s = []
    k4_containing = [[] for _ in range(num_triples)]
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        t1 = triple_to_idx[(a,b,c)]
        t2 = triple_to_idx[(a,b,d)]
        t3 = triple_to_idx[(a,c,d)]
        t4 = triple_to_idx[(b,c,d)]
        idx_tuple = (t1, t2, t3, t4)
        k4s.append(idx_tuple)
        for t in idx_tuple:
            k4_containing[t].append(len(k4s) - 1)
            
    k5s = []
    k5_containing = [[] for _ in range(num_triples)]
    for quint in combinations(range(n), 5):
        idxs = []
        for triple in combinations(quint, 3):
            idxs.append(triple_to_idx[triple])
        idx_tuple = tuple(idxs)
        k5s.append(idx_tuple)
        for t in idx_tuple:
            k5_containing[t].append(len(k5s) - 1)
            
    num_k4s = len(k4s)
    num_k5s = len(k5s)
    
    # State
    # colors: 0 (Red) or 1 (Blue)
    colors = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
    
    # Counts
    # red_count_k4[i] = number of red triples in K4 i
    red_count_k4 = np.zeros(num_k4s, dtype=np.int32)
    # blue_count_k5[i] = number of blue triples in K5 i
    blue_count_k5 = np.zeros(num_k5s, dtype=np.int32)
    
    # Initialize counts
    # Instead of iterating all K4s/K5s and all their triples (O(N^4)), 
    # we can iterate triples and update counts for containing K4s/K5s.
    for t in range(num_triples):
        c = colors[t]
        for k4_idx in k4_containing[t]:
            if c == 0:
                red_count_k4[k4_idx] += 1
        for k5_idx in k5_containing[t]:
            if c == 1:
                blue_count_k5[k5_idx] += 1
                
    # Calculate initial cost
    # Cost = (# K4s with red_count == 4) + (# K5s with blue_count == 10)
    cost = np.sum(red_count_k4 == 4) + np.sum(blue_count_k5 == 10)
    
    # Simulated Annealing
    # T starts high, decreases
    # We want to minimize cost.
    
    # Heuristic: We want to find a state with cost 0.
    
    # Parameters
    T = 1.0
    cooling = 0.999
    min_T = 1e-4
    
    # We will run SA until time_limit or cost == 0
    
    # Precompute lists of K4s and K5s for each triple for faster access in loop
    # k4_containing and k5_containing are already lists of lists.
    
    current_cost = cost
    
    # To speed up cost updates, we track the delta.
    # When we flip t from c to 1-c:
    # For each K4 containing t:
    #   if c==0 (was Red, now Blue): red_count decreases by 1.
    #   if c==1 (was Blue, now Red): red_count increases by 1.
    #   Check if this change creates/destroys a "bad" K4 (count==4).
    #   old_bad = (old_count == 4)
    #   new_bad = (new_count == 4)
    #   delta_cost += new_bad - old_bad
    # Same for K5s with blue count.
    
    rng = np.random.default_rng()
    
    while time.time() < time_limit:
        if current_cost == 0:
            return {'n': n, 'colors': colors.tolist()}
            
        # Pick a random triple to flip
        t = rng.integers(num_triples)
        old_c = colors[t]
        new_c = 1 - old_c
        
        delta_cost = 0
        
        # Update K4s
        for k4_idx in k4_containing[t]:
            old_count = red_count_k4[k4_idx]
            if old_c == 0: # Was Red, removing Red
                new_count = old_count - 1
                red_count_k4[k4_idx] = new_count
                # If it was bad (4) and now is not (3), cost decreases
                if old_count == 4:
                    delta_cost -= 1
                # If it was not bad (3) and now is bad (4)? Impossible since we decreased.
            else: # Was Blue, adding Red
                new_count = old_count + 1
                red_count_k4[k4_idx] = new_count
                # If it was not bad (3) and now is bad (4), cost increases
                if old_count == 3:
                    delta_cost += 1
                # If it was bad (4) and now is worse (5)? No, max is 4.
        
        # Update K5s
        for k5_idx in k5_containing[t]:
            old_count = blue_count_k5[k5_idx]
            if old_c == 1: # Was Blue, removing Blue
                new_count = old_count - 1
                blue_count_k5[k5_idx] = new_count
                if old_count == 10:
                    delta_cost -= 1
            else: # Was Red, adding Blue
                new_count = old_count + 1
                blue_count_k5[k5_idx] = new_count
                if old_count == 9:
                    delta_cost += 1
        
        new_cost = current_cost + delta_cost
        
        # Acceptance criterion
        if new_cost < current_cost or rng.random() < np.exp(-delta_cost / T):
            # Accept move
            colors[t] = new_c
            current_cost = new_cost
        else:
            # Reject move: Undo changes
            # Revert K4s
            for k4_idx in k4_containing[t]:
                if old_c == 0:
                    red_count_k4[k4_idx] += 1
                else:
                    red_count_k4[k4_idx] -= 1
            # Revert K5s
            for k5_idx in k5_containing[t]:
                if old_c == 1:
                    blue_count_k5[k5_idx] += 1
                else:
                    blue_count_k5[k5_idx] -= 1
            
        # Cool down
        T *= cooling
        if T < min_T:
            T = min_T
            # Optional: Restart with higher T if stuck?
            # For now, just keep going or break if no progress for a long time.
            # A simple restart strategy:
            # If we haven't improved in X steps, reset T and randomize slightly?
            # Let's stick to simple SA first.
            
    # If we finish time, check if we found a solution
    if current_cost == 0:
        return {'n': n, 'colors': colors.tolist()}
        
    return None
