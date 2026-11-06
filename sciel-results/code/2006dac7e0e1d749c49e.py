import numpy as np
import itertools
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110s safety margin
    
    # Try to find a valid coloring for n=34
    # If that fails, try smaller n
    for n in range(34, 0, -1):
        if time.time() > deadline:
            break
        result = solve_n(n, deadline)
        if result is not None:
            return result
            
    # Fallback: return a small valid coloring
    # e.g. n=1 is always valid
    return {'n': 1, 'colors': []}

def solve_n(n, deadline):
    # Precompute triples
    triples = list(itertools.combinations(range(n), 3))
    m = len(triples)
    
    # Map triple to index
    triple_to_idx = {t: i for i, t in enumerate(triples)}
    
    # Precompute 4-sets and their triples
    four_sets = list(itertools.combinations(range(n), 4))
    four_set_triples = []
    for s in four_sets:
        trs = [triple_to_idx[t] for t in itertools.combinations(s, 3)]
        four_set_triples.append(trs)
        
    # Precompute 5-sets and their triples
    five_sets = list(itertools.combinations(range(n), 5))
    five_set_triples = []
    for s in five_sets:
        trs = [triple_to_idx[t] for t in itertools.combinations(s, 3)]
        five_set_triples.append(trs)
        
    # For each triple, which 4-sets and 5-sets does it belong to?
    triple_to_four_sets = [[] for _ in range(m)]
    triple_to_five_sets = [[] for _ in range(m)]
    
    for i, trs in enumerate(four_set_triples):
        for t_idx in trs:
            triple_to_four_sets[t_idx].append(i)
            
    for j, trs in enumerate(five_set_triples):
        for t_idx in trs:
            triple_to_five_sets[t_idx].append(j)
            
    # Initial random coloring
    colors = np.random.randint(0, 2, m)
    
    # Initialize counts
    # red_count_4[i] = number of red triples in 4-set i
    red_count_4 = np.zeros(len(four_sets), dtype=int)
    # blue_count_5[j] = number of blue triples in 5-set j
    blue_count_5 = np.zeros(len(five_sets), dtype=int)
    
    for i, trs in enumerate(four_set_triples):
        red_count_4[i] = np.sum(colors[trs] == 0)
        
    for j, trs in enumerate(five_set_triples):
        blue_count_5[j] = np.sum(colors[trs] == 1)
        
    # Identify violations
    # Violation 4: red_count_4[i] == 4
    # Violation 5: blue_count_5[j] == 10
    
    # We maintain lists of violating indices
    # To make it efficient, we can just re-scan if the number of violations is small?
    # Or maintain a set.
    # Let's use a list of (type, index) for violations.
    # But updating it efficiently is tricky.
    # Given n=34, len(four_sets)=46k, len(five_sets)=278k.
    # Re-scanning is O(324k). If we do 1000 steps, that's 324M ops. Too slow in Python.
    # Incremental update is better.
    
    # Let's maintain sets of violating 4-sets and 5-sets.
    viol_4 = set(i for i in range(len(four_sets)) if red_count_4[i] == 4)
    viol_5 = set(j for j in range(len(five_sets)) if blue_count_5[j] == 10)
    
    if not viol_4 and not viol_5:
        return {'n': n, 'colors': colors.tolist()}
        
    steps = 0
    while time.time() < deadline:
        if not viol_4 and not viol_5:
            return {'n': n, 'colors': colors.tolist()}
            
        # Pick a random violation
        # Weight by number of violations? Or just random pick from combined list?
        # Let's pick from 4-sets with prob proportional to size, same for 5-sets.
        # Or just pick one randomly.
        
        if viol_4 and (not viol_5 or random.random() < 0.5):
            # Pick random 4-set violation
            v4_idx = random.choice(list(viol_4))
            trs = four_set_triples[v4_idx]
            # All these triples are red (since count is 4). 
            # We pick one and flip it to blue.
            t_idx = random.choice(trs)
        else:
            # Pick random 5-set violation
            v5_idx = random.choice(list(viol_5))
            trs = five_set_triples[v5_idx]
            # All these triples are blue (since count is 10).
            # We pick one and flip it to red.
            t_idx = random.choice(trs)
            
        # Flip the color of t_idx
        old_color = colors[t_idx]
        new_color = 1 - old_color
        colors[t_idx] = new_color
        
        # Update counts for affected 4-sets
        for i in triple_to_four_sets[t_idx]:
            if old_color == 0:
                # Was red, now blue. Red count decreases.
                red_count_4[i] -= 1
                if red_count_4[i] == 4:
                    viol_4.add(i)
                elif i in viol_4:
                    viol_4.remove(i)
            else:
                # Was blue, now red. Red count increases.
                red_count_4[i] += 1
                if red_count_4[i] == 4:
                    viol_4.add(i)
                elif i in viol_4:
                    viol_4.remove(i)
                    
        # Update counts for affected 5-sets
        for j in triple_to_five_sets[t_idx]:
            if old_color == 1:
                # Was blue, now red. Blue count decreases.
                blue_count_5[j] -= 1
                if blue_count_5[j] == 10:
                    viol_5.add(j)
                elif j in viol_5:
                    viol_5.remove(j)
            else:
                # Was red, now blue. Blue count increases.
                blue_count_5[j] += 1
                if blue_count_5[j] == 10:
                    viol_5.add(j)
                elif j in viol_5:
                    viol_5.remove(j)
                    
        steps += 1
        if steps % 1000 == 0:
            # Optional: check if we are stuck?
            pass
            
    # If we run out of time, return None to try smaller n
    return None
