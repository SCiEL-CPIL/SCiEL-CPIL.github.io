import numpy as np
import itertools
import random
import time

def construct():
    max_n = 34
    start_time = time.time()
    time_limit = 115  # seconds
    
    for n in range(max_n, 8, -1):
        if time.time() - start_time > time_limit:
            break
        
        triples = list(itertools.combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Build incidence: for each triple, which 4-sets and 5-sets contain it
        four_sets = list(itertools.combinations(range(n), 4))
        five_sets = list(itertools.combinations(range(n), 5))
        num_4 = len(four_sets)
        num_5 = len(five_sets)
        
        # For each 4-set, store its 4 triple indices
        four_set_triples = np.zeros((num_4, 4), dtype=np.int32)
        for i, s in enumerate(four_sets):
            for j, t in enumerate(itertools.combinations(s, 3)):
                four_set_triples[i, j] = triple_to_idx[t]
        
        # For each 5-set, store its 10 triple indices
        five_set_triples = np.zeros((num_5, 10), dtype=np.int32)
        for i, s in enumerate(five_sets):
            for j, t in enumerate(itertools.combinations(s, 3)):
                five_set_triples[i, j] = triple_to_idx[t]
        
        # Reverse: for each triple, list of (type, set_idx)
        # type 0 = 4-set, type 1 = 5-set
        triple_to_sets = [[] for _ in range(num_triples)]
        for idx in range(num_4):
            for t_idx in four_set_triples[idx]:
                triple_to_sets[int(t_idx)].append((0, idx))
        for idx in range(num_5):
            for t_idx in five_set_triples[idx]:
                triple_to_sets[int(t_idx)].append((1, idx))
        
        found = False
        for seed in range(20):
            if time.time() - start_time > time_limit:
                break
            
            rng = np.random.RandomState(seed)
            colors = rng.randint(0, 2, size=num_triples).astype(np.int8)
            
            # Compute red_count_4 and blue_count_5
            red_count_4 = np.zeros(num_4, dtype=np.int8)
            blue_count_5 = np.zeros(num_5, dtype=np.int8)
            
            for t_idx in range(num_triples):
                c = colors[t_idx]
                for set_type, set_idx in triple_to_sets[t_idx]:
                    if set_type == 0:
                        if c == 0:
                            red_count_4[set_idx] += 1
                    else:
                        if c == 1:
                            blue_count_5[set_idx] += 1
            
            penalty = np.sum(red_count_4 == 4) + np.sum(blue_count_5 == 10)
            
            if penalty == 0:
                found = True
                break
            
            # Local search with first-improvement
            max_iters = 3000
            for it in range(max_iters):
                if time.time() - start_time > time_limit:
                    break
                
                if penalty == 0:
                    found = True
                    break
                
                # Find bad sets and their triples
                bad_4_idx = np.where(red_count_4 == 4)[0]
                bad_5_idx = np.where(blue_count_5 == 10)[0]
                
                if len(bad_4_idx) == 0 and len(bad_5_idx) == 0:
                    penalty = 0
                    found = True
                    break
                
                # Collect candidate triples
                candidates = set()
                for i in bad_4_idx:
                    candidates.update(four_set_triples[i].tolist())
                for i in bad_5_idx:
                    candidates.update(five_set_triples[i].tolist())
                
                if not candidates:
                    break
                
                # Try each candidate, pick best gain
                best_gain = 0
                best_flip = -1
                cand_list = list(candidates)
                rng.shuffle(cand_list)
                
                for t_idx in cand_list:
                    c = colors[t_idx]
                    gain = 0
                    for set_type, set_idx in triple_to_sets[t_idx]:
                        if set_type == 0:
                            if c == 0:  # Red->Blue
                                if red_count_4[set_idx] == 4:
                                    gain += 1
                            else:  # Blue->Red
                                if red_count_4[set_idx] == 3:
                                    gain -= 1
                        else:
                            if c == 1:  # Blue->Red
                                if blue_count_5[set_idx] == 10:
                                    gain += 1
                            else:  # Red->Blue
                                if blue_count_5[set_idx] == 9:
                                    gain -= 1
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_flip = t_idx
                
                if best_flip == -1:
                    # No improving move, pick random
                    best_flip = rng.choice(cand_list)
                    # Calculate gain for this move
                    c = colors[best_flip]
                    best_gain = 0
                    for set_type, set_idx in triple_to_sets[best_flip]:
                        if set_type == 0:
                            if c == 0:
                                if red_count_4[set_idx] == 4:
                                    best_gain += 1
                            else:
                                if red_count_4[set_idx] == 3:
                                    best_gain -= 1
                        else:
                            if c == 1:
                                if blue_count_5[set_idx] == 10:
                                    best_gain += 1
                            else:
                                if blue_count_5[set_idx] == 9:
                                    best_gain -= 1
                
                # Apply flip
                t_idx = best_flip
                c = colors[t_idx]
                colors[t_idx] = 1 - c
                
                for set_type, set_idx in triple_to_sets[t_idx]:
                    if set_type == 0:
                        if c == 0:
                            red_count_4[set_idx] -= 1
                        else:
                            red_count_4[set_idx] += 1
                    else:
                        if c == 1:
                            blue_count_5[set_idx] -= 1
                        else:
                            blue_count_5[set_idx] += 1
                
                penalty += best_gain
        
        if found and penalty == 0:
            return {'n': n, 'colors': colors.tolist()}
    
    # Fallback: small valid n
    n = 9
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in itertools.combinations(range(n), 3)]
    return {'n': n, 'colors': colors}
