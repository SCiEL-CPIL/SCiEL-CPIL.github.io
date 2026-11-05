import itertools
import random
import time

def construct():
    max_n = 34
    start_time = time.time()
    time_limit = 110
    
    for n in range(max_n, 8, -1):
        if time.time() - start_time > time_limit:
            break
            
        triples = list(itertools.combinations(range(n), 3))
        num_triples = len(triples)
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        four_sets = list(itertools.combinations(range(n), 4))
        five_sets = list(itertools.combinations(range(n), 5))
        
        four_set_triples = []
        for s in four_sets:
            ts = [triple_to_idx[t] for t in itertools.combinations(s, 3)]
            four_set_triples.append(ts)
            
        five_set_triples = []
        for s in five_sets:
            ts = [triple_to_idx[t] for t in itertools.combinations(s, 3)]
            five_set_triples.append(ts)
            
        triple_to_sets = [[] for _ in range(num_triples)]
        for idx, ts in enumerate(four_set_triples):
            for t_idx in ts:
                triple_to_sets[t_idx].append((0, idx))
        for idx, ts in enumerate(five_set_triples):
            for t_idx in ts:
                triple_to_sets[t_idx].append((1, idx))
        
        found = False
        num_restarts = 20
        for seed in range(num_restarts):
            if time.time() - start_time > time_limit:
                break
                
            random.seed(seed * 137 + 42)
            colors = [random.randint(0, 1) for _ in range(num_triples)]
            
            red_count_4 = [0] * len(four_set_triples)
            blue_count_5 = [0] * len(five_set_triples)
            
            for t_idx, c in enumerate(colors):
                for set_type, set_idx in triple_to_sets[t_idx]:
                    if set_type == 0:
                        if c == 0:
                            red_count_4[set_idx] += 1
                    else:
                        if c == 1:
                            blue_count_5[set_idx] += 1
                            
            penalty = sum(1 for r in red_count_4 if r == 4) + sum(1 for b in blue_count_5 if b == 10)
            
            if penalty == 0:
                found = True
                break
            
            max_iters = 3000
            for it in range(max_iters):
                if time.time() - start_time > time_limit:
                    break
                    
                if penalty == 0:
                    found = True
                    break
                
                # Collect candidate triples from violating sets
                candidates = set()
                for i, r in enumerate(red_count_4):
                    if r == 4:
                        candidates.update(four_set_triples[i])
                for j, b in enumerate(blue_count_5):
                    if b == 10:
                        candidates.update(five_set_triples[j])
                        
                if not candidates:
                    break
                
                best_gain = 0
                best_flip = -1
                
                # Use list for faster iteration
                cand_list = list(candidates)
                for t_idx in cand_list:
                    c = colors[t_idx]
                    gain = 0
                    for set_type, set_idx in triple_to_sets[t_idx]:
                        if set_type == 0:
                            if c == 0:
                                if red_count_4[set_idx] == 4:
                                    gain += 1
                            else:
                                if red_count_4[set_idx] == 3:
                                    gain -= 1
                        else:
                            if c == 1:
                                if blue_count_5[set_idx] == 10:
                                    gain += 1
                            else:
                                if blue_count_5[set_idx] == 9:
                                    gain -= 1
                    if gain > best_gain:
                        best_gain = gain
                        best_flip = t_idx
                        
                if best_flip == -1:
                    # No improving move; try a random candidate to escape local minimum
                    best_flip = random.choice(cand_list)
                    c = colors[best_flip]
                    # Calculate actual gain for this random move
                    gain = 0
                    for set_type, set_idx in triple_to_sets[best_flip]:
                        if set_type == 0:
                            if c == 0:
                                if red_count_4[set_idx] == 4:
                                    gain += 1
                            else:
                                if red_count_4[set_idx] == 3:
                                    gain -= 1
                        else:
                            if c == 1:
                                if blue_count_5[set_idx] == 10:
                                    gain += 1
                            else:
                                if blue_count_5[set_idx] == 9:
                                    gain -= 1
                    best_gain = gain
                
                # Apply the flip
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

        if found:
            return {'n': n, 'colors': colors}
            
    # Fallback
    n = 9
    return {'n': n, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in itertools.combinations(range(n), 3)]}
