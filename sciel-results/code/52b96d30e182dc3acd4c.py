"""
Maximize number of triples in a 3-uniform hypergraph on 16 vertices
such that no three disjoint pairs have all 8 transversal triples present.

Strategy:
1. Precompute all K(2,2,2) configurations and per-triple config lists.
2. Greedy initialization with randomization.
3. Local search with efficient add/swap moves using numpy bitsets for speed.
4. WalkSAT-style fluctuation phases to escape local optima.
5. Multiple restarts with time bounding.
"""

import time
import random
import itertools
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s kill
    
    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    num_total = len(all_triples)  # 560
    
    # Precompute all K(2,2,2) configurations
    k222_configs = []
    triple_to_k222 = [[] for _ in range(num_total)]
    
    for six_vertices in itertools.combinations(range(N), 6):
        a, b, c, d, e, f = six_vertices
        partitions = [
            ((a,b),(c,d),(e,f)),
            ((a,b),(c,e),(d,f)),
            ((a,b),(c,f),(d,e)),
            ((a,c),(b,d),(e,f)),
            ((a,c),(b,e),(d,f)),
            ((a,c),(b,f),(d,e)),
            ((a,d),(b,c),(e,f)),
            ((a,d),(b,e),(c,f)),
            ((a,d),(b,f),(c,e)),
            ((a,e),(b,c),(d,f)),
            ((a,e),(b,d),(c,f)),
            ((a,e),(b,f),(c,d)),
            ((a,f),(b,c),(d,e)),
            ((a,f),(b,d),(c,e)),
            ((a,f),(b,e),(c,d)),
        ]
        for p1, p2, p3 in partitions:
            transversals = []
            for x in p1:
                for y in p2:
                    for z in p3:
                        t = tuple(sorted((x, y, z)))
                        transversals.append(t)
            trans_indices = frozenset(triple_to_idx[t] for t in transversals)
            k222_configs.append(trans_indices)
            config_idx = len(k222_configs) - 1
            for ti in trans_indices:
                triple_to_k222[ti].append(config_idx)
    
    num_k222 = len(k222_configs)
    
    def time_check():
        return time.time() >= deadline
    
    best_result = []
    best_count = 0
    
    # Greedy initialization
    def greedy_init(rng):
        present = np.zeros(num_total, dtype=np.uint8)
        k222_count = np.zeros(num_k222, dtype=np.int32)
        count = 0
        
        order = list(range(num_total))
        rng.shuffle(order)
        
        for ti in order:
            can_add = True
            for ci in triple_to_k222[ti]:
                if k222_count[ci] == 7:
                    can_add = False
                    break
            if can_add:
                present[ti] = 1
                count += 1
                for ci in triple_to_k222[ti]:
                    k222_count[ci] += 1
        return present, k222_count, count
    
    # Local search with efficient moves and WalkSAT-style fluctuation
    def local_search(present, k222_count, count, rng, max_steps=500000):
        steps = 0
        no_improve = 0
        max_no_improve = 20000
        fluctuate_every = 5000  # Every N steps, do a fluctuation
        
        while steps < max_steps and no_improve < max_no_improve:
            if time_check():
                break
            
            # Periodic fluctuation: randomly remove a few triples to escape local optima
            if steps > 0 and steps % fluctuate_every == 0:
                num_remove = rng.randint(3, 8)
                present_indices = np.where(present == 1)[0]
                if len(present_indices) > num_remove:
                    to_remove = rng.sample(present_indices.tolist(), num_remove)
                    for ti in to_remove:
                        present[ti] = 0
                        count -= 1
                        for ci in triple_to_k222[ti]:
                            k222_count[ci] -= 1
                no_improve = 0
                steps += 1
                continue
            
            moved = False
            
            # Phase 1: Try to add triples
            for attempt in range(30):
                ti = rng.randint(0, num_total - 1)
                if present[ti]:
                    continue
                can_add = True
                for ci in triple_to_k222[ti]:
                    if k222_count[ci] == 7:
                        can_add = False
                        break
                if can_add:
                    present[ti] = 1
                    count += 1
                    for ci in triple_to_k222[ti]:
                        k222_count[ci] += 1
                    moved = True
                    no_improve = 0
                    break
            
            if moved:
                steps += 1
                continue
            
            # Phase 2: Try swaps - remove one triple, then try to add one or more
            swap_done = False
            for attempt in range(30):
                ti_remove = rng.randint(0, num_total - 1)
                if not present[ti_remove]:
                    continue
                
                # Temporarily remove
                for ci in triple_to_k222[ti_remove]:
                    k222_count[ci] -= 1
                present[ti_remove] = 0
                count -= 1
                
                # Try to add up to 2 triples (net gain of 1 if we add 2)
                added = 0
                added_indices = []
                for add_attempt in range(200):
                    ti_candidate = rng.randint(0, num_total - 1)
                    if present[ti_candidate]:
                        continue
                    can_add = True
                    for ci in triple_to_k222[ti_candidate]:
                        if k222_count[ci] == 7:
                            can_add = False
                            break
                    if can_add:
                        present[ti_candidate] = 1
                        count += 1
                        for ci in triple_to_k222[ti_candidate]:
                            k222_count[ci] += 1
                        added += 1
                        added_indices.append(ti_candidate)
                        if added >= 2:
                            break
                
                if added >= 1:
                    # Accept the swap
                    swap_done = True
                    no_improve = 0
                else:
                    # Revert removal
                    for ci in triple_to_k222[ti_remove]:
                        k222_count[ci] += 1
                    present[ti_remove] = 1
                    count += 1
            
            if swap_done:
                steps += 1
                continue
            
            # Stuck - increment no_improve
            no_improve += 1
            steps += 1
        
        return present, count
    
    # Main loop with multiple restarts
    rng = random.Random(42)
    
    num_restarts = 0
    while not time_check() and num_restarts < 500:
        num_restarts += 1
        
        # Greedy init with randomization
        present, k222_count, count = greedy_init(rng)
        
        if count > best_count:
            best_count = count
            best_result = [list(all_triples[i]) for i in range(num_total) if present[i]]
        
        # Local search
        present, count = local_search(present, k222_count, count, rng)
        
        if count > best_count:
            best_count = count
            best_result = [list(all_triples[i]) for i in range(num_total) if present[i]]
        
        if time_check():
            break
    
    if not best_result:
        # Fallback: return a valid simple solution
        best_result = [[0, a, b] for a in range(1, 16) for b in range(a + 1, 16)]
    
    return best_result
