import time
import random
import itertools
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Try n=34 first
    target_n = 34
    
    # Fallbacks if 34 fails
    fallback_ns = [33, 32, 31, 30, 29, 28, 27, 26, 25]
    
    best_result = None
    best_n = 0

    for n in [target_n] + fallback_ns:
        if time.time() > deadline - 5.0:
            break
            
        # Precompute structures for this n
        triples = list(itertools.combinations(range(n), 3))
        num_triples = len(triples)
        
        # Map triple to index
        triple_to_idx = {t: i for i, t in enumerate(triples)}
        
        # Precompute quad and quint incidences
        # quads: list of lists of quad_indices for each triple
        # quints: list of lists of quint_indices for each triple
        
        # We need to iterate over all quads and quints to build these lists
        # and also to initialize counts.
        
        quads = list(itertools.combinations(range(n), 4))
        quints = list(itertools.combinations(range(n), 5))
        
        quad_to_idx = {q: i for i, q in enumerate(quads)}
        quint_to_idx = {q: i for i, q in enumerate(quints)}
        
        # Incidence lists: for each triple, which quads/quints contain it
        # triple_idx -> [quad_idx, ...]
        # triple_idx -> [quint_idx, ...]
        
        quad_inc = [[] for _ in range(num_triples)]
        quint_inc = [[] for _ in range(num_triples)]
        
        # Build quad_inc
        for qi, q in enumerate(quads):
            # q is (a,b,c,d)
            # triples are (a,b,c), (a,b,d), (a,c,d), (b,c,d)
            t1 = (q[0], q[1], q[2])
            t2 = (q[0], q[1], q[3])
            t3 = (q[0], q[2], q[3])
            t4 = (q[1], q[2], q[3])
            quad_inc[triple_to_idx[t1]].append(qi)
            quad_inc[triple_to_idx[t2]].append(qi)
            quad_inc[triple_to_idx[t3]].append(qi)
            quad_inc[triple_to_idx[t4]].append(qi)
            
        # Build quint_inc
        for qui, q in enumerate(quints):
            # q is (a,b,c,d,e)
            # 10 triples
            for t in itertools.combinations(q, 3):
                quint_inc[triple_to_idx[t]].append(qui)
                
        # Initialize random coloring
        colors = [random.randint(0, 1) for _ in range(num_triples)]
        
        # Initialize counts
        # red_counts_4[qi] = number of red triples in quad qi
        # blue_counts_5[qui] = number of blue triples in quint qui
        
        red_counts_4 = [0] * len(quads)
        blue_counts_5 = [0] * len(quints)
        
        for ti, t in enumerate(triples):
            c = colors[ti]
            for qi in quad_inc[ti]:
                if c == 0:
                    red_counts_4[qi] += 1
            for qui in quint_inc[ti]:
                if c == 1:
                    blue_counts_5[qui] += 1
                    
        # Count initial violations
        viol_4 = sum(1 for c in red_counts_4 if c == 4)
        viol_5 = sum(1 for c in blue_counts_5 if c == 10)
        total_viol = viol_4 + viol_5
        
        if total_viol == 0:
            return {'n': n, 'colors': colors}
            
        # Simulated Annealing
        # T starts high, decreases
        T = 1.0
        # We want to finish before deadline
        # Estimate iterations per second? Hard to know.
        # Just run until time limit or convergence.
        
        last_time = time.time()
        iter_count = 0
        
        while time.time() < deadline - 5.0:
            # Pick random triple
            ti = random.randint(0, num_triples - 1)
            old_c = colors[ti]
            new_c = 1 - old_c
            
            # Calculate delta E
            delta_4 = 0
            for qi in quad_inc[ti]:
                old_cnt = red_counts_4[qi]
                new_cnt = old_cnt
                if old_c == 0:
                    new_cnt -= 1
                else:
                    new_cnt += 1
                # Check if violation status changes
                if old_cnt == 4 and new_cnt != 4:
                    delta_4 -= 1
                elif old_cnt != 4 and new_cnt == 4:
                    delta_4 += 1
                red_counts_4[qi] = new_cnt
                
            delta_5 = 0
            for qui in quint_inc[ti]:
                old_cnt = blue_counts_5[qui]
                new_cnt = old_cnt
                if old_c == 1:
                    new_cnt -= 1
                else:
                    new_cnt += 1
                if old_cnt == 10 and new_cnt != 10:
                    delta_5 -= 1
                elif old_cnt != 10 and new_cnt == 10:
                    delta_5 += 1
                blue_counts_5[qui] = new_cnt
                
            delta_E = delta_4 + delta_5
            
            # Acceptance
            if delta_E < 0 or random.random() < pow(2.71828, -delta_E / T):
                colors[ti] = new_c
                total_viol += delta_E
                if total_viol == 0:
                    return {'n': n, 'colors': colors}
            else:
                # Revert counts
                for qi in quad_inc[ti]:
                    if old_c == 0:
                        red_counts_4[qi] += 1
                    else:
                        red_counts_4[qi] -= 1
                for qui in quint_inc[ti]:
                    if old_c == 1:
                        blue_counts_5[qui] += 1
                    else:
                        blue_counts_5[qui] -= 1
                        
            # Update temperature
            # Cool down slowly
            T *= 0.999
            if T < 0.01:
                T = 0.01
                
            iter_count += 1
            # If stuck for a long time, maybe restart?
            # For now, just continue.
            
        # If we exit the loop without finding a solution, save best n so far if any
        # But we only return if valid.
        # We can try to return a smaller n if we haven't found a valid one for larger n.
        # However, the loop above tries n=34, then 33, etc.
        # If n=34 fails, it moves to n=33.
        
    # If we didn't find a valid coloring for any n in the list, 
    # return the smallest n that is trivially valid? 
    # n=1, 2, 3 are always valid.
    # But we want to maximize n.
    # If we didn't find 34..25, we can try a simple valid coloring for n=9 (seed)
    # or just return n=9.
    
    # To be safe, return the seed solution if we fail to find better.
    n_seed = 9
    return {'n': n_seed, 'colors': [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in itertools.combinations(range(n_seed), 3)]}
