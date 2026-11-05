import itertools
import random
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin before 150s limit
    
    points = list(range(16))
    all_triples = list(itertools.combinations(points, 3))
    num_triples = len(all_triples)
    
    # Map each triple to an index
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Pre-compute all transversals (forbidden configurations)
    # A transversal is defined by 3 disjoint pairs on 6 points, giving 8 triples
    transversals = []
    for six_pts in itertools.combinations(points, 6):
        sp = list(six_pts)
        # Generate all 15 perfect matchings on 6 points
        # Fix first element paired with one of the other 5
        # Then pair remaining 4: 3 ways
        # Total: 5 * 3 = 15
        matchings = []
        remaining = [1, 2, 3, 4, 5]
        for j in remaining:
            # First pair: (0, j)
            rest = [x for x in remaining if x != j]
            # Pair up the remaining 4: 3 ways
            for k in range(3):
                if k == 0:
                    p2, p3 = (rest[0], rest[1]), (rest[2], rest[3])
                elif k == 1:
                    p2, p3 = (rest[0], rest[2]), (rest[1], rest[3])
                else:
                    p2, p3 = (rest[0], rest[3]), (rest[1], rest[2])
                matching = [(sp[0], sp[j]), sp[p2[0]] and sp[p2[1]] and (sp[p2[0]], sp[p2[1]]), (sp[p3[0]], sp[p3[1]])]
                # Fix the matching structure properly
                matching = [(sp[0], sp[j]), (sp[p2[0]], sp[p2[1]]), (sp[p3[0]], sp[p3[1]])]
                matchings.append(matching)
        
        # For each matching, generate the 8 transversal triples
        for matching in matchings:
            (a, b), (c, d), (e, f) = matching
            # 8 transversals: one from each pair
            t1 = tuple(sorted((a, c, e)))
            t2 = tuple(sorted((a, c, f)))
            t3 = tuple(sorted((a, d, e)))
            t4 = tuple(sorted((a, d, f)))
            t5 = tuple(sorted((b, c, e)))
            t6 = tuple(sorted((b, c, f)))
            t7 = tuple(sorted((b, d, e)))
            t8 = tuple(sorted((b, d, f)))
            idxs = tuple(triple_to_idx[t] for t in (t1, t2, t3, t4, t5, t6, t7, t8))
            transversals.append(idxs)
    
    num_transversals = len(transversals)
    
    # For each triple, precompute which transversals it belongs to
    triple_transversals = [[] for _ in range(num_triples)]
    for ti, trans in enumerate(transversals):
        for idx in trans:
            triple_transversals[idx].append(ti)
    
    # Greedy construction with randomized ordering
    def greedy_construct(seed=42):
        rng = random.Random(seed)
        present = bytearray(num_triples)
        for i in range(num_triples):
            present[i] = 1
        
        # Shuffle transversals
        trans_order = list(range(num_transversals))
        rng.shuffle(trans_order)
        
        # Iteratively resolve conflicts
        progress = True
        while progress:
            if time.time() > deadline - 5:
                break
            progress = False
            for ti in trans_order:
                trans = transversals[ti]
                # Check if all 8 are present
                if all(present[idx] for idx in trans):
                    # Remove the triple with highest degree (appears in most transversals)
                    best_idx = -1
                    best_degree = -1
                    for idx in trans:
                        degree = len(triple_transversals[idx])
                        if degree > best_degree:
                            best_degree = degree
                            best_idx = idx
                    present[best_idx] = 0
                    progress = True
        
        return present
    
    # Try multiple seeds
    best_count = 0
    best_present = None
    
    seeds = [42, 123, 456, 789, 1000, 2024, 7, 13, 99, 314]
    for seed in seeds:
        if time.time() > deadline - 20:
            break
        present = greedy_construct(seed)
        count = sum(present)
        if count > best_count:
            best_count = count
            best_present = present
    
    # Local search: try to add back removed triples
    if best_present is not None:
        if time.time() < deadline - 10:
            removed_indices = [i for i in range(num_triples) if not best_present[i]]
            rng = random.Random(999)
            rng.shuffle(removed_indices)
            
            for idx in removed_indices:
                if time.time() > deadline - 5:
                    break
                # Check if adding this triple creates any conflict
                can_add = True
                for ti in triple_transversals[idx]:
                    trans = transversals[ti]
                    # Check if all other 7 are present
                    all_others_present = True
                    for other_idx in trans:
                        if other_idx != idx and not best_present[other_idx]:
                            all_others_present = False
                            break
                    if all_others_present:
                        can_add = False
                        break
                
                if can_add:
                    best_present[idx] = 1
    
    # Build result
    if best_present is not None:
        result = [list(all_triples[i]) for i in range(num_triples) if best_present[i]]
    else:
        # Fallback: return all triples
        result = [list(t) for t in all_triples]
    
    return result
