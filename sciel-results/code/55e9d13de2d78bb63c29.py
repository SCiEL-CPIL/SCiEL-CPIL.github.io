"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations
import numpy as np

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    n = 16
    all_triples = list(combinations(range(n), 3))
    num_triples = len(all_triples)  # 560
    
    # Enumerate all K(2,2,2)s and build incidence structure
    triple_to_k222s = {t: [] for t in all_triples}
    k222_to_triples = []  # list of lists of 8 triple indices
    
    triple_index = {t: i for i, t in enumerate(all_triples)}
    
    for six_points in combinations(range(n), 6):
        a, b, c, d, e, f = six_points
        matchings = [
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
        for m in matchings:
            (p1, p2), (q1, q2), (r1, r2) = m
            triples_in_k222 = [
                tuple(sorted((p1, q1, r1))),
                tuple(sorted((p1, q1, r2))),
                tuple(sorted((p1, q2, r1))),
                tuple(sorted((p1, q2, r2))),
                tuple(sorted((p2, q1, r1))),
                tuple(sorted((p2, q1, r2))),
                tuple(sorted((p2, q2, r1))),
                tuple(sorted((p2, q2, r2))),
            ]
            if len(set(triples_in_k222)) == 8:
                k222_idx = len(k222_to_triples)
                triple_indices = [triple_index[t] for t in triples_in_k222]
                k222_to_triples.append(triple_indices)
                for ti in triple_indices:
                    triple_to_k222s[all_triples[ti]].append(k222_idx)
    
    num_k222s = len(k222_to_triples)
    
    # Precompute for each triple, the list of K(2,2,2) indices it belongs to
    triple_k222_list = [triple_to_k222s[all_triples[i]] for i in range(num_triples)]
    
    # Build incidence matrix as numpy array for fast operations
    # inc_matrix[i, j] = 1 if triple i is in k222 j
    inc_matrix = np.zeros((num_triples, num_k222s), dtype=np.int8)
    for i in range(num_triples):
        for kid in triple_k222_list[i]:
            inc_matrix[i, kid] = 1
    
    # Precompute: for each k222, which triples are in it (as a set for fast lookup)
    k222_triple_sets = [set(kids) for kids in k222_to_triples]
    
    # Greedy initial solution with multiple restarts
    best_count = 0
    best_selected = set()
    
    def greedy_random(seed, strategy):
        random.seed(seed)
        k222_count = np.zeros(num_k222s, dtype=np.int32)
        selected = set()
        
        if strategy == 0:
            order = list(range(num_triples))
            random.shuffle(order)
        elif strategy == 1:
            # Fewer K(2,2,2) memberships first (less restrictive)
            order = sorted(range(num_triples), key=lambda i: len(triple_k222_list[i]))
            random.shuffle(order)
        elif strategy == 2:
            # More K(2,2,2) memberships first (more restrictive)
            order = sorted(range(num_triples), key=lambda i: -len(triple_k222_list[i]))
            random.shuffle(order)
        else:
            # Random weighted by inverse degree
            weights = [1.0 / (len(triple_k222_list[i]) + 1) for i in range(num_triples)]
            order = list(range(num_triples))
            random.shuffle(order)
        
        for ti in order:
            # Check if adding ti would violate any constraint
            # ti is in k222s triple_k222_list[ti]
            # We need k222_count[kid] < 7 for all kid in triple_k222_list[ti]
            kids = triple_k222_list[ti]
            if not kids:
                selected.add(ti)
                continue
            # Fast check using numpy
            if np.all(k222_count[kids] < 7):
                selected.add(ti)
                k222_count[kids] += 1
        
        return selected, k222_count
    
    # Multiple restarts with different strategies
    num_restarts = 80
    for restart in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        strategy = restart % 4
        seed = restart * 7919 + 12345
        selected, k222_count = greedy_random(seed, strategy)
        
        if len(selected) > best_count:
            best_count = len(selected)
            best_selected = set(selected)
    
    # Local improvement: aggressive add loop with randomization
    search_deadline = deadline - 5
    improved_any = True
    max_iterations = 500
    iteration = 0
    while improved_any and time.time() < search_deadline and iteration < max_iterations:
        iteration += 1
        improved_any = False
        
        # Recompute k222 counts from best_selected
        k222_count = np.zeros(num_k222s, dtype=np.int32)
        for ti in best_selected:
            kids = triple_k222_list[ti]
            if kids:
                k222_count[kids] += 1
        
        # Try adding unselected triples in random order
        unselected = [i for i in range(num_triples) if i not in best_selected]
        random.shuffle(unselected)
        
        for ti in unselected:
            if time.time() > search_deadline:
                break
            kids = triple_k222_list[ti]
            if not kids:
                best_selected.add(ti)
                best_count += 1
                improved_any = True
                continue
            if np.all(k222_count[kids] < 7):
                best_selected.add(ti)
                best_count += 1
                k222_count[kids] += 1
                improved_any = True
        
        # If no add was possible, try swaps
        if not improved_any and best_count > 0:
            # Try 1-for-1 swaps
            selected_list = list(best_selected)
            random.shuffle(selected_list)
            unselected = [i for i in range(num_triples) if i not in best_selected]
            random.shuffle(unselected)
            
            swap_found = False
            num_swap_attempts = min(200, len(selected_list))
            
            for si in range(num_swap_attempts):
                if time.time() > search_deadline:
                    break
                
                rem = selected_list[si % len(selected_list)]
                
                # Temporarily remove
                best_selected.discard(rem)
                rem_kids = triple_k222_list[rem]
                if rem_kids:
                    k222_count[rem_kids] -= 1
                
                # Try to add one unselected triple that was previously blocked
                added = None
                for add in unselected:
                    add_kids = triple_k222_list[add]
                    if not add_kids or np.all(k222_count[add_kids] < 7):
                        added = add
                        break
                
                if added is not None:
                    best_selected.add(added)
                    add_kids = triple_k222_list[added]
                    if add_kids:
                        k222_count[add_kids] += 1
                    if added in unselected:
                        unselected.remove(added)
                    improved_any = True
                    swap_found = True
                    break
                else:
                    # Revert
                    best_selected.add(rem)
                    if rem_kids:
                        k222_count[rem_kids] += 1
            
            # Try 1-for-2 swaps: remove one, add two
            if not swap_found:
                for si in range(min(100, len(selected_list))):
                    if time.time() > search_deadline:
                        break
                    
                    rem = selected_list[si % len(selected_list)]
                    rem_kids = triple_k222_list[rem]
                    
                    # Temporarily remove
                    best_selected.discard(rem)
                    if rem_kids:
                        k222_count[rem_kids] -= 1
                    
                    # Try to add two unselected triples
                    added_list = []
                    for add in unselected:
                        if len(added_list) >= 2:
                            break
                        add_kids = triple_k222_list[add]
                        if not add_kids or np.all(k222_count[add_kids] < 7):
                            best_selected.add(add)
                            if add_kids:
                                k222_count[add_kids] += 1
                            added_list.append(add)
                            if add in unselected:
                                unselected.remove(add)
                    
                    if len(added_list) >= 2:
                        best_count += len(added_list) - 1
                        improved_any = True
                        swap_found = True
                        break
                    else:
                        # Revert
                        best_selected.add(rem)
                        if rem_kids:
                            k222_count[rem_kids] += 1
                        for add in added_list:
                            best_selected.discard(add)
                            add_kids = triple_k222_list[add]
                            if add_kids:
                                k222_count[add_kids] -= 1
                            if add not in unselected:
                                unselected.append(add)
            
            if not swap_found:
                # Try 2-for-1 swaps: remove two, add one (rarely helpful but try)
                # Or try a fresh random restart if time allows
                if time.time() < deadline - 10:
                    seed = int(time.time() * 1000000) % (2**31)
                    strategy = random.randint(0, 3)
                    selected2, k222_count2 = greedy_random(seed, strategy)
                    
                    if len(selected2) > best_count:
                        best_count = len(selected2)
                        best_selected = set(selected2)
                        improved_any = True
                else:
                    break

    final_selected = best_selected
    if not final_selected:
        final_selected = set(range(num_triples))
    
    result = [list(all_triples[i]) for i in sorted(final_selected)]
    return result
# EVOLVE-BLOCK-END
