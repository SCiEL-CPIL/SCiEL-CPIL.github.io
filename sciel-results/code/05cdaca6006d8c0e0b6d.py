import time
import random
from itertools import combinations
from collections import defaultdict
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 135  # Safety margin: evaluator kills at 150s
    n = 16
    all_triples = list(combinations(range(n), 3))
    triple_set = set(all_triples)
    num_triples = len(all_triples)  # 560
    
    # Precompute for each triple, the list of "partner triples" that would complete a K(2,2,2)
    # For triple T = (x, y, z), a partner triple P = (xp, yp, zp) completes a K(2,2,2)
    # where the 3 pairs are {x,xp}, {y,yp}, {z,zp}
    # The 8 transversal triples are all combinations of picking one from each pair.
    # T is one of them. The other 7 are the "forbidden" triples that must all be present.
    
    # For efficiency, precompute for each triple the list of (partner_triple, [other_7_triples])
    # But that's too much memory. Instead, for each triple, precompute the list of partner triples.
    # When checking if we can add T, we iterate over partners and check if the other 7 are present.
    
    # Actually, let's think about this differently.
    # For a given triple T = (x, y, z), the K(2,2,2)s it belongs to are determined by
    # choosing xp from {0..15}\{x,y,z}, yp from {0..15}\{x,y,z,xp}, zp from {0..15}\{x,y,z,xp,yp}
    # For each such choice, the 7 other transversal triples are determined.
    
    # Let's precompute: for each triple, a list of tuples (xp, yp, zp, other_7_set)
    # But other_7_set is 7 elements, and there are 13*12*11 = 1716 such tuples per triple.
    # 560 * 1716 = ~960K entries, each with 7 ints. That's manageable.
    
    # Actually, let's be smarter. For the greedy algorithm, we need to check:
    # "Can I add triple T?" -> "For any partner (xp,yp,zp), are all 7 other transversal triples in my set?"
    
    # Let's precompute for each triple T, a list of (xp, yp, zp, [t1, t2, t3, t4, t5, t6, t7])
    # where t1..t7 are the 7 other transversal triples.
    
    triple_to_partners = [None] * num_triples
    triple_index = {t: i for i, t in enumerate(all_triples)}
    
    for i, T in enumerate(all_triples):
        x, y, z = T
        others = [p for p in range(n) if p not in (x, y, z)]
        partners = []
        for xp in others:
            for yp in others:
                if yp == xp:
                    continue
                for zp in others:
                    if zp == xp or zp == yp:
                        continue
                    # 7 other transversal triples
                    t1 = tuple(sorted((xp, y, z)))
                    t2 = tuple(sorted((x, yp, z)))
                    t3 = tuple(sorted((x, y, zp)))
                    t4 = tuple(sorted((xp, yp, z)))
                    t5 = tuple(sorted((xp, y, zp)))
                    t6 = tuple(sorted((x, yp, zp)))
                    t7 = tuple(sorted((xp, yp, zp)))
                    partners.append((xp, yp, zp, (t1, t2, t3, t4, t5, t6, t7)))
        triple_to_partners[i] = partners
    
    # Greedy with conflict tracking
    # Maintain: for each K(2,2,2), how many of its 8 triples are selected.
    # A triple T can be added if for every K(2,2,2) containing T, the count is < 7
    # (i.e., adding T won't make it 8).
    
    # But enumerating all K(2,2,2)s and tracking counts is complex.
    # Alternative: just check directly using the partner list.
    
    def can_add(triple_idx, selected_set):
        T = all_triples[triple_idx]
        partners = triple_to_partners[triple_idx]
        for xp, yp, zp, other7 in partners:
            # Check if all 7 are in selected_set
            all_present = True
            for t in other7:
                if t not in selected_set:
                    all_present = False
                    break
            if all_present:
                return False
        return True
    
    best_triples = []
    best_count = 0
    
    # Multiple random greedy passes
    num_attempts = 30
    attempt = 0
    while attempt < num_attempts:
        if time.time() > deadline - 20:
            break
        
        # Random order with some structure: shuffle
        order = list(range(num_triples))
        random.shuffle(order)
        
        selected = set()
        selected_indices = []
        
        for idx in order:
            if time.time() > deadline - 15:
                break
            if can_add(idx, selected):
                selected.add(all_triples[idx])
                selected_indices.append(idx)
        
        if len(selected_indices) > best_count:
            best_count = len(selected_indices)
            best_triples = [list(all_triples[i]) for i in selected_indices]
        
        attempt += 1
    
    # Local search: try to add any triple that can now be added
    if time.time() < deadline - 10:
        selected = set(tuple(t) for t in best_triples)
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            for idx in range(num_triples):
                if time.time() > deadline - 10:
                    break
                T = all_triples[idx]
                if T not in selected and can_add(idx, selected):
                    selected.add(T)
                    best_triples = [list(t) for t in selected]
                    best_count = len(selected)
                    improved = True
                    break
    
    # Local search: try 1-for-1 swaps
    if time.time() < deadline - 5:
        selected = set(tuple(t) for t in best_triples)
        for _ in range(20):  # Limit iterations
            if time.time() > deadline - 5:
                break
            improved = False
            for remove_idx in range(num_triples):
                if time.time() > deadline - 5:
                    break
                T_remove = all_triples[remove_idx]
                if T_remove not in selected:
                    continue
                # Try removing T_remove and adding a different triple
                selected.remove(T_remove)
                for add_idx in range(num_triples):
                    if time.time() > deadline - 5:
                        break
                    T_add = all_triples[add_idx]
                    if T_add in selected:
                        continue
                    if can_add(add_idx, selected):
                        selected.add(T_add)
                        best_triples = [list(t) for t in selected]
                        improved = True
                        break
                if improved:
                    break
                # If we didn't add anything, put T_remove back
                if T_add != T_remove:
                    selected.add(T_remove)
            if not improved:
                break
    
    return best_triples
