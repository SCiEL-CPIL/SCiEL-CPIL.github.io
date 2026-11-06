import time
from itertools import combinations
from math import gcd
import random

def _lcm(a, b):
    return a // gcd(a, b) * b

def _precompute_bad_triples():
    bad = set()
    for a in range(1, 129):
        for b in range(a+1, 129):
            lab = _lcm(a, b)
            for c in range(b+1, 129):
                if _lcm(a, c) == lab and _lcm(b, c) == lab:
                    bad.add((a, b, c))
    return bad

def _can_add(candidate, selected, bad_triples):
    for a, b in combinations(selected, 2):
        triple = tuple(sorted((a, b, candidate)))
        if triple in bad_triples:
            return False
    return True

def _greedy(order, bad_triples):
    selected = set()
    for x in order:
        if _can_add(x, selected, bad_triples):
            selected.add(x)
    return selected

def _local_search(selected, bad_triples, deadline):
    selected = set(selected)
    all_elements = set(range(1, 129))
    
    # Phase 1: Add all possible
    changed = True
    while changed:
        changed = False
        for y in all_elements - selected:
            if _can_add(y, selected, bad_triples):
                selected.add(y)
                changed = True
                
    # Phase 2: Swap to enable additions
    improved = True
    while improved and time.time() < deadline:
        improved = False
        selected_list = list(selected)
        random.shuffle(selected_list)
        
        for x in selected_list:
            if time.time() > deadline:
                break
            remaining = all_elements - selected
            remaining_list = list(remaining)
            random.shuffle(remaining_list)
            
            for y in remaining_list:
                new_selected = (selected - {x}) | {y}
                
                # Check validity of y in new_selected
                valid_swap = True
                # Only check pairs involving y? No, y must not form bad triple with any pair in new_selected - {y}
                # Which is selected - {x}
                for a, b in combinations(new_selected - {y}, 2):
                    triple = tuple(sorted((a, b, y)))
                    if triple in bad_triples:
                        valid_swap = False
                        break
                if not valid_swap:
                    continue
                
                # Check if we can add any z
                added = False
                for z in all_elements - new_selected:
                    if _can_add(z, new_selected, bad_triples):
                        selected = new_selected | {z}
                        improved = True
                        added = True
                        break
                if added:
                    break
            if improved:
                break
    return list(selected)

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    bad_triples = _precompute_bad_triples()
    
    if time.time() > deadline:
        return [1]
        
    best_size = 0
    best_set = set()
    
    # 1. Deterministic Greedy
    degree = {i: 0 for i in range(1, 129)}
    for a, b, c in bad_triples:
        degree[a] += 1
        degree[b] += 1
        degree[c] += 1
        
    order = sorted(range(1, 129), key=lambda x: degree[x])
    s1 = _greedy(order, bad_triples)
    if len(s1) > best_size:
        best_size = len(s1)
        best_set = set(s1)
        
    # 2. Random Restarts
    candidates = [best_set]
    
    # Run random greeds
    # Estimate: 100 restarts * 1ms = 100ms. Safe.
    for i in range(100):
        if time.time() > deadline - 10: # Leave 10s for local search
            break
        order = list(range(1, 129))
        random.shuffle(order)
        s = _greedy(order, bad_triples)
        if len(s) > best_size:
            best_size = len(s)
            best_set = set(s)
        candidates.append(set(s))
        
    # Sort candidates by size
    candidates = sorted(candidates, key=len, reverse=True)
    
    # 3. Local Search on top candidates
    # We have ~10s left. Each local search takes ~1-2s.
    # So we can do ~5-10 local searches.
    
    for s in candidates[:10]:
        if time.time() > deadline - 2:
            break
        improved_set = _local_search(s, bad_triples, deadline)
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_set = set(improved_set)
            
    return list(best_set)
