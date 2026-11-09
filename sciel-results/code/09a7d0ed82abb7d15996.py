import random
from itertools import combinations
from math import gcd
import time

# Precompute: for each element x, which pairs (u,v) conflict with x
# i.e., gcd(u,v) = gcd(u,x) = gcd(v,x)
def compute_element_conflicts():
    """
    For each x, compute the set of pairs (u,v) such that {u,v,x} is a bad triple.
    """
    elem_conflicts = {x: set() for x in range(1, 129)}
    n = 128
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            g_ab = gcd(a, b)
            for c in range(1, n + 1):
                if c == a or c == b:
                    continue
                if gcd(a, c) == g_ab and gcd(b, c) == g_ab:
                    # {a,b,c} is bad
                    elem_conflicts[c].add((a, b))
                    elem_conflicts[a].add((b, c))
                    elem_conflicts[b].add((a, c))
    return elem_conflicts

ELEM_CONFLICTS = compute_element_conflicts()

def can_add_fast(x, current_pairs):
    """
    Check if x can be added given the set of pairs already in the set.
    current_pairs is a set of (u,v) tuples where u < v.
    """
    for pair in ELEM_CONFLICTS[x]:
        if pair in current_pairs:
            return False
    return True

def greedy_build(order):
    """
    Build a set by iterating through 'order' and adding if possible.
    """
    s = set()
    current_pairs = set()
    for x in order:
        if can_add_fast(x, current_pairs):
            # Add all pairs involving x
            for y in s:
                pair = (x, y) if x < y else (y, x)
                current_pairs.add(pair)
            s.add(x)
    return s

def local_search(s, deadline):
    """
    Try to improve set s by adding single elements or swapping.
    Returns the improved set.
    """
    # Precompute current pairs
    current_pairs = set()
    for u, v in combinations(s, 2):
        pair = (u, v) if u < v else (v, u)
        current_pairs.add(pair)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try to add any single element
        candidates = [x for x in range(1, 129) if x not in s]
        random.shuffle(candidates)
        for x in candidates:
            if can_add_fast(x, current_pairs):
                s.add(x)
                for y in s - {x}:
                    pair = (x, y) if x < y else (y, x)
                    current_pairs.add(pair)
                improved = True
                break
        
        if improved:
            continue
        
        # 2. Try to swap: remove one, add multiple
        list_s = list(s)
        random.shuffle(list_s)
        
        for u in list_s:
            # Remove u and its pairs
            s_without_u = s - {u}
            new_pairs = set()
            for a, b in combinations(s_without_u, 2):
                pair = (a, b) if a < b else (b, a)
                new_pairs.add(pair)
            
            new_s = set(s_without_u)
            addable = [x for x in range(1, 129) if x not in s]
            random.shuffle(addable)
            
            for x in addable:
                if can_add_fast(x, new_pairs):
                    new_s.add(x)
                    for y in new_s - {x}:
                        pair = (x, y) if x < y else (y, x)
                        new_pairs.add(pair)
            
            if len(new_s) > len(s):
                s = new_s
                current_pairs = new_pairs
                improved = True
                break
                
    return s

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5 second margin
    
    best_set = set()
    
    # Initial greedy with sorted order
    initial_order = list(range(1, 129))
    s = greedy_build(initial_order)
    s = local_search(s, deadline)
    if len(s) > len(best_set):
        best_set = s.copy()
        
    # Random restarts
    while time.time() < deadline:
        random_order = list(range(1, 129))
        random.shuffle(random_order)
        
        s = greedy_build(random_order)
        s = local_search(s, deadline)
        
        if len(s) > len(best_set):
            best_set = s.copy()
            
    return list(best_set)
