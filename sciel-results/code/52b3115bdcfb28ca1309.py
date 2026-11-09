import random
from itertools import combinations
from math import gcd
import time

N = 128

def compute_bad_triples():
    """Precompute all bad triples as a set of sorted tuples."""
    bad = set()
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = gcd(a, b)
            for c in range(b + 1, N + 1):
                if gcd(a, c) == g_ab and gcd(b, c) == g_ab:
                    bad.add((a, b, c))
    return bad

BAD_TRIPLES = compute_bad_triples()

def can_add(x, current_set):
    """Check if x can be added to current_set without forming a bad triple."""
    # For each pair (u, v) in current_set, check if {x, u, v} is a bad triple
    # Optimize: iterate through elements and check pairs
    elems = list(current_set)
    for i in range(len(elems)):
        for j in range(i + 1, len(elems)):
            u, v = elems[i], elems[j]
            # Create sorted triple for lookup
            triple = tuple(sorted((x, u, v)))
            if triple in BAD_TRIPLES:
                return False
    return True

def greedy_build(order):
    """Build a set by iterating through order and adding if possible."""
    s = set()
    for x in order:
        if can_add(x, s):
            s.add(x)
    return s

def local_search(s, deadline):
    """Try to improve set s by adding elements or swapping."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # 1. Try to add any single element
        candidates = [x for x in range(1, N + 1) if x not in s]
        random.shuffle(candidates)
        for x in candidates:
            if can_add(x, s):
                s.add(x)
                improved = True
                break
        
        if improved:
            continue
        
        # 2. Try swaps: remove one element, try to add multiple
        list_s = list(s)
        random.shuffle(list_s)
        
        for u in list_s:
            s_without_u = s - {u}
            new_s = set(s_without_u)
            
            addable = [x for x in range(1, N + 1) if x not in s]
            random.shuffle(addable)
            
            for x in addable:
                if can_add(x, new_s):
                    new_s.add(x)
            
            if len(new_s) > len(s):
                s = new_s
                improved = True
                break
                
    return s

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit
    
    best_set = set()
    
    # Initial greedy with sorted order
    initial_order = list(range(1, N + 1))
    s = greedy_build(initial_order)
    s = local_search(s, deadline)
    if len(s) > len(best_set):
        best_set = s.copy()
    
    # Multiple random restarts
    while time.time() < deadline:
        random_order = list(range(1, N + 1))
        random.shuffle(random_order)
        
        s = greedy_build(random_order)
        s = local_search(s, deadline)
        
        if len(s) > len(best_set):
            best_set = s.copy()
    
    return list(best_set)
