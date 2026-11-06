import time
import random
from itertools import combinations

def can_add(S, x):
    # S is a list of selected numbers
    # x is the candidate
    
    # 1. Check if x divides sum of any 3 in S
    if len(S) >= 3:
        for u, v, w in combinations(S, 3):
            if (u + v + w) % x == 0:
                return False
                
    # 2. Check if any a in S divides x + u + v
    # u, v are distinct elements from S, and u != a, v != a
    # We iterate over pairs (u, v) from S
    # For each pair, we check all a in S such that a != u and a != v
    for u, v in combinations(S, 2):
        s = x + u + v
        for a in S:
            if a == u or a == v:
                continue
            if s % a == 0:
                return False
    return True

def construct():
    start_time = time.time()
    best_set = []
    max_time = 85.0 # Safety margin
    
    # Try deterministic strategies first
    for order in ['desc', 'asc']:
        if time.time() - start_time > max_time:
            break
        if order == 'desc':
            candidates = list(range(192, 0, -1))
        else:
            candidates = list(range(1, 193))
        
        current_set = []
        for x in candidates:
            if can_add(current_set, x):
                current_set.append(x)
        
        if len(current_set) > len(best_set):
            best_set = current_set[:]
            
    # Random restarts
    while time.time() - start_time < max_time:
        candidates = list(range(1, 193))
        random.shuffle(candidates)
        
        current_set = []
        for x in candidates:
            if can_add(current_set, x):
                current_set.append(x)
        
        if len(current_set) > len(best_set):
            best_set = current_set[:]
            
    return best_set
