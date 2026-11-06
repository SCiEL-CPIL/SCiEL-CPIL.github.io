import math
import random
import time
from collections import defaultdict

def construct():
    N = 128
    deadline = time.time() + 80.0  # 10s safety margin
    
    # Precompute LCM table
    lcm = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(1, N + 1):
            lcm[i][j] = i * j // math.gcd(i, j)
    
    # Precompute for each x, the set of pairs (a,b) with a<b, a!=x, b!=x
    # such that lcm(a,b) == lcm(a,x) == lcm(b,x)
    bad_pairs = [set() for _ in range(N + 1)]
    for x in range(1, N + 1):
        for a in range(1, N + 1):
            if a == x:
                continue
            for b in range(a + 1, N + 1):
                if b == x:
                    continue
                if lcm[a][b] == lcm[a][x] == lcm[b][x]:
                    bad_pairs[x].add((a, b))
    
    def can_add(selected_set, x):
        """Check if x can be added to selected_set without forming a bad triple."""
        for (a, b) in bad_pairs[x]:
            if a in selected_set and b in selected_set:
                return False
        return True
    
    def greedy_construct(order, selected_set=None):
        if selected_set is None:
            selected_set = set()
        for x in order:
            if x not in selected_set and can_add(selected_set, x):
                selected_set.add(x)
        return selected_set
    
    # Multiple greedy starts with different orderings
    best_set = set()
    best_size = 0
    
    # 1. Ascending
    s = greedy_construct(range(1, N + 1))
    if len(s) > best_size:
        best_size = len(s)
        best_set = set(s)
    
    # 2. Descending
    s = greedy_construct(range(N, 0, -1))
    if len(s) > best_size:
        best_size = len(s)
        best_set = set(s)
    
    # 3. Randomized greedy starts
    rng = random.Random(42)
    for trial in range(200):
        if time.time() > deadline:
            break
        order = list(range(1, N + 1))
        rng.shuffle(order)
        s = greedy_construct(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = set(s)
    
    # Local search: try to add elements, then try swaps
    current_set = best_set
    current_size = len(current_set)
    
    all_nums = set(range(1, N + 1))
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Try adding any missing element
        missing = list(all_nums - current_set)
        rng.shuffle(missing)
        for v in missing:
            if can_add(current_set, v):
                current_set.add(v)
                current_size += 1
                improved = True
                break
        
        if improved:
            continue
        
        # Try 1-swap: remove one, try to add more
        removal_candidates = list(current_set)
        rng.shuffle(removal_candidates)
        for u in removal_candidates:
            if time.time() > deadline:
                break
            
            temp_set = current_set - {u}
            added = 0
            new_elems = []
            missing_for_temp = list(all_nums - temp_set)
            rng.shuffle(missing_for_temp)
            
            for v in missing_for_temp:
                if v == u:
                    continue
                if can_add(temp_set, v):
                    temp_set.add(v)
                    added += 1
                    new_elems.append(v)
            
            if added > 1:
                current_set = temp_set
                current_size = len(current_set)
                improved = True
                break
    
    if not current_set:
        current_set = {1}
    
    return list(current_set)
