import time
import random
from itertools import combinations
from math import gcd

# Precompute GCDs
N = 128
gcd_mat = [[0] * (N + 1) for _ in range(N + 1)]
for i in range(1, N + 1):
    for j in range(i, N + 1):
        g = gcd(i, j)
        gcd_mat[i][j] = g
        gcd_mat[j][i] = g

def can_add(S, x):
    """Check if x can be added to set S without creating a bad triple."""
    # S is a set of ints
    # Check all pairs in S
    # Optimization: if S is small, this is fast.
    s_list = list(S)
    n = len(s_list)
    for i in range(n):
        a = s_list[i]
        ga_x = gcd_mat[a][x]
        for j in range(i + 1, n):
            b = s_list[j]
            ga_b = gcd_mat[a][b]
            gb_x = gcd_mat[b][x]
            if ga_b == ga_x == gb_x:
                return False
    return True

def is_valid(S):
    """Check if set S is valid."""
    s_list = list(S)
    n = len(s_list)
    for i in range(n):
        a = s_list[i]
        for j in range(i + 1, n):
            b = s_list[j]
            ga_b = gcd_mat[a][b]
            for k in range(j + 1, n):
                c = s_list[k]
                ga_c = gcd_mat[a][c]
                gb_c = gcd_mat[b][c]
                if ga_b == ga_c == gb_c:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin for 60s limit
    
    best_set = set()
    best_size = 0
    
    # Initial Greedy Phase
    # Try multiple random orderings to find a good start
    for _ in range(20):
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        current = set()
        for x in candidates:
            if can_add(current, x):
                current.add(x)
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = set(current)
            
        if time.time() > deadline:
            return list(best_set)
            
    # Simulated Annealing / Local Search Phase
    # We will try to improve best_set
    current_set = set(best_set)
    T = 1.0
    cooling_rate = 0.999
    
    # Helper to get random element from set
    def rand_elem(s):
        return random.choice(list(s))
        
    while time.time() < deadline:
        improved = False
        
        # 1. Try to add any element
        # Randomly shuffle candidates outside current_set
        outside = list(set(range(1, N + 1)) - current_set)
        random.shuffle(outside)
        
        added = False
        for x in outside:
            if can_add(current_set, x):
                current_set.add(x)
                added = True
                break # Add one at a time to allow re-evaluation? 
                       # Or add all possible? Adding all possible is greedy.
                       # Let's add all possible in a loop.
        
        # Actually, let's just do one add per iteration to keep it simple
        # and allow swapping if no add is possible.
        
        if not added:
            # Try a swap
            inside = list(current_set)
            if inside and outside:
                x = random.choice(inside)
                y = random.choice(outside)
                
                # Try swap: remove x, add y
                # Check if current_set - {x} + {y} is valid
                # We can check validity of the new set
                # Or check if removing x makes y addable?
                # No, removing x might break other triples? No, removing elements
                # never creates new violations.
                # So if y is addable to (current_set - {x}), the new set is valid.
                
                # Check can_add(current_set - {x}, y)
                # This is O(|S|^2)
                
                # Optimization: 
                # If y was not addable to current_set, it means there exists a pair (a,b) in current_set
                # such that gcd(a,b)=gcd(a,y)=gcd(b,y).
                # If we remove x, the pair (a,b) might still be there.
                # So we need to check if the "blocking" pairs involved x.
                
                # Simpler: just call can_add on the modified set.
                temp_set = set(current_set)
                temp_set.discard(x)
                
                if can_add(temp_set, y):
                    # Accept swap
                    current_set = temp_set
                    current_set.add(y)
                    improved = True
                    # Restart temperature or keep?
                    # Since size is same, we didn't improve size, but structure changed.
                    # We might be able to add more now.
                    
        if improved:
            # If we swapped, we might be able to add more in the next iteration
            continue
            
        # If no add and no successful swap (or swap failed), cool down
        T *= cooling_rate
        if T < 0.01:
            # Try random restarts?
            # Or just break
            pass
            
    # Final check
    if len(current_set) > best_size:
        best_set = current_set
        
    return list(best_set)
