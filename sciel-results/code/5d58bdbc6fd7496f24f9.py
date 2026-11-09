import time
from collections import deque

def construct():
    p = 1009
    inv = [0] * p
    for i in range(1, p):
        inv[i] = pow(i, p - 2, p)
    
    # A[i] = inv[i] if inv[i] < i else 0
    # B[i] = inv[i] if inv[i] > i else p (infinity for min)
    A = [0] * p
    B = [p] * p
    for i in range(1, p):
        if inv[i] < i:
            A[i] = inv[i]
        elif inv[i] > i:
            B[i] = inv[i]
        # else inv[i] == i (1, 1008), A=0, B=p (handled by validity check? 
        # If i=1, A[1]=0, B[1]=p. 
        # Window [1,1]: max(A)=0 < 1 (True). min(B)=p > 1 (True).
        # But 1 is self-inverse, so [1,1] should be invalid.
        # My condition: max(A[s..e]) < s AND min(B[s..e]) > e.
        # For [1,1]: 0 < 1 is True. p > 1 is True.
        # So it says valid. But it is NOT valid.
        # Why?
        # Because for x=1, inv[1]=1. 
        # Condition 1: inv[x] < s. 1 < 1 is False.
        # Condition 2: inv[x] > e. 1 > 1 is False.
        # So it fails both?
        # My derivation:
        # If inv[x] < x, we need inv[x] < s.
        # If inv[x] > x, we need inv[x] > e.
        # If inv[x] == x, it's in [s,e] always. So it's always invalid.
        # So we must explicitly check for self-inverses.
        
    # Let's refine A and B to handle self-inverses.
    # If i is self-inverse, mark it as invalid.
    # We can just check if any self-inverse is in the window.
    
    # Simpler: 
    # Valid if:
    # 1. No self-inverse in [s,e].
    # 2. max(inv[x] for x in [s,e] if inv[x] < x) < s
    # 3. min(inv[x] for x in [s,e] if inv[x] > x) > e
    
    self_inv = {1, p-1}
    
    def is_valid(s, e):
        # Check self-inverses
        if 1 >= s and 1 <= e: return False
        if p-1 >= s and p-1 <= e: return False
        
        max_a = 0
        min_b = p
        for x in range(s, e+1):
            if inv[x] < x:
                if inv[x] > max_a: max_a = inv[x]
            elif inv[x] > x:
                if inv[x] < min_b: min_b = inv[x]
                
        return max_a < s and min_b > e

    # Binary search for max length
    lo, hi = 1, p-1
    best_len = 0
    best_start = 1
    
    # Since N is small, we can just iterate L from p-1 down to 1
    # and return the first valid one.
    # This is O(N^2) worst case, but likely fast.
    # Or binary search.
    
    # Let's use the original O(N^2) loop but with the explicit is_valid check?
    # No, the original loop was fine.
    
    # I will stick to the original logic but add a time guard.
    start_time = time.time()
    
    best_len = 0
    best_start = 1
    
    for s in range(1, p):
        if p - 1 - s + 1 <= best_len:
            break
        for e in range(s, p):
            if time.time() - start_time > 50:
                return [best_start, best_len]
            
            if e in self_inv:
                break
            
            if s <= inv[e] <= e - 1:
                break
            
            current_len = e - s + 1
            if current_len > best_len:
                best_len = current_len
                best_start = s
                
    return [best_start, best_len]
