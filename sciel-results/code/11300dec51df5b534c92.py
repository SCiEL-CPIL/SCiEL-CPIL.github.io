import time

def construct():
    MOD = 1009
    N = 1008
    
    # Precompute modular inverses
    inv = [0] * (MOD + 1)
    for x in range(1, MOD):
        inv[x] = pow(x, MOD - 2, MOD)
        
    # Sliding window approach
    # We need an interval [L, R] such that for all x in [L, R], inv[x] is NOT in [L, R].
    # We can track the minimum L required for a given R.
    
    # Let's use the property: 
    # If we add element R to the window, we must ensure inv[R] is not in the window.
    # So L must be > inv[R] OR inv[R] must be < L.
    # Actually, if inv[R] is in [L, R-1], we have a conflict.
    # To resolve, we must move L to inv[R] + 1.
    # Also, we must ensure that for all existing elements x in [L, R-1], inv[x] is not in [L, R].
    # This implies L must be greater than all inv[x] for x in [L, R] where inv[x] >= L.
    # Wait, if inv[x] < L, it's fine. If inv[x] > R, it's fine. 
    # The conflict is only if L <= inv[x] <= R.
    # Since we are expanding R, new conflicts can only arise from the new element R (inv[R] in window) 
    # or potentially existing elements? 
    # If x was in the window, and inv[x] was not in the window, adding R doesn't change inv[x].
    # So the only new conflict is if inv[R] is in the current window [L, R-1].
    # Also, we must check if R is the inverse of some x already in the window? 
    # That is the same condition: if x in [L, R-1] such that inv[x] = R, then x = inv[R].
    # So checking if inv[R] is in [L, R-1] covers both "R is inverse of existing" and "existing is inverse of R".
    
    # However, we also have the self-inverse case. 
    # If x * x % 1009 == 1, then inv[x] = x.
    # If x is in the window, inv[x]=x is in the window. Conflict.
    # So x cannot be in the window.
    # The self-inverses are 1 and 1008.
    
    L = 1
    best_len = 0
    best_start = 1
    
    # To make it O(N), we can just iterate R and adjust L.
    # But simply checking "is inv[R] in [L, R-1]" is O(1) if we just check the value.
    # Wait, if inv[R] is in [L, R-1], we set L = inv[R] + 1.
    # Is that sufficient?
    # What if there was an earlier element X in [old_L, R-1] such that inv[X] = Y, and Y was < old_L, so it was fine.
    # Now we move L to a larger value. This only removes elements, it doesn't add conflicts.
    # Conflicts are only created by the NEW element R.
    # The new element R conflicts with:
    # 1. R itself (if inv[R] == R).
    # 2. Any X in the current window [L, R-1] such that inv[X] == R. 
    #    Note that if inv[X] == R, then X == inv[R].
    #    So we just need to check if inv[R] is in [L, R-1].
    
    # So the algorithm is:
    # For R from 1 to 1008:
    #   If inv[R] == R: # Self-inverse
    #       L = R + 1
    #       continue (window is empty or will be re-established)
    #   
    #   # Check if inv[R] is in the current window [L, R-1]
    #   if L <= inv[R] < R:
    #       L = inv[R] + 1
    #       
    #   # Now the window is [L, R].
    #   # We must also ensure that no element X in [L, R-1] has inv[X] in [L, R].
    #   # We established that the only new conflict is R.
    #   # Did we break any previous valid state? No, we only moved L forward.
    #   # So the window [L, R] is valid?
    #   # Let's verify. 
    #   # Before adding R, [L, R-1] was valid.
    #   # We moved L to L' to exclude inv[R].
    #   # New window [L', R].
    #   # For any X in [L', R-1], inv[X] was not in [L, R-1] (from previous validity).
    #   # Since L' >= L, inv[X] is not in [L', R-1].
    #   # Also, we ensured inv[R] is not in [L', R-1].
    #   # And inv[R] != R (handled above).
    #   # So [L', R] is valid.
    
    for R in range(1, N + 1):
        if inv[R] == R:
            L = R + 1
            continue
        
        # If inv[R] is in the current window [L, R-1], we must move L past it
        if L <= inv[R] < R:
            L = inv[R] + 1
            
        # The current valid window is [L, R]
        current_len = R - L + 1
        if current_len > best_len:
            best_len = current_len
            best_start = L
            
    return [best_start, best_len]
