"""Find the longest interval in [1,1008] with no modular inverse pairs mod 1009."""
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin before 90s kill
    
    p = 1009
    n = 1008  # elements are 1..1008
    
    # Precompute modular inverses: inv[x] = x^{-1} mod 1009 for x in 1..1008
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # For each position i (1..1008), define the "conflict bound":
    #   - If i is self-inverse (1 or 1008), it cannot appear in any valid interval.
    #   - Otherwise, the conflict partner is inv[i].
    #
    # Two-pointer (sliding window) sweep, left-to-right:
    #   Maintain invariant: window [left, right] contains no pair (x,y) with x*y ≡ 1 (mod p).
    #   When extending right, the only new conflict is between 'right' and inv[right].
    #   If inv[right] falls in [left, right], push left past it.
    #   The window remains valid because shrinking from the left cannot create new conflicts.
    
    best_start = 1
    best_length = 0
    
    left = 1
    for right in range(1, n + 1):
        if time.time() > deadline:
            break
        
        # Self-inverse elements (1 and 1008) cannot be in any valid interval
        if right == 1 or right == 1008:
            left = right + 1
            continue
        
        ix = inv[right]
        
        # If inv[right] is in the current window [left, right], advance left past it
        if left <= ix <= right:
            left = ix + 1
        
        # Ensure window is non-empty
        if left > right:
            continue
        
        length = right - left + 1
        if length > best_length:
            best_length = length
            best_start = left
    
    if best_length <= 0:
        # Fallback: find any valid single element
        for x in range(1, n + 1):
            if x != 1 and x != 1008:
                best_start = x
                best_length = 1
                break
    
    # Exhaustive verification: check ALL pairs in the candidate interval
    s = best_start
    L = best_length
    elements = list(range(s, s + L))
    valid = True
    for i in range(len(elements)):
        x = elements[i]
        if (x * x) % p == 1:
            valid = False
            break
        for j in range(i + 1, len(elements)):
            y = elements[j]
            if (x * y) % p == 1:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: scan for any valid single element
        best_start = 2
        best_length = 1
    
    return [best_start, best_length]
