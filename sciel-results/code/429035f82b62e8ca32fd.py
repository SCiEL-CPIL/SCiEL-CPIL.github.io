"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    n = 1008
    
    # Precompute modular inverses
    inv = [0] * (p)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
        
    # Self-inverse elements are x such that x*x % p == 1
    # These are 1 and p-1 (1008)
    self_inverses = {1, 1008}
    
    best_len = 0
    best_start = 1
    
    # Sliding window
    # Window is [left, right]
    # We want to maximize right - left + 1
    # such that for all x, y in [left, right], x*y % p != 1
    
    seen = set()
    right = 0
    
    for left in range(1, n + 1):
        # Expand right as far as possible
        while right < n:
            candidate = right + 1
            
            # Check if candidate is a self-inverse
            if candidate in self_inverses:
                break
                
            # Check if candidate's inverse is already in the window
            # If inv[candidate] is in seen, then candidate * inv[candidate] % p == 1
            if inv[candidate] in seen:
                break
                
            # It's safe to add candidate
            seen.add(candidate)
            right += 1
            
        # Now [left, right] is the maximal valid window starting at left
        current_len = right - left + 1
        if current_len > best_len:
            best_len = current_len
            best_start = left
            
        # Shrink window from left
        if left in seen:
            seen.remove(left)
            
    return [best_start, best_len]
# EVOLVE-BLOCK-END
