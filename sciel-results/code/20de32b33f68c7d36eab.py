    next_val = e + 1
    inv_next = inv[next_val]
    
    # Check if partner is already in window
    if in_window[inv_next]:
        break
        
    # Check if self-inverse
    if (next_val * next_val) % MOD == 1:
        break
        
    in_window[next_val] = True
    e = next_val
    