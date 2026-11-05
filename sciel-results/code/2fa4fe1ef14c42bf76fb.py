def construct():
    MOD = 1009
    N = 1008
    inv = [0] * (MOD + 1)
    for i in range(1, MOD):
        inv[i] = pow(i, MOD - 2, MOD)
    
    best_len = 0
    best_start = 1
    r = 0
    
    for l in range(1, N + 1):
        # Ensure r is at least l-1
        if r < l - 1:
            r = l - 1
        
        # Try to extend r
        while r + 1 <= N:
            next_r = r + 1
            # Check if next_r can be added to [l, r]
            # Condition 1: next_r is not self-inverse
            if next_r == 1 or next_r == N:
                break
            # Condition 2: inv[next_r] is not in [l, r]
            # Note: inv[next_r] == next_r is handled by Condition 1
            if l <= inv[next_r] <= r:
                break
            # If both conditions pass, we can add next_r
            r = next_r
        
        current_len = r - l + 1
        if current_len > best_len:
            best_len = current_len
            best_start = l
            
    return [best_start, best_len]
