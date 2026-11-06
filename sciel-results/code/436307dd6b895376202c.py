    for right in range(1, N + 1):
        if right in self_inverse:
            left = right + 1
            continue
        
        r_inv = inv[right]
        
        if left <= r_inv < right:
            left = r_inv + 1
            
        current_length = right - left + 1
        if current_length > best_length:
            best_length = current_length
            best_start = left
