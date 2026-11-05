L = 1
best_len = 0
best_start = 1
for R in range(1, 1009):
    if R == 1 or R == 1008:
        L = R + 1
        continue
    
    I = inv[R]
    if I >= L:
        L = I + 1
        
    if L <= R:
        current_len = R - L + 1
        if current_len > best_len:
            best_len = current_len
            best_start = L
