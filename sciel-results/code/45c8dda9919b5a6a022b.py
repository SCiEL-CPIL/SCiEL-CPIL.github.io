def _is_valid_mask(mask):
    for i in range(N):
        if not (mask >> i) & 1:
            continue
        a = i + 1
        for j in range(i + 1, N):
            if not (mask >> j) & 1:
                continue
            b = j + 1
            
            if i + j + 1 >= N:
                continue
            if not (mask >> (i + j + 1)) & 1:
                continue
            
            candidates = mask & (mask >> (i + 1)) & (mask >> (j + 1))
            if candidates & ~( (1 << i) | (1 << j) ):
                return False
    return True
