"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    p = 1009
    inv = [0] * p
    for i in range(1, p):
        inv[i] = pow(i, p - 2, p)
    
    self_inv = {1, p - 1}
    
    best_len = 0
    best_start = 1
    
    e = 0
    for s in range(1, p):
        if p - s <= best_len:
            break
        if e < s:
            e = s - 1
        while e + 1 < p:
            e_next = e + 1
            if e_next in self_inv:
                break
            inv_e = inv[e_next]
            if s <= inv_e <= e_next - 1:
                break
            e = e_next
        current_len = e - s + 1
        if current_len > best_len:
            best_len = current_len
            best_start = s
    
    if best_len == 0:
        return [2, 1]
    
    return [best_start, best_len]
# EVOLVE-BLOCK-END
