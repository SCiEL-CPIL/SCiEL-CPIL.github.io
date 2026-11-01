for r in range(2):
    for c in range(2):
        local_idx = r * 2 + c
        global_idx = r * 8 + (block * 2 + c)
        V[global_idx][offset + k] = V_s[local_idx][k]
        
        w_idx = (block * 2 + c) * 2 + r
        W[w_idx][offset + k] = W_s[local_idx][k]
