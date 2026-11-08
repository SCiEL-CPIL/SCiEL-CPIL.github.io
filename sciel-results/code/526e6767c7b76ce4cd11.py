<<<<<<< SEARCH
    # Fill the matrices for 3 blocks
    for b in range(3):
        base = b * 11
        for t_idx, term in enumerate(terms):
            col = base + t_idx
            
            # U
            for idx, val in term['u'].items():
                U[idx][col] = val
            
            # V
            for l_idx, val in term['v'].items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = j_local * p + (b * 2 + k_local)
                V[g_idx][col] = val
            
            # W
            for l_idx, val in term['w'].items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = (b * 2 + k_local) * n + i_local
                W[g_idx][col] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
=======
    # Fill the matrices for 3 blocks
    for b in range(3):
        base = b * 11
        for t_idx, term in enumerate(terms):
            col = base + t_idx
            
            # U
            for idx, val in term['u'].items():
                U[idx, col] = val
            
            # V
            for l_idx, val in term['v'].items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = j_local * p + (b * 2 + k_local)
                V[g_idx, col] = val
            
            # W
            for l_idx, val in term['w'].items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = (b * 2 + k_local) * n + i_local
                W[g_idx, col] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
