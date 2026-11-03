def _expand_decomposition(decomp_232, n, m, p, block_size):
    """
    Expand a 2x3x2 decomposition to 2x3x6 by repeating it.
    """
    U_232 = np.array(decomp_232["U"])
    V_232 = np.array(decomp_232["V"])
    W_232 = np.array(decomp_232["W"])
    
    r_232 = U_232.shape[1]
    num_blocks = p // block_size
    r_total = r_232 * num_blocks
    
    U_full = np.zeros((n * m, r_total))
    V_full = np.zeros((m * p, r_total))
    W_full = np.zeros((p * n, r_total))
    
    for block in range(num_blocks):
        offset = block * r_232
        
        # Copy U
        U_full[:, offset:offset + r_232] = U_232
        
        # Copy V (shift rows)
        for j in range(m):
            V_full[j * p + block * block_size:(j * p + (block + 1) * block_size), offset:offset + r_232] = \
                V_232[j * block_size:(j + 1) * block_size, :]
        
        # Copy W (shift rows)
        for i in range(n):
            for k in range(block_size):
                W_full[(block * block_size + k) * n + i, offset:offset + r_232] = W_232[k * n + i, :]
    
    # Verify the expanded decomposition
    if _verify_decomposition_fast(U_full, V_full, W_full, n, m, p):
        return {"U": U_full.tolist(), "V": V_full.tolist(), "W": W_full.tolist()}
    
    return None
