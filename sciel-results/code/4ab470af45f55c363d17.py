<<<<<<< SEARCH
    col = 0
    for bi in range(3):  # block row of A
        for bk in range(3):  # block col of A = block row of B
            for bj in range(3):  # block col of B
                for t in range(7):
                    u_offsets, u_coeffs = strassen_terms[t][0]
                    v_offsets, v_coeffs = strassen_terms[t][1]
                    
                    # Set U factors
                    for off, coeff in zip(u_offsets, u_coeffs):
                        r, c = off // 2, off % 2
                        global_row = 2 * bi + r
                        global_col = 2 * bk + c
                        U[global_row * 6 + global_col][col] += coeff
                    
                    # Set V factors
                    for off, coeff in zip(v_offsets, v_coeffs):
                        r, c = off // 2, off % 2
                        global_row = 2 * bk + r
                        global_col = 2 * bj + c
                        V[global_row * 6 + global_col][col] += coeff
                    
                    # Set W factors
                    for c_off, coeff in c_contributions[t]:
                        r, c = c_off // 2, c_off % 2
                        global_row = 2 * bj + c
                        global_col = 2 * bi + r
                        W[global_row * 6 + global_col][col] += coeff
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
=======
    col = 0
    for bi in range(3):  # block row of A
        for bk in range(3):  # block col of A = block row of B
            for bj in range(3):  # block col of B
                for t in range(7):
                    if time.time() > _DEADLINE:
                        return {"U": U, "V": V, "W": W}
                    u_offsets, u_coeffs = strassen_terms[t][0]
                    v_offsets, v_coeffs = strassen_terms[t][1]
                    
                    # Set U factors
                    for off, coeff in zip(u_offsets, u_coeffs):
                        r, c = off // 2, off % 2
                        global_row = 2 * bi + r
                        global_col = 2 * bk + c
                        U[global_row * 6 + global_col][col] += coeff
                    
                    # Set V factors
                    for off, coeff in zip(v_offsets, v_coeffs):
                        r, c = off // 2, off % 2
                        global_row = 2 * bk + r
                        global_col = 2 * bj + c
                        V[global_row * 6 + global_col][col] += coeff
                    
                    # Set W factors
                    for c_off, coeff in c_contributions[t]:
                        r, c = c_off // 2, c_off % 2
                        global_row = 2 * bj + c
                        global_col = 2 * bi + r
                        W[global_row * 6 + global_col][col] += coeff
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
