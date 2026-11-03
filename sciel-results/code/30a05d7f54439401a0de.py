    r_idx = 0
    for s in range(7):
        s_def = strassen_defs[s]
        # Get Strassen coefficients for A and B blocks
        # We need to map block indices to global indices.
        # For each BCLT term t:
        for t in range(23):
            # U3[i*3+k][t], V3[k*3+j][t], W3[j*3+i][t]
            # We need to iterate over all i, k, j in 0..2
            for i in range(3):
                for k in range(3):
                    for j in range(3):
                        u_coeff = U3[i*3+k][t]
                        v_coeff = V3[k*3+j][t]
                        w_coeff = W3[j*3+i][t]
                        
                        if u_coeff != 0.0 or v_coeff != 0.0 or w_coeff != 0.0:
                            # Calculate global indices
                            # This is inefficient if done for every i,k,j.
                            # But 3*3*3*23*7 = 4617 operations, which is fine.
                            
                            # U update
                            if u_coeff != 0.0:
                                for block_idx, coeff_a in s_def['A']:
                                    br = block_idx // 2
                                    bc = block_idx % 2
                                    i_glob = br * bs + i
                                    k_glob = bc * bs + k
                                    u_row = i_glob * m + k_glob
                                    U[u_row][r_idx] += coeff_a * u_coeff
                                    
                            # V update
                            if v_coeff != 0.0:
                                for block_idx, coeff_b in s_def['B']:
                                    br = block_idx // 2
                                    bc = block_idx % 2
                                    k_glob = br * bs + k
                                    j_glob = bc * bs + j
                                    v_row = k_glob * p + j_glob
                                    V[v_row][r_idx] += coeff_b * v_coeff
                                    
                            # W update
                            if w_coeff != 0.0:
                                for c_block_idx in range(4):
                                    gamma = 0.0
                                    for (s_idx, g) in recon_coeffs[c_block_idx]:
                                        if s_idx == s:
                                            gamma = g
                                            break
                                    if gamma != 0.0:
                                        br = c_block_idx // 2
                                        bc = c_block_idx % 2
                                        i_glob = br * bs + i
                                        j_glob = bc * bs + j
                                        w_row = j_glob * n + i_glob
                                        W[w_row][r_idx] += gamma * w_coeff
                                        
            r_idx += 1
