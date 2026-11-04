def _strassen_4x4_terms():
    """Generate the 49-term Strassen decomposition for 4x4 matrix multiplication."""
    strassen_2x2 = _strassen_2x2()
    terms = []
    
    for l1_term in strassen_2x2:
        u_blocks, u_block_coeffs, v_blocks, v_block_coeffs, w_blocks, w_block_coeffs = l1_term
        
        # Build 2x2 block coefficient matrices
        u_block_mat = [[0]*2 for _ in range(2)]
        for li, coeff in zip(u_blocks, u_block_coeffs):
            br = li // 2
            bc = li % 2
            u_block_mat[br][bc] += coeff
        
        v_block_mat = [[0]*2 for _ in range(2)]
        for li, coeff in zip(v_blocks, v_block_coeffs):
            br = li // 2
            bc = li % 2
            v_block_mat[br][bc] += coeff
        
        w_block_mat = [[0]*2 for _ in range(2)]
        for li, coeff in zip(w_blocks, w_block_coeffs):
            br = li // 2
            bc = li % 2
            w_block_mat[br][bc] += coeff
        
        # Apply level-2 Strassen to the 2x2 matrix product
        for l2_term in strassen_2x2:
            u_scalars, u_scalar_coeffs, v_scalars, v_scalar_coeffs, w_scalars, w_scalar_coeffs = l2_term
            
            u_dict = {}
            for si, sc in zip(u_scalars, u_scalar_coeffs):
                sr = si // 2
                sc_col = si % 2
                for br in range(2):
                    for bc in range(2):
                        block_coeff = u_block_mat[br][bc]
                        if block_coeff != 0 and sc != 0:
                            gr = br * 2 + sr
                            gc = bc * 2 + sc_col
                            idx = gr * 4 + gc
                            u_dict[idx] = u_dict.get(idx, 0) + block_coeff * sc
            
            v_dict = {}
            for si, sc in zip(v_scalars, v_scalar_coeffs):
                sr = si // 2
                sc_col = si % 2
                for br in range(2):
                    for bc in range(2):
                        block_coeff = v_block_mat[br][bc]
                        if block_coeff != 0 and sc != 0:
                            gr = br * 2 + sr
                            gc = bc * 2 + sc_col
                            idx = gr * 4 + gc
                            v_dict[idx] = v_dict.get(idx, 0) + block_coeff * sc
            
            w_dict = {}
            for si, sc in zip(w_scalars, w_scalar_coeffs):
                sr = si // 2
                sc_col = si % 2
                for br in range(2):
                    for bc in range(2):
                        block_coeff = w_block_mat[br][bc]
                        if block_coeff != 0 and sc != 0:
                            gr = br * 2 + sr
                            gc = bc * 2 + sc_col
                            idx = gr * 4 + gc
                            w_dict[idx] = w_dict.get(idx, 0) + block_coeff * sc
            
            terms.append((u_dict, v_dict, w_dict))
    
    return terms
