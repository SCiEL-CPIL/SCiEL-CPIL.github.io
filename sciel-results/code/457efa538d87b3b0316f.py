<<<<<<< SEARCH
    # Iterate through the 7 big multiplications
    for big_idx, (a_parts, b_parts) in enumerate(big_combos):
        # For each big multiplication, we compute a 2x2 matrix product.
        # This 2x2 product is decomposed into 7 scalar multiplications.
        for s_idx in range(7):
            x_coeffs, y_coeffs = strassen_2x2_terms[s_idx]
            
            # Construct global U coeffs
            u_coeffs = []
            for (rx, cx), cx_c in x_coeffs:
                for a_block, a_sign in a_parts:
                    idx = a_idx(a_block, rx, cx)
                    u_coeffs.append((idx, a_sign * cx_c))
            
            # Construct global V coeffs
            v_coeffs = []
            for (ry, cy), cy_c in y_coeffs:
                for b_block, b_sign in b_parts:
                    idx = b1_idx(b_block, ry, cy)
                    v_coeffs.append((idx, b_sign * cy_c))
            
            # Construct global W coeffs
            # We need to determine which C1 blocks this term (s_idx within big_idx) contributes to.
            # First, find which C1 blocks the big term big_idx contributes to.
            # Then, for each such C1 block, find which 2x2 position (rx, cx) the small term s_idx contributes to.
            
            w_coeffs = []
            for c_block, big_contribs in big_result_map.items():
                # Check if big_idx is in big_contribs for this c_block
                big_coeff = 0
                for m_idx, m_c in big_contribs:
                    if m_idx == big_idx:
                        big_coeff = m_c
                        break
                if big_coeff == 0:
                    continue
                
                # Now, for this c_block, we need to see where s_idx lands in the 2x2 result.
                # The 2x2 result of the small Strassen is Z.
                # Z[rx, cx] = sum ...
                # We need to invert result_map: for a given s_idx, which (rx, cx) does it contribute to?
                # Or just iterate over (rx, cx) in result_map.
                for rc, s_contribs in result_map.items():
                    small_coeff = 0
                    for s_check, s_c in s_contribs:
                        if s_check == s_idx:
                            small_coeff = s_c
                            break
                    if small_coeff == 0:
                        continue
                    
                    # This term contributes to c_block at position rc with coeff big_coeff * small_coeff
                    # c_block is '00', '01', '10', '11'
                    # rc is (r, c) in 0..1
                    # Global C index:
                    idx = c1_idx(c_block, rc[0], rc[1])
                    w_coeffs.append((idx, big_coeff * small_coeff))
            
            add_term(u_coeffs, v_coeffs, w_coeffs)
=======
    # Iterate through the 7 big multiplications
    for big_idx, (a_parts, b_parts) in enumerate(big_combos):
        # For each big multiplication, we compute a 2x2 matrix product.
        # This 2x2 product is decomposed into 7 scalar multiplications.
        for s_idx in range(7):
            x_coeffs, y_coeffs = strassen_2x2_terms[s_idx]
            
            # Construct global U coeffs
            u_coeffs = []
            for (rx, cx), cx_c in x_coeffs:
                for a_block, a_sign in a_parts:
                    idx = a_idx(a_block, rx, cx)
                    u_coeffs.append((idx, a_sign * cx_c))
            
            # Construct global V coeffs
            v_coeffs = []
            for (ry, cy), cy_c in y_coeffs:
                for b_block, b_sign in b_parts:
                    idx = b1_idx(b_block, ry, cy)
                    v_coeffs.append((idx, b_sign * cy_c))
            
            # Construct global W coeffs
            w_coeffs = []
            for c_block, big_contribs in big_result_map.items():
                big_coeff = 0
                for m_idx, m_c in big_contribs:
                    if m_idx == big_idx:
                        big_coeff = m_c
                        break
                if big_coeff == 0:
                    continue
                
                for rc, s_contribs in result_map.items():
                    small_coeff = 0
                    for s_check, s_c in s_contribs:
                        if s_check == s_idx:
                            small_coeff = s_c
                            break
                    if small_coeff == 0:
                        continue
                    
                    idx = c1_idx(c_block, rc[0], rc[1])
                    w_coeffs.append((idx, big_coeff * small_coeff))
            
            add_term(u_coeffs, v_coeffs, w_coeffs)
>>>>>>> REPLACE

