def _generate_terms():
    """Generate all (u_row, u_val, v_row, v_val, w_row, w_val) tuples."""
    n, m, p = 3, 5, 6
    terms = []
    
    # Partition: n: [1,2], m: [2,2,1], p: [2,2,2]
    n_parts = [(0, 1), (1, 3)]  # (start, end)
    m_parts = [(0, 2), (2, 4), (4, 5)]
    p_parts = [(0, 2), (2, 4), (4, 6)]
    
    for (n_start, n_end) in n_parts:
        for (m_start, m_end) in m_parts:
            for (p_start, p_end) in p_parts:
                n_size = n_end - n_start
                m_size = m_end - m_start
                p_size = p_end - p_start
                
                if (n_size, m_size, p_size) == (1, 2, 2):
                    # Rank 4
                    for j in range(2):
                        for k in range(2):
                            terms.append((
                                (n_start) * m + (m_start + j), 1,  # U row
                                (m_start + j) * p + (p_start + k), 1,  # V row
                                (p_start + k) * n + (n_start), 1  # W row
                            ))
                elif (n_size, m_size, p_size) == (1, 1, 2):
                    # Rank 2
                    for k in range(2):
                        terms.append((
                            (n_start) * m + (m_start), 1,
                            (m_start) * p + (p_start + k), 1,
                            (p_start + k) * n + (n_start), 1
                        ))
                elif (n_size, m_size, p_size) == (2, 2, 2):
                    # Rank 7: Strassen
                    # A is 2x2, B is 2x2, C is 2x2
                    # i,j,k in {0,1} relative to block
                    # U row: (n_start+i)*m + (m_start+j)
                    # V row: (m_start+j)*p + (p_start+k)
                    # W row: (p_start+k)*n + (n_start+i)
                    # Strassen rank-7:
                    # t1 = (a00+a11)*(b00+b11)
                    # t2 = (a10+a11)*b00
                    # t3 = a00*(b01-b11)
                    # t4 = a11*(b10-b00)
                    # t5 = (a00+a01)*b11
                    # t6 = (a10-a00)*(b00+b01)
                    # t7 = (a01-a11)*(b10+b11)
                    # C00 = t1 + t4 - t5 + t7
                    # C01 = t3 + t5
                    # C10 = t2 + t4
                    # C11 = t1 - t2 + t3 + t6
                    
                    # Let me verify this is correct.
                    # a00*b00 + a00*b01 + a01*b10 + a01*b11 + ... no.
                    # C[0,0] = a00*b00 + a01*b10
                    # C[0,1] = a00*b01 + a01*b11
                    # C[1,0] = a10*b00 + a11*b10
                    # C[1,1] = a10*b01 + a11*b11
                    
                    # t1 = (a00+a11)(b00+b11) = a00*b00 + a00*b11 + a11*b00 + a11*b11
                    # t2 = (a10+a11)*b00 = a10*b00 + a11*b00
                    # t3 = a00*(b01-b11) = a00*b01 - a00*b11
                    # t4 = a11*(b10-b00) = a11*b10 - a11*b00
                    # t5 = (a00+a01)*b11 = a00*b11 + a01*b11
                    # t6 = (a10-a00)*(b00+b01) = a10*b00 + a10*b01 - a00*b00 - a00*b01
                    # t7 = (a01-a11)*(b10+b11) = a01*b10 + a01*b11 - a11*b10 - a11*b11
                    
                    # C[0,0] should be a00*b00 + a01*b10
                    # t1+t4-t5+t7 = (a00*b00 + a00*b11 + a11*b00 + a11*b11) + (a11*b10 - a11*b00) - (a00*b11 + a01*b11) + (a01*b10 + a01*b11 - a11*b10 - a11*b11)
                    # = a00*b00 + a00*b11 + a11*b00 + a11*b11 + a11*b10 - a11*b00 - a00*b11 - a01*b11 + a01*b10 + a01*b11 - a11*b10 - a11*b11
                    # = a00*b00 + a01*b10  ✓
                    
                    # C[0,1] should be a00*b01 + a01*b11
                    # t3+t5 = (a00*b01 - a00*b11) + (a00*b11 + a01*b11) = a00*b01 + a01*b11  ✓
                    
                    # C[1,0] should be a10*b00 + a11*b10
                    # t2+t4 = (a10*b00 + a11*b00) + (a11*b10 - a11*b00) = a10*b00 + a11*b10  ✓
                    
                    # C[1,1] should be a10*b01 + a11*b11
                    # t1-t2+t3+t6 = (a00*b00 + a00*b11 + a11*b00 + a11*b11) - (a10*b00 + a11*b00) + (a00*b01 - a00*b11) + (a10*b00 + a10*b01 - a00*b00 - a00*b01)
                    # = a00*b00 + a00*b11 + a11*b00 + a11*b11 - a10*b00 - a11*b00 + a00*b01 - a00*b11 + a10*b00 + a10*b01 - a00*b00 - a00*b01
                    # = a11*b11 + a10*b01  ✓
                    
                    # Great! Now express each term as U,V,W contributions.
                    # Each t is a product of a linear combination of A entries and a linear combination of B entries,
                    # contributing to specific C entries.
                    
                    # For each t, we need to decompose it into U*V*W contributions.
                    # t = (sum_u u_coeff * A[u]) * (sum_v v_coeff * B[v]) * (sum_w w_coeff * C[w])
                    # This is a rank-1 tensor in the decomposition sense.
                    
                    # t1: U part: a00 + a11, V part: b00 + b11, W part: C00 + C11
                    # In the decomposition, t1 contributes to C00 and C11.
                    # U: A[0,0] coeff 1, A[1,1] coeff 1
                    # V: B[0,0] coeff 1, B[1,1] coeff 1
                    # W: C[0,0] coeff 1, C[1,1] coeff 1
                    
                    # But wait, in the tensor decomposition, each column c has:
                    # U[i*m+j, c] * V[j'*p+k', c] * W[k''*n+i'', c]
                    # and the product sums over c to give the tensor.
                    
                    # For t1, we need:
                    # (A[0,0] + A[1,1]) * (B[0,0] + B[1,1]) * (C[0,0] + C[1,1])
                    # This expands to 4 terms, but in the tensor decomposition, it's ONE column
                    # where U has non-zero entries at (0,0) and (1,1), V at (0,0) and (1,1), W at (0,0) and (1,1).
                    
                    # So for t1:
                    # U[0*m+0] = 1, U[1*m+1] = 1  (relative to block)
                    # V[0*p+0] = 1, V[1*p+1] = 1
                    # W[0*n+0] = 1, W[1*n+1] = 1
                    
                    # Let me define the Strassen terms with their U, V, W coefficients.
                    
                    strassen_terms = []
                    # Each entry: (u_coeffs, v_coeffs, w_coeffs)
                    # u_coeffs: list of (rel_i, rel_j, coeff)
                    # v_coeffs: list of (rel_j, rel_k, coeff)
                    # w_coeffs: list of (rel_k, rel_i, coeff)
                    
                    # t1: (a00+a11)(b00+b11) -> C00+C11
                    strassen_terms.append(
                        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)])
                    )
                    # t2: (a10+a11)b00 -> C10
                    strassen_terms.append(
                        ([(1,0,1),(1,1,1)], [(0,0,1)], [(0,1,1)])
                    )
                    # t3: a00(b01-b11) -> C01
                    strassen_terms.append(
                        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(1,0,1)])
                    )
                    # t4: a11(b10-b00) -> C00
                    strassen_terms.append(
                        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1)])
                    )
                    # t5: (a00+a01)b11 -> C00
                    strassen_terms.append(
                        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,1)])
                    )
                    # t6: (a10-a00)(b00+b01) -> C11
                    strassen_terms.append(
                        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)])
                    )
                    # t7: (a01-a11)(b10+b11) -> C00
                    strassen_terms.append(
                        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)])
                    )
                    
                    for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
                        # Create one column for this term
                        u_row_val = {}
                        for (ri, rj, coeff) in u_coeffs:
                            row = (n_start + ri) * m + (m_start + rj)
                            u_row_val[row] = u_row_val.get(row, 0) + coeff
                        v_row_val = {}
                        for (rj, rk, coeff) in v_coeffs:
                            row = (m_start + rj) * p + (p_start + rk)
                            v_row_val[row] = v_row_val.get(row, 0) + coeff
                        w_row_val = {}
                        for (rk, ri, coeff) in w_coeffs:
                            row = (p_start + rk) * n + (n_start + ri)
                            w_row_val[row] = w_row_val.get(row, 0) + coeff
                        
                        terms.append((u_row_val, v_row_val, w_row_val))
                elif (n_size, m_size, p_size) == (2, 1, 2):
                    # Rank 4
                    for i in range(2):
                        for k in range(2):
                            terms.append((
                                (n_start + i) * m + (m_start), 1,
                                (m_start) * p + (p_start + k), 1,
                                (p_start + k) * n + (n_start + i), 1
                            ))
    
    return terms
