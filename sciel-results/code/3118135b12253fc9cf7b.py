"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a block decomposition:
    # n=3 split into 2, 1
    # m=5 split into 2, 2, 1
    # p=5 split into 2, 2, 1
    
    # Offsets
    n_offsets = [0, 2]
    m_offsets = [0, 2, 4]
    p_offsets = [0, 2, 4]
    
    n_sizes = [2, 1]
    m_sizes = [2, 2, 1]
    p_sizes = [2, 2, 1]
    
    # Total rank estimate
    # We will build the matrices dynamically
    
    # We need a list of "terms". Each term is a rank-1 decomposition component.
    # But Strassen uses rank-7 for 2x2.
    # We will iterate over block pairs and add their contributions.
    
    # Helper to add a schoolbook block multiplication
    def add_schoolbook(i_idx, j_idx, k_idx, l_idx, U, V, W, current_rank):
        # Block A: rows i_idx, cols j_idx
        # Block B: rows k_idx, cols l_idx
        # Block C: rows i_idx, cols l_idx
        
        h_i = n_sizes[i_idx]
        w_j = m_sizes[j_idx]
        h_k = m_sizes[k_idx] # Must match w_j
        w_l = p_sizes[l_idx]
        
        # Check dimensions match
        if w_j != h_k:
            return current_rank
            
        # Schoolbook rank is h_i * w_j * w_l
        rank_block = h_i * w_j * w_l
        
        # Expand U, V, W if necessary? 
        # Better to pre-allocate max rank or use lists of lists.
        # We'll use lists of lists and append.
        
        for r in range(h_i):
            for c in range(w_j):
                for d in range(w_l):
                    # U index: (n_offsets[i_idx] + r) * m + (m_offsets[j_idx] + c)
                    u_idx = (n_offsets[i_idx] + r) * m + (m_offsets[j_idx] + c)
                    # V index: (m_offsets[k_idx] + c) * p + (p_offsets[l_idx] + d)
                    v_idx = (m_offsets[k_idx] + c) * p + (p_offsets[l_idx] + d)
                    # W index: (p_offsets[l_idx] + d) * n + (n_offsets[i_idx] + r)
                    w_idx = (p_offsets[l_idx] + d) * n + (n_offsets[i_idx] + r)
                    
                    # Append to U, V, W
                    # We are building columns. 
                    # U[u_idx] needs to be extended.
                    # This is inefficient if we do it one by one.
                    # Let's collect all (u_idx, v_idx, w_idx, coeff) tuples first.
                    pass
        return current_rank

    # This approach of appending is messy. Let's pre-allocate based on estimated rank.
    # Estimated rank 71. Let's allocate 100 to be safe.
    max_rank = 100
    U = [[0] * max_rank for _ in range(n * m)]
    V = [[0] * max_rank for _ in range(m * p)]
    W = [[0] * max_rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen 2x2 decomposition coefficients
    # P1 = (A00+A11)(B00+B11)
    # P2 = (A10+A11)B00
    # P3 = A00(B01-B11)
    # P4 = A11(B10-B00)
    # P5 = (A00+A01)B11
    # P6 = (A10-A00)(B00+B01)
    # P7 = (A01-A11)(B10+B11)
    
    # C00 = P1 + P4 - P5 + P7
    # C01 = P3 + P5
    # C10 = P2 + P4
    # C11 = P1 - P2 + P3 + P6
    
    # We define a helper to add a specific rank-1 term (u_idx, v_idx, w_idx, coeff)
    def add_term(u_idx, v_idx, w_idx, coeff):
        nonlocal col
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff
    
    # We need to handle Strassen blocks specially.
    # For 2x2 blocks, we add 7 terms.
    
    def add_strassen_block(i_idx, j_idx, k_idx, l_idx):
        nonlocal col
        # A block: rows i_idx, cols j_idx (2x2)
        # B block: rows k_idx, cols l_idx (2x2)
        # C block: rows i_idx, cols l_idx (2x2)
        
        # Local indices 0,1 for rows/cols within the block
        # Global offsets
        n_off = n_offsets[i_idx]
        m_off_a = m_offsets[j_idx]
        m_off_b = m_offsets[k_idx]
        p_off = p_offsets[l_idx]
        
        # Verify 2x2
        if n_sizes[i_idx] != 2 or m_sizes[j_idx] != 2 or m_sizes[k_idx] != 2 or p_sizes[l_idx] != 2:
            # Fallback to schoolbook if not 2x2 (should not happen for the specific blocks we call this on)
            return
        
        # P1: (A00+A11)(B00+B11)
        # A00: u=(n_off, m_off_a), v=(m_off_b, p_off), w=(p_off, n_off)
        # A11: u=(n_off+1, m_off_a+1), v=(m_off_b+1, p_off+1), w=(p_off+1, n_off+1)
        # B00: v=(m_off_b, p_off)
        # B11: v=(m_off_b+1, p_off+1)
        
        # Term 1: A00 * B00
        add_term(n_off * m + m_off_a, m_off_b * p + p_off, p_off * n + n_off, 1)
        # Term 2: A00 * B11
        add_term(n_off * m + m_off_a, m_off_b * p + (p_off+1), (p_off+1) * n + n_off, 1)
        # Term 3: A11 * B00
        add_term((n_off+1) * m + (m_off_a+1), m_off_b * p + p_off, p_off * n + (n_off+1), 1)
        # Term 4: A11 * B11
        add_term((n_off+1) * m + (m_off_a+1), m_off_b * p + (p_off+1), (p_off+1) * n + (n_off+1), 1)
        
        # P2: (A10+A11)B00
        # A10: u=(n_off+1, m_off_a)
        # A11: u=(n_off+1, m_off_a+1)
        # B00: v=(m_off_b, p_off)
        
        # Term 5: A10 * B00
        add_term((n_off+1) * m + m_off_a, m_off_b * p + p_off, p_off * n + (n_off+1), 1)
        # Term 6: A11 * B00
        add_term((n_off+1) * m + (m_off_a+1), m_off_b * p + p_off, p_off * n + (n_off+1), 1)
        
        # P3: A00(B01-B11)
        # A00: u=(n_off, m_off_a)
        # B01: v=(m_off_b, p_off+1)
        # B11: v=(m_off_b+1, p_off+1)
        
        # Term 7: A00 * B01
        add_term(n_off * m + m_off_a, m_off_b * p + (p_off+1), (p_off+1) * n + n_off, 1)
        # Term 8: A00 * (-B11)
        add_term(n_off * m + m_off_a, (m_off_b+1) * p + (p_off+1), (p_off+1) * n + n_off, -1)
        
        # P4: A11(B10-B00)
        # A11: u=(n_off+1, m_off_a+1)
        # B10: v=(m_off_b+1, p_off)
        # B00: v=(m_off_b, p_off)
        
        # Term 9: A11 * B10
        add_term((n_off+1) * m + (m_off_a+1), (m_off_b+1) * p + p_off, p_off * n + (n_off+1), 1)
        # Term 10: A11 * (-B00)
        add_term((n_off+1) * m + (m_off_a+1), m_off_b * p + p_off, p_off * n + (n_off+1), -1)
        
        # P5: (A00+A01)B11
        # A00: u=(n_off, m_off_a)
        # A01: u=(n_off, m_off_a+1)
        # B11: v=(m_off_b+1, p_off+1)
        
        # Term 11: A00 * B11
        add_term(n_off * m + m_off_a, (m_off_b+1) * p + (p_off+1), (p_off+1) * n + n_off, 1)
        # Term 12: A01 * B11
        add_term(n_off * m + (m_off_a+1), (m_off_b+1) * p + (p_off+1), (p_off+1) * n + n_off, 1)
        
        # P6: (A10-A00)(B00+B01)
        # A10: u=(n_off+1, m_off_a)
        # A00: u=(n_off, m_off_a)
        # B00: v=(m_off_b, p_off)
        # B01: v=(m_off_b, p_off+1)
        
        # Term 13: A10 * B00
        add_term((n_off+1) * m + m_off_a, m_off_b * p + p_off, p_off * n + (n_off+1), 1)
        # Term 14: A10 * B01
        add_term((n_off+1) * m + m_off_a, m_off_b * p + (p_off+1), (p_off+1) * n + (n_off+1), 1)
        # Term 15: (-A00) * B00
        add_term(n_off * m + m_off_a, m_off_b * p + p_off, p_off * n + n_off, -1)
        # Term 16: (-A00) * B01
        add_term(n_off * m + m_off_a, m_off_b * p + (p_off+1), (p_off+1) * n + n_off, -1)
        
        # P7: (A01-A11)(B10+B11)
        # A01: u=(n_off, m_off_a+1)
        # A11: u=(n_off+1, m_off_a+1)
        # B10: v=(m_off_b+1, p_off)
        # B11: v=(m_off_b+1, p_off+1)
        
        # Term 17: A01 * B10
        add_term(n_off * m + (m_off_a+1), (m_off_b+1) * p + p_off, p_off * n + n_off, 1)
        # Term 18: A01 * B11
        add_term(n_off * m + (m_off_a+1), (m_off_b+1) * p + (p_off+1), (p_off+1) * n + n_off, 1)
        # Term 19: (-A11) * B10
        add_term((n_off+1) * m + (m_off_a+1), (m_off_b+1) * p + p_off, p_off * n + (n_off+1), -1)
        # Term 20: (-A11) * B11
        add_term((n_off+1) * m + (m_off_a+1), (m_off_b+1) * p + (p_off+1), (p_off+1) * n + (n_off+1), -1)
        
        # Now we have added 20 terms to the global matrices, but they are spread across 7 "logical" columns?
        # No, the add_term function adds to the SAME column index `col`.
        # This is WRONG. Strassen requires 7 separate multiplications.
        # Each multiplication P_k is a separate rank-1 tensor sum?
        # No, P_k is a product of two linear forms.
        # So each P_k contributes 1 to the rank.
        # But P_k = (Sum A)(Sum B).
        # This is a rank-1 tensor in the space of the block?
        # Yes.
        # So for each P_k, we need to add ONE column to U, V, W.
        # The entries in that column are the coefficients of the linear forms.
        
        # My add_term above added to the SAME column for all 20 terms.
        # This effectively computed the sum of all 20 terms in one column.
        # This is incorrect.
        
        # Correct approach:
        # For each P_k, create a new column.
        # Fill U, V, W for that column with the coefficients of the linear forms for A and B.
        
        # Let's refactor.
        pass

    # Refactored approach:
    # We will build the columns one by one.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_strassen_col(i_idx, j_idx, k_idx, l_idx):
        # Returns 7 columns
        cols = []
        n_off = n_offsets[i_idx]
        m_off_a = m_offsets[j_idx]
        m_off_b = m_offsets[k_idx]
        p_off = p_offsets[l_idx]
        
        # P1: (A00+A11)(B00+B11)
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[n_off*m + m_off_a] = 1
        u[(n_off+1)*m + (m_off_a+1)] = 1
        v[m_off_b*p + p_off] = 1
        v[(m_off_b+1)*p + (p_off+1)] = 1
        # W is determined by the product? No, W is the output coefficient.
        # The decomposition is T = sum U_r V_r W_r.
        # For P1, the output is C00, C01, C10, C11 with coefficients from the combination.
        # Wait, the Strassen algorithm computes the products P1..P7.
        # Then C is formed by linear combinations of P1..P7.
        # So the "W" part of the tensor decomposition for the block is not just 1s.
        # The tensor is sum over r of u_r v_r w_r.
        # For Strassen, we have 7 products.
        # Each product P_k corresponds to a column in U, V.
        # The W column for that product must be such that when summed, it gives the correct C.
        # Actually, the standard tensor decomposition view:
        # T = sum_{k=1}^7 u_k v_k w_k.
        # Here u_k, v_k are the linear forms in A and B.
        # w_k is the linear form in C? No, w_k is the coefficient vector for the output.
        # The output C is a linear combination of the products.
        # C_00 = P1 + P4 - P5 + P7.
        # So the "W" vector for column 1 (P1) should have 1 at C_00 and C_11?
        # No. The decomposition is:
        # C = sum_k (u_k(A) v_k(B)) w_k.
        # Here w_k is a vector in the output space.
        # For P1, it contributes to C_00 and C_11.
        # So w_1 has 1 at index C_00 and 1 at index C_11.
        
        # Let's verify indices for W.
        # W index is k*n + i.
        # C_00 is i=0, k=0. Index 0.
        # C_11 is i=1, k=1. Index 1*n + 1 = n+1.
        
        # So for P1:
        w[n_off*n + p_off] = 1 # C_00
        w[(n_off+1)*n + (p_off+1)] = 1 # C_11
        cols.append((u, v, w))
        
        # P2: (A10+A11)B00
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[(n_off+1)*m + m_off_a] = 1
        u[(n_off+1)*m + (m_off_a+1)] = 1
        v[m_off_b*p + p_off] = 1
        # C_10 = P2 + P4. So P2 contributes to C_10.
        # C_11 = P1 - P2 + P3 + P6. So P2 contributes to C_11 with -1.
        w[(n_off+1)*n + p_off] = 1 # C_10
        w[(n_off+1)*n + (p_off+1)] = -1 # C_11
        cols.append((u, v, w))
        
        # P3: A00(B01-B11)
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[n_off*m + m_off_a] = 1
        v[m_off_b*p + (p_off+1)] = 1
        v[(m_off_b+1)*p + (p_off+1)] = -1
        # C_01 = P3 + P5.
        # C_11 = ... + P3 ...
        w[n_off*n + (p_off+1)] = 1 # C_01
        w[(n_off+1)*n + (p_off+1)] = 1 # C_11
        cols.append((u, v, w))
        
        # P4: A11(B10-B00)
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[(n_off+1)*m + (m_off_a+1)] = 1
        v[(m_off_b+1)*p + p_off] = 1
        v[m_off_b*p + p_off] = -1
        # C_00 = P1 + P4 ...
        # C_10 = P2 + P4 ...
        w[n_off*n + p_off] = 1 # C_00
        w[(n_off+1)*n + p_off] = 1 # C_10
        cols.append((u, v, w))
        
        # P5: (A00+A01)B11
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[n_off*m + m_off_a] = 1
        u[n_off*m + (m_off_a+1)] = 1
        v[(m_off_b+1)*p + (p_off+1)] = 1
        # C_00 = ... - P5 ...
        # C_01 = P3 + P5 ...
        w[n_off*n + p_off] = -1 # C_00
        w[n_off*n + (p_off+1)] = 1 # C_01
        cols.append((u, v, w))
        
        # P6: (A10-A00)(B00+B01)
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[(n_off+1)*m + m_off_a] = 1
        u[n_off*m + m_off_a] = -1
        v[m_off_b*p + p_off] = 1
        v[m_off_b*p + (p_off+1)] = 1
        # C_11 = ... + P6 ...
        w[(n_off+1)*n + (p_off+1)] = 1 # C_11
        cols.append((u, v, w))
        
        # P7: (A01-A11)(B10+B11)
        u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
        u[n_off*m + (m_off_a+1)] = 1
        u[(n_off+1)*m + (m_off_a+1)] = -1
        v[(m_off_b+1)*p + p_off] = 1
        v[(m_off_b+1)*p + (p_off+1)] = 1
        # C_00 = ... + P7 ...
        w[n_off*n + p_off] = 1 # C_00
        cols.append((u, v, w))
        
        return cols

    def add_schoolbook_col(i_idx, j_idx, k_idx, l_idx):
        cols = []
        n_off = n_offsets[i_idx]
        m_off_a = m_offsets[j_idx]
        m_off_b = m_offsets[k_idx]
        p_off = p_offsets[l_idx]
        
        h_i = n_sizes[i_idx]
        w_j = m_sizes[j_idx]
        w_l = p_sizes[l_idx]
        
        for r in range(h_i):
            for c in range(w_j):
                for d in range(w_l):
                    u = [0]*(n*m); v = [0]*(m*p); w = [0]*(p*n)
                    u[(n_off+r)*m + (m_off_a+c)] = 1
                    v[(m_off_b+c)*p + (p_off+d)] = 1
                    w[(p_off+d)*n + (n_off+r)] = 1
                    cols.append((u, v, w))
        return cols

    # Iterate over all block pairs
    for i in range(len(n_sizes)):
        for j in range(len(m_sizes)):
            for k in range(len(m_sizes)):
                if m_sizes[j] != m_sizes[k]:
                    continue
                for l in range(len(p_sizes)):
                    if n_sizes[i] == 2 and m_sizes[j] == 2 and p_sizes[l] == 2:
                        cols = add_strassen_col(i, j, k, l)
                    else:
                        cols = add_schoolbook_col(i, j, k, l)
                    
                    for u, v, w in cols:
                        U_cols.append(u)
                        V_cols.append(v)
                        W_cols.append(w)
                        
    rank = len(U_cols)
    
    # Convert lists of columns to matrices
    U = [[U_cols[r][i] for r in range(rank)] for i in range(n*m)]
    V = [[V_cols[r][i] for r in range(rank)] for i in range(m*p)]
    W = [[W_cols[r][i] for r in range(rank)] for i in range(p*n)]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
