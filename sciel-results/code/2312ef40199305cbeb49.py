"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will construct a decomposition of rank < 100 using block Strassen.
    # Split A (4x5) into 4 blocks of size 2x2, 2x3, 2x2, 2x3.
    # Split B (5x5) into 4 blocks of size 2x2, 2x3, 3x2, 3x3.
    # This is complex to assemble manually. 
    # Instead, we use a known rank-96 or lower construction if available, 
    # or fall back to a simple Strassen-based block decomposition.
    
    # Let's implement the block decomposition with Strassen for 2x2x2 parts.
    # Rank estimate: 92.
    
    # Helper to add a term to U, V, W
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # We build the factors by concatenating the factors of the sub-blocks.
    
    U_rows = []
    V_rows = []
    W_rows = []
    
    # We need to map global indices to local block indices.
    # A blocks:
    # A11: i in [0,1], j in [0,1]
    # A12: i in [0,1], j in [2,4]
    # A21: i in [2,3], j in [0,1]
    # A22: i in [2,3], j in [2,4]
    
    # B blocks:
    # B11: j in [0,1], k in [0,1]
    # B12: j in [0,1], k in [2,4]
    # B21: j in [2,4], k in [0,1]
    # B22: j in [2,4], k in [2,4]
    
    # Products:
    # C11 = A11 B11 + A12 B21
    # C12 = A11 B12 + A12 B22
    # C21 = A21 B11 + A22 B21
    # C22 = A21 B12 + A22 B22
    
    # We will generate the decomposition for each product term separately and sum them.
    # Since the terms are independent in the tensor sum (they correspond to disjoint sets of (i,j,k) tuples? 
    # No, they correspond to specific (i,j,k) tuples.
    # The tensor T is the sum of these block tensors.
    # T = T11 + T12 + T21 + T22 where T11 corresponds to A11B11 etc.
    # The rank is the sum of the ranks of the block decompositions.
    
    # We need a function to get the U, V, W factors for a block multiplication n x m x p.
    
    def get_block_factors(n_b, m_b, p_b, type_strassen=False):
        """
        Returns U, V, W for block size n_b, m_b, p_b.
        U shape: (n_b*m_b, rank)
        V shape: (m_b*p_b, rank)
        W shape: (p_b*n_b, rank)
        """
        if n_b == 2 and m_b == 2 and p_b == 2:
            if type_strassen:
                # Strassen rank 7
                # M1 = (A11+A22)(B11+B22)
                # M2 = (A21+A22)B11
                # M3 = A11(B21-B22)
                # M4 = A22(B12-B11)
                # M5 = (A11+A12)B22
                # M6 = (A21-A11)(B11+B12)
                # M7 = (A12-A22)(B21+B22)
                # C11 = M1+M4-M5+M7
                # C12 = M3+M5
                # C21 = M2+M4
                # C22 = M1-M2+M3+M6
                
                U = [[0]*7 for _ in range(4)] # 2x2
                V = [[0]*7 for _ in range(4)]
                W = [[0]*7 for _ in range(4)]
                
                # Indices for U: (i, j) -> i*2+j
                # Indices for V: (j, k) -> j*2+k
                # Indices for W: (k, i) -> k*2+i
                
                # M1: (A11+A22)(B11+B22)
                # A part: A11 (0,0), A22 (1,1)
                U[0][0] = 1; U[3][0] = 1
                # B part: B11 (0,0), B22 (1,1)
                V[0][0] = 1; V[3][0] = 1
                # C part: C11 (0,0), C22 (1,1)
                W[0][0] = 1; W[3][0] = 1
                
                # M2: (A21+A22)B11
                # A part: A21 (1,0), A22 (1,1)
                U[2][1] = 1; U[3][1] = 1
                # B part: B11 (0,0)
                V[0][1] = 1
                # C part: C21 (0,1)
                W[2][1] = 1
                
                # M3: A11(B21-B22)
                # A part: A11 (0,0)
                U[0][2] = 1
                # B part: B21 (1,0), B22 (1,1)
                V[2][2] = 1; V[3][2] = -1
                # C part: C12 (1,0)
                W[1][2] = 1
                
                # M4: A22(B12-B11)
                # A part: A22 (1,1)
                U[3][3] = 1
                # B part: B12 (0,1), B11 (0,0)
                V[1][3] = 1; V[0][3] = -1
                # C part: C21 (0,1)
                W[2][3] = 1
                
                # M5: (A11+A12)B22
                # A part: A11 (0,0), A12 (0,1)
                U[0][4] = 1; U[1][4] = 1
                # B part: B22 (1,1)
                V[3][4] = 1
                # C part: C11 (0,0) with -1
                W[0][4] = -1
                
                # M6: (A21-A11)(B11+B12)
                # A part: A21 (1,0), A11 (0,0)
                U[2][5] = 1; U[0][5] = -1
                # B part: B11 (0,0), B12 (0,1)
                V[0][5] = 1; V[1][5] = 1
                # C part: C22 (1,1)
                W[3][5] = 1
                
                # M7: (A12-A22)(B21+B22)
                # A part: A12 (0,1), A22 (1,1)
                U[1][6] = 1; U[3][6] = -1
                # B part: B21 (1,0), B22 (1,1)
                V[2][6] = 1; V[3][6] = 1
                # C part: C11 (0,0)
                W[0][6] = 1
                
                return U, V, W
            else:
                # Schoolbook 2x2x2 rank 8
                U = [[0]*8 for _ in range(4)]
                V = [[0]*8 for _ in range(4)]
                W = [[0]*8 for _ in range(4)]
                for i in range(2):
                    for j in range(2):
                        for k in range(2):
                            col = i*4 + j*2 + k
                            U[i*2+j][col] = 1
                            V[j*2+k][col] = 1
                            W[k*2+i][col] = 1
                return U, V, W
                
        elif n_b == 2 and m_b == 2 and p_b == 3:
            # 2x2x3. Split p into 2+1.
            # Rank 7 (for 2x2x2) + 4 (for 2x2x1) = 11.
            U1, V1, W1 = get_block_factors(2, 2, 2, True)
            U2, V2, W2 = get_block_factors(2, 2, 1, False)
            
            U = []
            V = []
            W = []
            for r in range(11):
                u_row = [0]*4
                v_row = [0]*6
                w_row = [0]*6
                if r < 7:
                    for i in range(4): u_row[i] = U1[i][r]
                    for j in range(2):
                        for k in range(2):
                            v_row[j*3+k] = V1[j*2+k][r]
                    for k in range(2):
                        for i in range(2):
                            w_row[k*2+i] = W1[k*2+i][r]
                else:
                    rr = r - 7
                    for i in range(4): u_row[i] = U2[i][rr]
                    for j in range(2):
                        for k in range(1):
                            v_row[j*3 + 2 + k] = V2[j*1+k][rr] # k=2
                    for k in range(1):
                        for i in range(2):
                            w_row[k*2+i] = W2[k*2+i][rr] # k=2? No, W index is k*n+i.
                            # In 2x2x1, k is 0. So W[0*2+i].
                            # But here k is 2. So W[2*2+i].
                            w_row[2*2+i] = W2[0*2+i][rr]
                U.append(u_row)
                V.append(v_row)
                W.append(w_row)
            return U, V, W
            
        elif n_b == 2 and m_b == 3 and p_b == 2:
            # 2x3x2. Split m into 2+1.
            # Rank 7 (for 2x2x2) + 4 (for 2x1x2) = 11.
            U1, V1, W1 = get_block_factors(2, 2, 2, True)
            U2, V2, W2 = get_block_factors(2, 1, 2, False)
            
            U = []
            V = []
            W = []
            for r in range(11):
                u_row = [0]*6
                v_row = [0]*6
                w_row = [0]*4
                if r < 7:
                    for j in range(2):
                        for i in range(2):
                            u_row[i*3+j] = U1[i*2+j][r]
                    for j in range(2):
                        for k in range(2):
                            v_row[j*2+k] = V1[j*2+k][r]
                    for k in range(2):
                        for i in range(2):
                            w_row[k*2+i] = W1[k*2+i][r]
                else:
                    rr = r - 7
                    for j in range(1):
                        for i in range(2):
                            u_row[i*3 + 2 + j] = U2[i*1+j][rr]
                    for j in range(1):
                        for k in range(2):
                            v_row[2*2 + j*2+k] = V2[j*2+k][rr]
                    for k in range(2):
                        for i in range(2):
                            w_row[k*2+i] = W2[k*2+i][rr]
                U.append(u_row)
                V.append(v_row)
                W.append(w_row)
            return U, V, W
            
        elif n_b == 2 and m_b == 3 and p_b == 3:
            # 2x3x3. Split m into 2+1, p into 2+1.
            # Rank 7 (2x2x2) + 4 (2x1x2) + 4 (2x2x1) + 4 (2x1x1) = 19?
            # Schoolbook is 18. So this is worse.
            # Try split p into 3? No.
            # Use schoolbook for 2x3x3? Rank 18.
            U = [[0]*18 for _ in range(6)]
            V = [[0]*18 for _ in range(9)]
            W = [[0]*18 for _ in range(6)]
            for i in range(2):
                for j in range(3):
                    for k in range(3):
                        col = i*9 + j*3 + k
                        U[i*3+j][col] = 1
                        V[j*3+k][col] = 1
                        W[k*2+i][col] = 1
            return U, V, W
            
        else:
            # Fallback schoolbook
            rank = n_b * m_b * p_b
            U = [[0]*rank for _ in range(n_b*m_b)]
            V = [[0]*rank for _ in range(m_b*p_b)]
            W = [[0]*rank for _ in range(p_b*n_b)]
            for i in range(n_b):
                for j in range(m_b):
                    for k in range(p_b):
                        col = i*(m_b*p_b) + j*p_b + k
                        U[i*m_b+j][col] = 1
                        V[j*p_b+k][col] = 1
                        W[k*n_b+i][col] = 1
            return U, V, W

    # Assemble blocks
    # A11 (2x2), A12 (2x3), A21 (2x2), A22 (2x3)
    # B11 (2x2), B12 (2x3), B21 (3x2), B22 (3x3)
    
    blocks = [
        (0, 0, 2, 2, 2, 2, True),  # A11 B11
        (0, 2, 2, 3, 3, 2, False), # A12 B21 (2x3 x 3x2 -> 2x2)
        (0, 0, 2, 2, 2, 3, False), # A11 B12 (2x2 x 2x3 -> 2x3)
        (0, 2, 2, 3, 3, 3, False), # A12 B22 (2x3 x 3x3 -> 2x3)
        (2, 0, 2, 2, 2, 2, True),  # A21 B11
        (2, 2, 2, 3, 3, 2, False), # A22 B21
        (2, 0, 2, 2, 2, 3, False), # A21 B12
        (2, 2, 2, 3, 3, 3, False), # A22 B22
    ]
    
    # Note: The get_block_factors function assumes local indices 0..n-1 etc.
    # We need to map local indices to global indices.
    
    total_rank = 0
    U_global = []
    V_global = []
    W_global = []
    
    for i_off, j_off, n_b, m_b, p_b, p_b2, use_strassen in blocks:
        # p_b is the output rows? No, p_b is B cols.
        # A is n_b x m_b. B is m_b x p_b.
        # Wait, in the block list:
        # A11 B11: A(2x2) B(2x2). n_b=2, m_b=2, p_b=2.
        # A12 B21: A(2x3) B(3x2). n_b=2, m_b=3, p_b=2.
        
        if n_b == 2 and m_b == 2 and p_b == 2:
            U_b, V_b, W_b = get_block_factors(2, 2, 2, use_strassen)
        elif n_b == 2 and m_b == 2 and p_b == 3:
            U_b, V_b, W_b = get_block_factors(2, 2, 3, False)
        elif n_b == 2 and m_b == 3 and p_b == 2:
            U_b, V_b, W_b = get_block_factors(2, 3, 2, False)
        elif n_b == 2 and m_b == 3 and p_b == 3:
            U_b, V_b, W_b = get_block_factors(2, 3, 3, False)
        else:
            U_b, V_b, W_b = get_block_factors(n_b, m_b, p_b, False)
            
        rank_b = len(U_b[0])
        
        # Map to global
        for r in range(rank_b):
            u_row = [0] * 20 # 4*5
            v_row = [0] * 25 # 5*5
            w_row = [0] * 20 # 5*4
            
            for i in range(n_b):
                for j in range(m_b):
                    gi = i_off + i
                    gj = j_off + j
                    u_row[gi*5 + gj] = U_b[i*m_b+j][r]
                    
            for j in range(m_b):
                for k in range(p_b):
                    gj = j_off + j
                    gk = k + (0 if p_b == 2 else 2) # B col offset?
                    # B cols: 0-1 for B11/B12, 2-4 for B21/B22?
                    # B11 (2x2): j 0-1, k 0-1
                    # B12 (2x3): j 0-1, k 2-4
                    # B21 (3x2): j 2-4, k 0-1
                    # B22 (3x3): j 2-4, k 2-4
                    
                    # My block list has p_b as the size of B's columns.
                    # For B11, p_b=2. k_off=0.
                    # For B12, p_b=3. k_off=2.
                    # For B21, p_b=2. k_off=0.
                    # For B22, p_b=3. k_off=2.
                    
                    # I need to pass k_off.
                    pass
            
            # This mapping is getting too complex for the loop.
            # I will simplify by just using schoolbook for the non-strassen blocks 
            # and only applying Strassen to the two 2x2x2 blocks.
            
            # Re-evaluating: 
            # Only A11B11 and A21B11 are 2x2x2.
            # All others are schoolbook.
            # Rank = 7 + 7 + (12+12+18+12+18+12+18) = 14 + 102 = 116?
            # No, schoolbook for the others sums to 92?
            # 100 - 8 - 8 = 84.
            # 7 + 7 + 84 = 98.
            
            # Let's just implement the 98 rank version.
            
            if n_b == 2 and m_b == 2 and p_b == 2 and use_strassen:
                # Strassen
                pass
            else:
                # Schoolbook
                pass

    # Given the complexity and time limit, I will return the schoolbook rank 100 
    # to ensure validity, or a simple Strassen on 2x2x2 if I can map it easily.
    # Actually, the prompt asks for a "stronger algorithm". 
    # I will provide the rank 98 construction by explicitly handling the two 2x2x2 blocks with Strassen.
    
    U = [[0]*98 for _ in range(20)]
    V = [[0]*98 for _ in range(25)]
    W = [[0]*98 for _ in range(20)]
    
    col = 0
    # Block A11 B11 (2x2x2) Strassen
    # i_off=0, j_off=0, k_off=0
    U_b, V_b, W_b = get_block_factors(2, 2, 2, True)
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U[i*5+j][col+r] = U_b[i*2+j][r]
        for j in range(2):
            for k in range(2):
                V[j*5+k][col+r] = V_b[j*2+k][r]
        for k in range(2):
            for i in range(2):
                W[k*4+i][col+r] = W_b[k*2+i][r]
    col += 7
    
    # Block A21 B11 (2x2x2) Strassen
    # i_off=2, j_off=0, k_off=0
    for r in range(7):
        for i in range(2):
            for j in range(2):
                U[(2+i)*5+j][col+r] = U_b[i*2+j][r]
        for j in range(2):
            for k in range(2):
                V[j*5+k][col+r] = V_b[j*2+k][r]
        for k in range(2):
            for i in range(2):
                W[k*4+(2+i)][col+r] = W_b[k*2+i][r]
    col += 7
    
    # Remaining blocks: Schoolbook
    # A11 B12 (2x2x3)
    # i_off=0, j_off=0, k_off=2
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[i*5+j][col] = 1
                V[j*5+(2+k)][col] = 1
                W[(2+k)*4+i][col] = 1
                col += 1
                
    # A12 B21 (2x3x2)
    # i_off=0, j_off=2, k_off=0
    for i in range(2):
        for j in range(3):
            for k in range(2):
                U[i*5+(2+j)][col] = 1
                V[(2+j)*5+k][col] = 1
                W[k*4+i][col] = 1
                col += 1
                
    # A12 B22 (2x3x3)
    # i_off=0, j_off=2, k_off=2
    for i in range(2):
        for j in range(3):
            for k in range(3):
                U[i*5+(2+j)][col] = 1
                V[(2+j)*5+(2+k)][col] = 1
                W[(2+k)*4+i][col] = 1
                col += 1
                
    # A21 B12 (2x2x3)
    # i_off=2, j_off=0, k_off=2
    for i in range(2):
        for j in range(2):
            for k in range(3):
                U[(2+i)*5+j][col] = 1
                V[j*5+(2+k)][col] = 1
                W[(2+k)*4+(2+i)][col] = 1
                col += 1
                
    # A22 B21 (2x3x2)
    # i_off=2, j_off=2, k_off=0
    for i in range(2):
        for j in range(3):
            for k in range(2):
                U[(2+i)*5+(2+j)][col] = 1
                V[(2+j)*5+k][col] = 1
                W[k*4+(2+i)][col] = 1
                col += 1
                
    # A22 B22 (2x3x3)
    # i_off=2, j_off=2, k_off=2
    for i in range(2):
        for j in range(3):
            for k in range(3):
                U[(2+i)*5+(2+j)][col] = 1
                V[(2+j)*5+(2+k)][col] = 1
                W[(2+k)*4+(2+i)][col] = 1
                col += 1
                
    # col should be 98
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
