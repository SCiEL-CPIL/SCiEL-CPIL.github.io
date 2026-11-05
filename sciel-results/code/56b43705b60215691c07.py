import time

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    n, m, p = DIMENSIONS
    
    # Initialize with zeros
    # We will determine the total rank first
    # Components:
    # 1. 3x3x3 (Schoolbook): 27
    # 2. 3x2x3 (Schoolbook): 18
    # 3. 3x3x2 (Schoolbook): 18
    # 4. 3x2x2 (Schoolbook): 12
    # 5. 2x3x3 (Schoolbook): 18
    # 6. 2x2x3 (Schoolbook): 12
    # 7. 2x3x2 (Schoolbook): 12
    # 8. 2x2x2 (Strassen): 7
    
    rank_total = 27 + 18 + 18 + 12 + 18 + 12 + 12 + 7
    rank_total = 124
    
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    col_offset = 0
    
    # Helper to add schoolbook component
    def add_schoolbook(n_loc, m_loc, p_loc, i_base, j_base, k_base, offset):
        # n_loc: rows of A
        # m_loc: cols of A / rows of B
        # p_loc: cols of B
        # i_base: global start index for i
        # j_base: global start index for j
        # k_base: global start index for k
        
        rank_loc = n_loc * m_loc * p_loc
        for i in range(n_loc):
            for j in range(m_loc):
                for k in range(p_loc):
                    c = offset + (i * m_loc + j) * p_loc + k
                    U[i_base + i][c] = 1
                    V[j_base + j][c] = 1
                    W[k_base + k][c] = 1
        return rank_loc

    # Helper to add Strassen 2x2x2 component
    def add_strassen_2x2(i_base, j_base, k_base, offset):
        # Strassen for 2x2 matrices
        # A = [[A11, A12], [A21, A22]]
        # B = [[B11, B12], [B21, B22]]
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Coefficients for U, V, W
        # U[i,j], V[j,k], W[k,i]
        
        # M1
        # A part: A11 (1,1), A22 (1,1) -> U[i,j] += 1 for (0,0) and (1,1)
        # B part: B11 (1,1), B22 (1,1) -> V[j,k] += 1 for (0,0) and (1,1)
        # C part: C11 (1,1), C22 (1,1) -> W[k,i] += 1 for (0,0) and (1,1)
        
        # We need to define the 7 terms explicitly.
        
        # Term 1: (A11+A22)(B11+B22)
        # U: (0,0)+1, (1,1)+1
        # V: (0,0)+1, (1,1)+1
        # W: (0,0)+1, (1,1)+1
        U[i_base+0][offset+0] += 1
        U[i_base+1][offset+0] += 1
        V[j_base+0][offset+0] += 1
        V[j_base+1][offset+0] += 1
        W[k_base+0][offset+0] += 1
        W[k_base+1][offset+0] += 1
        
        # Term 2: (A21+A22)B11
        # U: (1,0)+1, (1,1)+1
        # V: (0,0)+1
        # W: (0,0)+1
        U[i_base+1][offset+1] += 1
        U[i_base+1][offset+1] += 1 # Wait, A21 is (1,0), A22 is (1,1).
        # Correction: U[i,j] corresponds to A_ij.
        # A21 is i=1, j=0. A22 is i=1, j=1.
        U[i_base+1][offset+1] += 1 # A21
        U[i_base+1][offset+1] += 1 # A22 -> This adds 2 to same spot? No, different terms?
        # No, in the decomposition, each term is a product of linear forms.
        # The column 'offset+1' corresponds to the second factor in the rank decomposition.
        # U[i,j] is the coefficient of A_ij in the linear form for A.
        # So for Term 2, the linear form for A is A21 + A22.
        # So U[i_base+1, j_base+0] should be 1? No, U is indexed by global i, j.
        # U[i_base+1][offset+1] += 1 (for A21)
        # U[i_base+1][offset+1] += 1 (for A22) -> This would make it 2.
        # But A21 and A22 are different elements.
        # A21 is at (1,0). A22 is at (1,1).
        # So U[i_base+1, j_base+0] += 1
        # U[i_base+1, j_base+1] += 1
        
        # Let's restart the Strassen mapping carefully.
        # We are adding to global U, V, W.
        # U index: i_global * m + j_global? No, U is (n*m, rank).
        # U[i_global * m + j_global, c]
        
        # Term 1: (A11+A22)(B11+B22) -> C11 + C22
        # A: A11 (0,0), A22 (1,1)
        U[i_base*5 + j_base + 0][offset+0] += 1
        U[i_base*5 + j_base + 1][offset+0] += 1
        # B: B11 (0,0), B22 (1,1)
        V[j_base*5 + k_base + 0][offset+0] += 1
        V[j_base*5 + k_base + 1][offset+0] += 1
        # C: C11 (0,0), C22 (1,1)
        W[k_base*5 + i_base + 0][offset+0] += 1
        W[k_base*5 + i_base + 1][offset+0] += 1

        # Term 2: (A21+A22)B11 -> C21
        # A: A21 (1,0), A22 (1,1)
        U[i_base*5 + j_base + 0][offset+1] += 1
        U[i_base*5 + j_base + 1][offset+1] += 1
        # B: B11 (0,0)
        V[j_base*5 + k_base + 0][offset+1] += 1
        # C: C21 (1,0)
        W[k_base*5 + i_base + 1][offset+1] += 1

        # Term 3: A11(B12-B22) -> C12
        # A: A11 (0,0)
        U[i_base*5 + j_base + 0][offset+2] += 1
        # B: B12 (0,1) - B22 (1,1)
        V[j_base*5 + k_base + 1][offset+2] += 1
        V[j_base*5 + k_base + 1][offset+2] -= 1 # Wait, B22 is (1,1). B12 is (0,1).
        # B indices: j, k.
        # B12: j=0, k=1.
        # B22: j=1, k=1.
        V[j_base*5 + k_base + 1][offset+2] += 1 # B12
        V[j_base*5 + k_base + 1][offset+2] -= 1 # B22? No, B22 is j=1, k=1.
        # V index is j*5 + k.
        # B12: j=0, k=1 -> idx 1.
        # B22: j=1, k=1 -> idx 6.
        V[j_base*5 + k_base + 1][offset+2] += 1 # B12
        V[j_base*5 + k_base + 6][offset+2] -= 1 # B22
        # C: C12 (0,1)
        W[k_base*5 + i_base + 0][offset+2] += 1 # C12 is i=0, k=1?
        # C index is k*5 + i.
        # C12: i=0, k=1 -> idx 1.
        W[k_base*5 + i_base + 1][offset+2] += 1

        # Term 4: A22(B21-B11) -> C22
        # A: A22 (1,1)
        U[i_base*5 + j_base + 1][offset+3] += 1
        # B: B21 (1,0) - B11 (0,0)
        # B21: j=1, k=0 -> idx 5.
        # B11: j=0, k=0 -> idx 0.
        V[j_base*5 + k_base + 5][offset+3] += 1
        V[j_base*5 + k_base + 0][offset+3] -= 1
        # C: C22 (1,1)
        # C22: i=1, k=1 -> idx 6.
        W[k_base*5 + i_base + 6][offset+3] += 1

        # Term 5: (A11+A12)B22 -> C22? No, C12?
        # Strassen: C12 = M3 + M5.
        # M5 = (A11+A12)B22.
        # A: A11 (0,0), A12 (0,1)
        U[i_base*5 + j_base + 0][offset+4] += 1
        U[i_base*5 + j_base + 1][offset+4] += 1
        # B: B22 (1,1)
        V[j_base*5 + k_base + 6][offset+4] += 1
        # C: C12 (0,1)
        W[k_base*5 + i_base + 1][offset+4] += 1

        # Term 6: (A21-A11)(B11+B12) -> C22?
        # Strassen: C22 = M1 - M2 + M3 + M6.
        # M6 = (A21-A11)(B11+B12).
        # A: A21 (1,0) - A11 (0,0)
        U[i_base*5 + j_base + 0][offset+5] += 1
        U[i_base*5 + j_base + 0][offset+5] -= 1 # A11 is (0,0).
        # Wait, A21 is (1,0). A11 is (0,0).
        U[i_base*5 + j_base + 0][offset+5] += 1 # A21
        U[i_base*5 + j_base + 0][offset+5] -= 1 # A11
        # B: B11 (0,0) + B12 (0,1)
        V[j_base*5 + k_base + 0][offset+5] += 1
        V[j_base*5 + k_base + 1][offset+5] += 1
        # C: C22 (1,1)
        W[k_base*5 + i_base + 6][offset+5] += 1

        # Term 7: (A12-A22)(B21+B22) -> C11?
        # Strassen: C11 = M1 + M4 - M5 + M7.
        # M7 = (A12-A22)(B21+B22).
        # A: A12 (0,1) - A22 (1,1)
        U[i_base*5 + j_base + 1][offset+6] += 1
        U[i_base*5 + j_base + 1][offset+6] -= 1
        # B: B21 (1,0) + B22 (1,1)
        V[j_base*5 + k_base + 5][offset+6] += 1
        V[j_base*5 + k_base + 6][offset+6] += 1
        # C: C11 (0,0)
        W[k_base*5 + i_base + 0][offset+6] += 1
        
        return 7

    # Execute components
    # 1. A11 B11 (3x3x3)
    col_offset += add_schoolbook(3, 3, 3, 0, 0, 0, col_offset)
    
    # 2. A12 B21 (3x2x3)
    # A12: i in 0-2, j in 3-4
    # B21: j in 3-4, k in 0-2
    col_offset += add_schoolbook(3, 2, 3, 0, 3, 0, col_offset)
    
    # 3. A11 B12 (3x3x2)
    # A11: i in 0-2, j in 0-2
    # B12: j in 0-2, k in 3-4
    col_offset += add_schoolbook(3, 3, 2, 0, 0, 3, col_offset)
    
    # 4. A12 B22 (3x2x2)
    # A12: i in 0-2, j in 3-4
    # B22: j in 3-4, k in 3-4
    col_offset += add_schoolbook(3, 2, 2, 0, 3, 3, col_offset)
    
    # 5. A21 B11 (2x3x3)
    # A21: i in 3-4, j in 0-2
    # B11: j in 0-2, k in 0-2
    col_offset += add_schoolbook(2, 3, 3, 3, 0, 0, col_offset)
    
    # 6. A22 B21 (2x2x3)
    # A22: i in 3-4, j in 3-4
    # B21: j in 3-4, k in 0-2
    col_offset += add_schoolbook(2, 2, 3, 3, 3, 0, col_offset)
    
    # 7. A21 B12 (2x3x2)
    # A21: i in 3-4, j in 0-2
    # B12: j in 0-2, k in 3-4
    col_offset += add_schoolbook(2, 3, 2, 3, 0, 3, col_offset)
    
    # 8. A22 B22 (2x2x2) Strassen
    col_offset += add_strassen_2x2(3, 3, 3, col_offset)
    
    return {"U": U, "V": V, "W": W}
