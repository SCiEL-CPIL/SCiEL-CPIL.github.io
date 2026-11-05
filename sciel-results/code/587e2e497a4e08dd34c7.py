"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 34 decomposition
    # Block 1: C[0:2, 0:2] -> M[2,3,2] (Rank 11)
    # Block 2: C[0:2, 2:4] -> M[2,3,2] (Rank 11)
    # Block 3: C[2, 0:4]   -> M[1,3,4] (Rank 12)
    
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        nonlocal col
        U[u_idx][col] += coeff
        V[v_idx][col] += coeff
        W[w_idx][col] += coeff
        col += 1

    # --- Block 1: C[0:2, 0:2] = A[0:2, 0:3] * B[0:3, 0:2] ---
    # Decomposed into M[2,1,2] (Rank 4) + M[2,2,2] (Rank 7)
    # M[2,1,2]: A is 2x1 (col 0 of A[0:2,0:3]), B is 1x2 (row 0 of B[0:3,0:2])
    # A_part1: rows 0,1; col 0. Indices: (0,0), (1,0) -> U idx 0, 3
    # B_part1: row 0; cols 0,1. Indices: (0,0), (0,1) -> V idx 0, 1
    # C_part1: rows 0,1; cols 0,1. Indices: (0,0), (0,1), (1,0), (1,1) -> W idx 0, 1, 4, 5
    
    # Term 1: A[0,0] * B[0,0] -> C[0,0]
    add_term(0, 0, 0)
    # Term 2: A[0,0] * B[0,1] -> C[0,1]
    add_term(0, 1, 1)
    # Term 3: A[1,0] * B[0,0] -> C[1,0]
    add_term(3, 0, 4)
    # Term 4: A[1,0] * B[0,1] -> C[1,1]
    add_term(3, 1, 5)
    
    # M[2,2,2]: A is 2x2 (cols 1,2 of A[0:2,0:3]), B is 2x2 (rows 1,2 of B[0:3,0:2])
    # A_part2: rows 0,1; cols 1,2. 
    #   A[0,1] idx 1, A[0,2] idx 2
    #   A[1,1] idx 4, A[1,2] idx 5
    # B_part2: rows 1,2; cols 0,1.
    #   B[1,0] idx 4, B[1,1] idx 5
    #   B[2,0] idx 8, B[2,1] idx 9
    # C_part2: rows 0,1; cols 0,1.
    #   C[0,0] idx 0, C[0,1] idx 1
    #   C[1,0] idx 4, C[1,1] idx 5
    
    # Strassen M1..M7
    # M1 = (A00+A11)(B00+B11)
    # A00=1, A11=4, B00=4, B11=9
    add_term(1, 4, 0)  # A00 B00 -> C00
    add_term(1, 9, 0)  # A00 B11 -> C00
    add_term(4, 4, 0)  # A11 B00 -> C00
    add_term(4, 9, 0)  # A11 B11 -> C00
    
    # M2 = (A10+A11)B00
    # A10=4, A11=5? No, A10 is A[1,1] idx 4, A11 is A[1,2] idx 5?
    # Wait, local indices for Strassen:
    # Local A: [[A00, A01], [A10, A11]]
    # Global A[0,1]=1, A[0,2]=2, A[1,1]=4, A[1,2]=5
    # So Local A00=1, A01=2, A10=4, A11=5
    # Local B: [[B00, B01], [B10, B11]]
    # Global B[1,0]=4, B[1,1]=5, B[2,0]=8, B[2,1]=9
    # So Local B00=4, B01=5, B10=8, B11=9
    
    # Reset col? No, col is global.
    # I need to be careful with the "add_term" logic. It adds to the same column.
    # Strassen requires separate columns for each multiplication.
    # My previous add_term calls for M[2,1,2] used cols 0-3.
    # Now I am at col 4.
    
    # M1: (A00+A11)(B00+B11)
    # (1+4) * (4+9) -> 5*13
    # U: 1, 4. V: 4, 9. W: 0.
    # This is not a single term, it's a sum.
    # Strassen terms are products of sums.
    # (A00+A11) is a linear combo.
    # We need to expand:
    # (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
    # These are 4 separate scalar multiplications in the tensor decomposition if we don't combine them into a single "rank 1" term?
    # No, Rank is the number of terms.
    # A term is $u \otimes v \otimes w$.
    # If $u$ has two non-zero entries, it's still one term.
    # So $u = e_1 + e_4$, $v = e_4 + e_9$, $w = e_0$.
    # This is ONE term.
    
    # So for M1:
    # U[1][col]=1, U[4][col]=1
    # V[4][col]=1, V[9][col]=1
    # W[0][col]=1
    # And we need to subtract/add for the other C entries.
    # C00 += M1
    # C11 -= M1
    # So W[0][col]=1, W[5][col]=-1
    
    # Let's implement a helper for Strassen terms.
    
    def add_strassen_term(u1, u2, v1, v2, w1, w2, coeff_w1=1, coeff_w2=0):
        nonlocal col
        U[u1][col] = 1
        if u2 is not None:
            U[u2][col] = 1
        V[v1][col] = 1
        if v2 is not None:
            V[v2][col] = 1
        W[w1][col] = coeff_w1
        if w2 is not None:
            W[w2][col] = coeff_w2
        col += 1

    # M1: (A00+A11)(B00+B11) -> C00 += M1, C11 -= M1
    # Local A00=1, A11=5. Local B00=4, B11=9.
    # Global U: 1, 5. Global V: 4, 9. Global W: 0, 5.
    add_strassen_term(1, 5, 4, 9, 0, 5, 1, -1)
    
    # M2: (A10+A11)B00 -> C10 += M2
    # Local A10=4, A11=5. Local B00=4.
    add_strassen_term(4, 5, 4, None, 4, None, 1, 0)
    
    # M3: A00(B01-B11) -> C01 += M3
    # Local A00=1. Local B01=5, B11=9.
    add_strassen_term(1, None, 5, 9, 1, None, 1, 0)
    # Wait, B01-B11 means V[5]=1, V[9]=-1.
    # My helper sets V[v2]=1. I need to handle signs.
    
    # Let's rewrite the helper to take coefficients.
    
    # Re-do Block 1 M[2,2,2] part properly.
    # Current col is 4.
    
    # M1
    U[1][4]=1; U[5][4]=1; V[4][4]=1; V[9][4]=1; W[0][4]=1; W[5][4]=-1; col+=1
    # M2
    U[4][5]=1; U[5][5]=1; V[4][5]=1; W[4][5]=1; col+=1
    # M3
    U[1][6]=1; V[5][6]=1; V[9][6]=-1; W[1][6]=1; col+=1
    # M4
    U[5][7]=1; V[8][7]=1; V[4][7]=-1; W[4][7]=1; W[0][7]=-1; col+=1
    # M5
    U[1][8]=1; U[2][8]=1; V[9][8]=1; W[1][8]=1; col+=1
    # M6
    U[4][9]=1; U[1][9]=-1; V[4][9]=1; V[5][9]=1; W[0][9]=1; col+=1
    # M7
    U[2][10]=1; U[5][10]=-1; V[8][10]=1; V[9][10]=1; W[5][10]=1; col+=1
    
    # --- Block 2: C[0:2, 2:4] = A[0:2, 0:3] * B[0:3, 2:4] ---
    # Same structure, but B cols are 2,3.
    # B indices:
    # B[0,2]=2, B[0,3]=3
    # B[1,2]=6, B[1,3]=7
    # B[2,2]=10, B[2,3]=11
    # C indices:
    # C[0,2]=2, C[0,3]=3
    # C[1,2]=6, C[1,3]=7
    
    # M[2,1,2] part:
    # A col 0 (idx 0,3). B row 0 (idx 2,3).
    # C[0,2]=2, C[0,3]=3, C[1,2]=6, C[1,3]=7.
    
    add_term(0, 2, 2)
    add_term(0, 3, 3)
    add_term(3, 2, 6)
    add_term(3, 3, 7)
    
    # M[2,2,2] part:
    # A cols 1,2 (idx 1,2,4,5).
    # B rows 1,2 (idx 6,7,10,11).
    # C cols 2,3 (idx 2,3,6,7).
    
    # M1: (A00+A11)(B00+B11) -> C[0,2]+=, C[1,3]-=
    # Local A00=1, A11=5. Local B00=6, B11=11.
    U[1][18]=1; U[5][18]=1; V[6][18]=1; V[11][18]=1; W[2][18]=1; W[7][18]=-1; col+=1
    # M2: (A10+A11)B00 -> C[1,2]+=
    # Local A10=4, A11=5. Local B00=6.
    U[4][19]=1; U[5][19]=1; V[6][19]=1; W[6][19]=1; col+=1
    # M3: A00(B01-B11) -> C[0,3]+=
    # Local A00=1. Local B01=7, B11=11.
    U[1][20]=1; V[7][20]=1; V[11][20]=-1; W[3][20]=1; col+=1
    # M4: A11(B10-B00) -> C[1,2]+=, C[0,2]-=
    # Local A11=5. Local B10=10, B00=6.
    U[5][21]=1; V[10][21]=1; V[6][21]=-1; W[6][21]=1; W[2][21]=-1; col+=1
    # M5: (A00+A01)B11 -> C[0,3]+=
    # Local A00=1, A01=2. Local B11=11.
    U[1][22]=1; U[2][22]=1; V[11][22]=1; W[3][22]=1; col+=1
    # M6: (A10-A00)(B00+B01) -> C[0,2]+=
    # Local A10=4, A00=1. Local B00=6, B01=7.
    U[4][23]=1; U[1][23]=-1; V[6][23]=1; V[7][23]=1; W[2][23]=1; col+=1
    # M7: (A01-A11)(B10+B11) -> C[1,3]+=
    # Local A01=2, A11=5. Local B10=10, B11=11.
    U[2][24]=1; U[5][24]=-1; V[10][24]=1; V[11][24]=1; W[7][24]=1; col+=1
    
    # --- Block 3: C[2, 0:4] = A[2, 0:3] * B[0:3, 0:4] ---
    # M[1,3,4]. Rank 12.
    # A row 2: idx 6,7,8.
    # B: all 12 entries.
    # C row 2: idx 8,9,10,11.
    
    for j in range(3):
        for k in range(4):
            u_idx = 6 + j
            v_idx = j * 4 + k
            w_idx = 8 + k
            add_term(u_idx, v_idx, w_idx)
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
