"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    # Rank-26 decomposition of the 3x3x3 matrix multiplication tensor.
    # Uses Strassen's 7-term decomposition for the 2x2 sub-block (indices 0,1)
    # and 19 standard terms for interactions involving index 2.
    # Total rank: 7 + 19 = 26.
    
    # U: shape (9, 26) -> rows are (i*m+j) = i*3+j for i,j in 0..2
    # V: shape (9, 26) -> rows are (j*p+k) = j*3+k for j,k in 0..2
    # W: shape (9, 26) -> rows are (k*n+i) = k*3+i for k,i in 0..2
    #
    # Tensor: T[i,j,k] = U[i*3+j, c] * V[j*3+k, c] * W[k*3+i, c]
    # We need sum_c U[i*3+j, c] * V[j*3+k, c] * W[k*3+i, c] = 1 for all i,j,k
    #
    # Let's define the 26 terms.
    # Terms 0-6: Strassen for the 2x2 block (i in 0,1; j in 0,1; k in 0,1)
    # Let A be the 2x2 block of the first matrix (rows i=0,1; cols j=0,1)
    # Let B be the 2x2 block of the second matrix (rows j=0,1; cols k=0,1)
    # C = A*B is the 2x2 block of the result (rows i=0,1; cols k=0,1)
    #
    # Strassen:
    # m1 = (A00 + A11) * (B00 + B11)
    # m2 = (A10 + A11) * B00
    # m3 = A00 * (B01 - B11)
    # m4 = A11 * (B10 - B00)
    # m5 = (A00 + A01) * B11
    # m6 = (A10 - A00) * (B00 + B01)
    # m7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    #
    # So for the 2x2 block, the contributions to W are:
    # Term 0 (m1): U[0,0]=1, U[3,0]=1; V[0,0]=1, V[3,0]=1; W[0,0]=1, W[4,0]=1, W[1,0]=-1, W[3,0]=1
    # Wait, let's be careful with indices.
    # i,j,k in {0,1} for the core.
    # U[i*3+j, c], V[j*3+k, c], W[k*3+i, c]
    #
    # Let's define the 7 Strassen terms for c=0..6.
    # c=0: m1
    # U[0,0]=1 (i=0,j=0), U[3,0]=1 (i=1,j=1)
    # V[0,0]=1 (j=0,k=0), V[3,0]=1 (j=1,k=1)
    # W[0,0]=1 (k=0,i=0), W[4,0]=1 (k=1,i=1)
    # W[1,0]=-1 (k=0,i=1) -> C10? No, C10 = m2+m4. C00 = m1+m4-m5+m7.
    # Let's re-verify Strassen formulas.
    # C00 = m1 + m4 - m5 + m7
    # C01 = m3 + m5
    # C10 = m2 + m4
    # C11 = m1 - m2 + m3 + m6
    #
    # So for each term, we add to W[k*3+i].
    # m1: C00 += 1, C11 += 1 -> W[0,0] += 1, W[4,0] += 1
    # m2: C10 += 1 -> W[0,1] += 1 (k=0, i=1)
    # m3: C01 += 1 -> W[1,0] += 1 (k=1, i=0)
    # m4: C00 += 1, C10 += 1 -> W[0,0] += 1, W[0,1] += 1
    # m5: C00 -= 1, C01 += 1 -> W[0,0] -= 1, W[1,0] += 1
    # m6: C11 += 1 -> W[4,1] += 1 (k=1, i=1)
    # m7: C00 += 1, C11 += 1 -> W[0,0] += 1, W[4,1] += 1
    #
    # Let's build U, V, W for the 7 terms.
    n, m, p = 3, 3, 3
    rank = 26
    U = [[0]*rank for _ in range(9)]
    V = [[0]*rank for _ in range(9)]
    W = [[0]*rank for _ in range(9)]
    
    # Term 0: m1 = (A00+A11)(B00+B11)
    U[0][0] = 1; U[3][0] = 1  # i=0,j=0 and i=1,j=1
    V[0][0] = 1; V[3][0] = 1  # j=0,k=0 and j=1,k=1
    W[0][0] = 1; W[4][0] = 1  # k=0,i=0 and k=1,i=1
    
    # Term 1: m2 = (A10+A11)B00
    U[3][1] = 1; U[4][1] = 1  # i=1,j=0 and i=1,j=1
    V[0][1] = 1; V[3][1] = 1  # j=0,k=0 and j=1,k=0
    W[0][1] = 1               # k=0,i=1
    
    # Term 2: m3 = A00(B01-B11)
    U[0][2] = 1               # i=0,j=0
    V[1][2] = 1; V[4][2] = -1 # j=0,k=1 and j=1,k=1
    W[1][2] = 1               # k=1,i=0
    
    # Term 3: m4 = A11(B10-B00)
    U[3][3] = 1               # i=1,j=1
    V[3][3] = 1; V[0][3] = -1 # j=1,k=0 and j=0,k=0
    W[0][3] = 1; W[0][3] = 1  # Wait, C00 and C10 both get m4.
    # C00: k=0,i=0 -> W[0]
    # C10: k=0,i=1 -> W[3]
    W[0][3] = 1; W[3][3] = 1
    
    # Term 4: m5 = (A00+A01)B11
    U[0][4] = 1; U[1][4] = 1  # i=0,j=0 and i=0,j=1
    V[4][4] = 1               # j=1,k=1
    W[0][4] = -1; W[1][4] = 1 # C00 -= 1, C01 += 1
    
    # Term 5: m6 = (A10-A00)(B00+B01)
    U[3][5] = 1; U[0][5] = -1 # i=1,j=0 and i=0,j=0
    V[0][5] = 1; V[1][5] = 1  # j=0,k=0 and j=0,k=1
    W[4][5] = 1               # C11 += 1 -> k=1,i=1
    
    # Term 6: m7 = (A01-A11)(B10+B11)
    U[1][6] = 1; U[3][6] = -1 # i=0,j=1 and i=1,j=1
    V[3][6] = 1; V[4][6] = 1  # j=1,k=0 and j=1,k=1
    W[0][6] = 1; W[4][6] = 1  # C00 += 1, C11 += 1
    
    # Now the border terms. We need to handle all (i,j,k) where at least one index is 2.
    # There are 27 - 8 = 19 such terms.
    # We can use standard multiplication for these: U[i*3+j] = 1, V[j*3+k] = 1, W[k*3+i] = 1.
    # But we can also optimize? The known rank-23 decomposition exists, but rank-26 is safer.
    # Let's just add the 19 standard terms.
    
    c = 7
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if i == 0 and j == 0 and k == 0: continue
                if i < 2 and j < 2 and k < 2: continue # Already handled by Strassen
                U[i*3+j][c] = 1
                V[j*3+k][c] = 1
                W[k*3+i][c] = 1
                c += 1
                if c >= rank: break
            if c >= rank: break
        if c >= rank: break
    
    # Verify c == 26
    assert c == 26, f"Expected 26 terms, got {c}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
