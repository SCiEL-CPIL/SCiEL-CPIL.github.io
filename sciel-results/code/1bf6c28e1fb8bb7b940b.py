"""Matrix multiplication 2,5,5 with rank-49 decomposition using Strassen for the 2x2x2 block."""

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank: Strassen(2x2x2) + schoolbook(2x2x3) + schoolbook(2x3x2) + schoolbook(2x3x3)
    # = 7 + 12 + 12 + 18 = 49
    rank = 49
    
    U = [[0.0] * rank for _ in range(n * m)]  # 10 rows
    V = [[0.0] * rank for _ in range(m * p)]  # 25 rows
    W = [[0.0] * rank for _ in range(p * n)]  # 10 rows
    
    col = 0
    
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1} - Strassen (rank 7)
    # Strassen 2x2x2 formulas:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # Strassen U,V,W coefficients (local indices)
    # U row index: i*2+j (local 2x2)
    # V row index: j*2+k (local 2x2)
    # W row index: k*2+i (local 2x2, since n=2 for this block)
    
    # For each Strassen multiplication M_t, we have:
    # U[i*2+j, t] = coefficient of A[i,j] in M_t
    # V[j*2+k, t] = coefficient of B[j,k] in M_t
    # W[k*2+i, t] = coefficient of M_t in C[i,k]
    
    strassen_data = [
        # (i,j) -> U coeffs, (j,k) -> V coeffs, (k,i) -> W coeffs for each of 7 mults
        # M1: (A00+A11)*(B00+B11)
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00+=1, C11+=1
        {
            'U': {(0,0): 1, (1,1): 1},
            'V': {(0,0): 1, (1,1): 1},
            'W': {(0,0): 1, (1,1): 1}  # W[k,i]: k=0,i=0 -> C[0,0]; k=1,i=1 -> C[1,1]
        },
        # M2: (A10+A11)*B00
        {
            'U': {(1,0): 1, (1,1): 1},
            'V': {(0,0): 1},
            'W': {(0,1): 1}  # C[1,0] += M2, so W[k=0, i=1]
        },
        # M3: A00*(B01-B11)
        {
            'U': {(0,0): 1},
            'V': {(0,1): 1, (1,1): -1},
            'W': {(0,0): 1, (1,0): 1}  # C[0,0]+=M3? No: C00 = M1+M4-M5+M7, C01=M3+M5, C10=M2+M4, C11=M1-M2+M3+M6
        },
    ]
    
    # Let me redo this carefully.
    # Strassen:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # U coefficients (A side), V coefficients (B side), W coefficients (C side)
    # For block 1, local indices: i in {0,1}, j in {0,1}, k in {0,1}
    # Global: U row = i*5+j, V row = j*5+k, W row = k*2+i
    
    # I'll build this programmatically
    # Each term: list of (U_index, coeff), (V_index, coeff), (W_index, coeff)
    
    # M1: U: (0,0)=1,(1,1)=1 ; V: (0,0)=1,(1,1)=1 ; W: (0,0)=1,(1,1)=1
    terms = []
    # M1
    terms.append((
        [(0,0,1), (1,1,1)],  # U: i,j,coeff
        [(0,0,1), (1,1,1)],  # V: j,k,coeff
        [(0,0,1), (1,1,1)]   # W: k,i,coeff -> C[i,k]: (k=0,i=0) and (k=1,i=1)
    ))
    # M2
    terms.append((
        [(1,0,1), (1,1,1)],
        [(0,0,1)],
        [(0,1,1)]  # C[1,0] += M2, so k=0,i=1
    ))
    # M3
    terms.append((
        [(0,0,1)],
        [(0,1,1), (1,1,-1)],
        [(1,0,1), (0,1,1)]  # C[0,1]+=M3 -> k=1,i=0; C[1,1]+=M3 -> k=1,i=1
    ))
    # Wait, C01 = M3+M5 means C[i=0,k=1] += M3, so W[k=1, i=0] += 1
    # C11 = M1-M2+M3+M6 means C[i=1,k=1] += M3, so W[k=1, i=1] += 1
    terms[-1] = (
        [(0,0,1)],
        [(0,1,1), (1,1,-1)],
        [(1,0,1), (1,1,1)]
    )
    # M4
    terms.append((
        [(1,1,1)],
        [(1,0,1), (0,0,-1)],
        [(0,0,1), (0,1,1)]  # C[0,0]+=M4 -> W[0,0]; C[1,0]+=M4 -> W[0,1]
    ))
    # M5
    terms.append((
        [(0,0,1), (0,1,1)],
        [(1,1,1)],
        [(0,1,-1), (1,1,1)]  # C[0,1] -= M5 -> W[1,0] = -1; C[1,1] += M5 -> W[1,1] = 1
    ))
    # M6
    terms.append((
        [(1,0,1), (0,0,-1)],
        [(0,0,1), (0,1,1)],
        [(1,1,1)]  # C[1,1] += M6 -> W[1,1] += 1
    ))
    # M7
    terms.append((
        [(0,1,1), (1,1,-1)],
        [(0,1,1), (1,1,1)],
        [(0,0,1)]  # C[0,0] += M7 -> W[0,0] += 1
    ))
    
    # Now apply Strassen terms to Block 1
    # Global mapping for Block 1: i_global=i_local, j_global=j_local, k_global=k_local
    for u_terms, v_terms, w_terms in terms:
        for (i_l, j_l, coeff) in u_terms:
            i_g, j_g = i_l, j_l
            U[i_g * m + j_g][col] += coeff
        for (j_l, k_l, coeff) in v_terms:
            j_g, k_g = j_l, k_l
            V[j_g * p + k_g][col] += coeff
        for (k_l, i_l, coeff) in w_terms:
            k_g, i_g = k_l, i_l
            W[k_g * n + i_g][col] += coeff
        col += 1
    
    # Block 2: i in {0,1}, j in {0,1}, k in {2,3,4} - Schoolbook (rank 12 = 2*2*3)
    for i_l in range(2):
        for j_l in range(2):
            for k_l in range(3):
                i_g = i_l
                j_g = j_l
                k_g = k_l + 2
                U[i_g * m + j_g][col] = 1
                V[j_g * p + k_g][col] = 1
                W[k_g * n + i_g][col] = 1
                col += 1
    
    # Block 3: i in {0,1}, j in {2,3,4}, k in {0,1} - Schoolbook (rank 12 = 2*3*2)
    for i_l in range(2):
        for j_l in range(3):
            for k_l in range(2):
                i_g = i_l
                j_g = j_l + 2
                k_g = k_l
                U[i_g * m + j_g][col] = 1
                V[j_g * p + k_g][col] = 1
                W[k_g * n + i_g][col] = 1
                col += 1
    
    # Block 4: i in {0,1}, j in {2,3,4}, k in {2,3,4} - Schoolbook (rank 18 = 2*3*3)
    for i_l in range(2):
        for j_l in range(3):
            for k_l in range(3):
                i_g = i_l
                j_g = j_l + 2
                k_g = k_l + 2
                U[i_g * m + j_g][col] = 1
                V[j_g * p + k_g][col] = 1
                W[k_g * n + i_g][col] = 1
                col += 1
    
    assert col == rank, f"Column count {col} != rank {rank}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
