import numpy as np

DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Total rank estimate: 7 + 7 + 4 + 10 = 28
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper to add a term
    def add_term(i, j, k, u_val, v_val, w_val):
        nonlocal col
        U[i * m + j, col] = u_val
        V[j * p + k, col] = v_val
        W[k * n + i, col] = w_val
        col += 1

    # Helper to add Strassen block for A[i1:i2, j1:j2] and B[j1:j2, k1:k2]
    # Assumes i1=0, i2=2, j1, j2, k1, k2
    def add_strassen(i1, i2, j1, j2, k1, k2):
        # i1, i2 are indices in A rows (0, 1)
        # j1, j2 are indices in A cols / B rows (0, 1)
        # k1, k2 are indices in B cols (0, 1)
        
        # M1 = (A00 + A11)(B00 + B11)
        add_term(i1, j1, k1, 1, 1, 1) # A00 B00
        add_term(i1, j2, k1, 1, 1, 1) # A01 B00 -> Wait, Strassen uses linear combos.
        
        # This manual term-by-term addition is error-prone.
        # Better to construct the U, V, W vectors for the 7 terms directly.
        
        # Term 1: (A00+A11)(B00+B11)
        # U: A00, A11. V: B00, B11. W: C00, C11 (with coeffs)
        # C00 += M1, C11 += M1
        # M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
        # This expansion is 4 terms. But Strassen reuses.
        # The decomposition is:
        # M1 = (A00+A11)(B00+B11)
        # We need to represent the linear forms.
        
        # Let's use the standard Strassen coefficients for U, V, W.
        # r=0: M1
        # U: A00=1, A11=1. V: B00=1, B11=1. W: C00=1, C11=1.
        # Note: C00 and C11 are at indices k*n+i.
        # C00 index: 0*n+0=0. C11 index: 1*n+1=3 (if n=2).
        
        # We will add 7 terms for this block.
        
        # 1. M1 = (A00+A11)(B00+B11)
        # U: A00=1, A11=1
        # V: B00=1, B11=1
        # W: C00=1, C11=1
        U[i1*m+j1, col] = 1; V[j1*p+k1, col] = 1; W[k1*n+i1, col] = 1
        U[i1*m+j2, col] = 1; V[j1*p+k2, col] = 1; W[k1*n+i1, col] = 1
        U[i2*m+j1, col] = 1; V[j2*p+k1, col] = 1; W[k2*n+i2, col] = 1
        U[i2*m+j2, col] = 1; V[j2*p+k2, col] = 1; W[k2*n+i2, col] = 1
        # Wait, this adds 4 entries to U/V/W for ONE rank index?
        # No, one rank index corresponds to ONE multiplication of two linear forms.
        # The linear forms are sums.
        # U vector for this rank index has 1s at A00, A11.
        # V vector has 1s at B00, B11.
        # W vector has 1s at C00, C11.
        # So for this col:
        U[i1*m+j1, col] = 1
        U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1
        V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1
        W[k2*n+i2, col] = 1
        col += 1
        
        # 2. M2 = (A10+A11)B00
        U[i2*m+j1, col] = 1
        U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1
        W[k1*n+i2, col] = 1 # C10
        W[k2*n+i2, col] = 1 # C11? No, M2 contributes to C10 and C11?
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # So M2 contributes to C10 (coeff 1) and C11 (coeff -1).
        W[k1*n+i2, col] = 1
        W[k2*n+i2, col] = -1
        col += 1

        # 3. M3 = A00(B01-B11)
        U[i1*m+j1, col] = 1
        V[j1*p+k2, col] = 1
        V[j2*p+k2, col] = -1
        W[k2*n+i1, col] = 1 # C01
        W[k2*n+i2, col] = 1 # C11
        col += 1

        # 4. M4 = A11(B10-B00)
        U[i2*m+j2, col] = 1
        V[j2*p+k1, col] = 1
        V[j1*p+k1, col] = -1
        W[k1*n+i1, col] = 1 # C00
        W[k1*n+i2, col] = 1 # C10
        col += 1

        # 5. M5 = (A00+A01)B11
        U[i1*m+j1, col] = 1
        U[i1*m+j2, col] = 1
        V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = -1 # C00
        W[k2*n+i1, col] = 1 # C01
        col += 1

        # 6. M6 = (A10-A00)(B00+B01)
        U[i2*m+j1, col] = 1
        U[i1*m+j1, col] = -1
        V[j1*p+k1, col] = 1
        V[j1*p+k2, col] = 1
        W[k2*n+i2, col] = 1 # C11
        col += 1

        # 7. M7 = (A01-A11)(B10+B11)
        U[i1*m+j2, col] = 1
        U[i2*m+j2, col] = -1
        V[j2*p+k1, col] = 1
        V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1 # C00
        col += 1

    # Block 1: A[0:2, 0:2] * B[0:2, 0:2]
    add_strassen(0, 1, 0, 1, 0, 1)
    
    # Block 2: A[0:2, 0:2] * B[0:2, 2:4]
    add_strassen(0, 1, 0, 1, 2, 3)
    
    # Block 3: A[0:2, 0:2] * B[0:2, 4:5] (Schoolbook, Rank 4)
    # i in 0,1; j in 0,1; k=4
    for i in range(2):
        for j in range(2):
            add_term(i, j, 4, 1, 1, 1)
            
    # Block 4: A[0:2, 2:3] * B[2:3, 0:5] (Schoolbook, Rank 10)
    # i in 0,1; j=2; k in 0..4
    for i in range(2):
        for k in range(5):
            add_term(i, 2, k, 1, 1, 1)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
