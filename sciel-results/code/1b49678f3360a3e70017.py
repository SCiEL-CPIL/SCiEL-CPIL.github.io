import numpy as np

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 decomposition
    # U2, V2, W2 are 4x7
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    
    W2 = np.array([
        [1, 0, -1, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Check W2 indices:
    # Row 0: k=0, i=0 (C11)
    # Row 1: k=0, i=1 (C12)
    # Row 2: k=1, i=0 (C21)
    # Row 3: k=1, i=1 (C22)
    
    # Verify W2 values from derivation:
    # Col 1: [1, 0, 0, 1] -> C11+, C22+. My W2 col 1 is [1,0,0,1]?
    # My W2 above:
    # Col 1: 1, 0, 0, 1. Correct.
    # Col 2: 0, 0, 1, -1. Correct.
    # Col 3: -1, 1, 0, 1. Correct.
    # Col 4: 1, 0, 1, 0. Correct.
    # Col 5: -1, 1, 0, 0. Correct.
    # Col 6: 0, 0, 0, 1. Correct.
    # Col 7: 1, 0, 0, 0. Correct.
    
    # The W2 array in the code above is transposed?
    # U2, V2, W2 are defined as rows.
    # U2 row 0: 1,0,1,0,1,-1,0.
    # My derived U2 col 1: [1,0,0,1]. So U2[0,0]=1, U2[3,0]=1.
    # In the array above, U2[0] is the first row.
    # U2[0] = [1, 0, 1, 0, 1, -1, 0]
    # This means U2[0,0]=1, U2[0,2]=1, U2[0,4]=1, U2[0,5]=-1.
    # This corresponds to A11 being involved in terms 1, 3, 5, 6.
    # Term 1: A11+A22. Yes.
    # Term 3: A11. Yes.
    # Term 5: A11+A12. Yes.
    # Term 6: A21-A11. Yes.
    # So U2 is correct.
    
    # V2 row 0: 1,1,0,-1,0,1,0
    # V2[0,0]=1, V2[0,1]=1, V2[0,3]=-1, V2[0,5]=1.
    # B11 involved in 1, 2, 4, 6.
    # Term 1: B11+B22. Yes.
    # Term 2: B11. Yes.
    # Term 4: B21-B11. Yes.
    # Term 6: B11+B12. Yes.
    # Correct.
    
    # W2 row 0: 1,0,-1,1,-1,0,1
    # W2[0,0]=1, W2[0,2]=-1, W2[0,3]=1, W2[0,4]=-1, W2[0,6]=1.
    # C11 involved in 1, 3, 4, 5, 7.
    # Term 1: +M1. Yes.
    # Term 3: -M3. Yes.
    # Term 4: +M4. Yes.
    # Term 5: -M5. Yes.
    # Term 7: +M7. Yes.
    # Correct.

    # Construct 4x4x4 decomposition via Kronecker product
    # U4 = U2 (x) U2
    # The Kronecker product of two matrices A (n1 x m1) and B (n2 x m2) is (n1*n2 x m1*m2).
    # np.kron(A, B)
    U4 = np.kron(U2, U2)
    V4 = np.kron(V2, V2)
    W4 = np.kron(W2, W2)
    
    # U4 shape: 16x49
    # V4 shape: 16x49
    # W4 shape: 16x49
    
    # Now construct for 4x4x8
    # Rank 98
    U = np.zeros((16, 98))
    V = np.zeros((32, 98))
    W = np.zeros((32, 98))
    
    # Block 1: columns 0-48
    U[:, 0:49] = U4
    W[:, 0:49] = W4
    # V for Block 1
    # V rows are 8j + k. k in 0..3.
    # V4 rows are 4j + k.
    for j in range(4):
        for k in range(4):
            V[8*j + k, 0:49] = V4[4*j + k, :]
            
    # Block 2: columns 49-97
    U[:, 49:98] = U4
    # W for Block 2
    # W rows are 4k + i. k in 4..7.
    # Let k' = k - 4.
    # W4 rows are 4k' + i.
    for k in range(4, 8):
        k_prime = k - 4
        for i in range(4):
            W[4*k + i, 49:98] = W4[4*k_prime + i, :]
            
    # V for Block 2
    # V rows are 8j + k. k in 4..7.
    # Let k' = k - 4.
    # V4 rows are 4j + k'.
    for j in range(4):
        for k in range(4, 8):
            k_prime = k - 4
            V[8*j + k, 49:98] = V4[4*j + k_prime, :]

    # Convert to lists of lists of floats/ints
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
